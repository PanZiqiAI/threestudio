import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

from argparse import Namespace
from custom_pkg.io.config import ConfigArgs


@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False

    rays_d_normalize: bool = True


class SingleImageDataBase:
    """ 数据集基类. """
    def __init__(self, cfg, phase):
        self.cfg: SingleImageDataModuleConfig = cfg
        # Config.
        self._phase = phase
        self._cargs = ConfigArgs(**self.cfg)
        self._cargs.heights = [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        delattr(self._cargs, 'height')
        self._cargs.widths = [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        delattr(self._cargs, 'width')
        assert len(self._cargs.heights) == len(self._cargs.widths)

        # --------------------------------------------------------------------------------------------------------------
        # 添加RandomCamera数据集.
        # --------------------------------------------------------------------------------------------------------------
        if self._cargs.use_random_camera:
            random_camera_cfg = parse_structured(RandomCameraDataModuleConfig, self.cfg.get("random_camera", {}))
            if phase == "train":
                self._random_camera_generator = RandomCameraIterableDataset(random_camera_cfg)
            else:
                self._random_camera_generator = RandomCameraDataset(random_camera_cfg, phase)

        # --------------------------------------------------------------------------------------------------------------
        # 相机位置.
        # --------------------------------------------------------------------------------------------------------------
        self._elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        elevation = self._elevation_deg * math.pi / 180
        self._azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        azimuth = self._azimuth_deg * math.pi / 180
        self._camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])
        self._camera_position: Float[Tensor, "1 3"] = torch.stack([
            self._camera_distance * torch.cos(elevation) * torch.cos(azimuth),
            self._camera_distance * torch.cos(elevation) * torch.sin(azimuth),
            self._camera_distance * torch.sin(elevation)], dim=-1)
        # --------------------------------------------------------------------------------------------------------------
        # 变换矩阵.
        # --------------------------------------------------------------------------------------------------------------
        center = torch.zeros_like(self._camera_position)
        lookat = F.normalize(center - self._camera_position, dim=-1)
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w = torch.cat([torch.stack([right, up, -lookat], dim=-1), self._camera_position[:, :, None]], dim=-1)
        c2w = torch.cat([c2w, torch.zeros_like(c2w[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0
        self._c2w = c2w
        # --------------------------------------------------------------------------------------------------------------
        # 其它.
        # --------------------------------------------------------------------------------------------------------------
        self._directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self._cargs.heights, self._cargs.widths)]
        self._fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))
        self._focal_lengths = [0.5 * height / torch.tan(0.5 * self._fovy) for height in self._cargs.heights]
        self._light_position = self._camera_position

        """ 当前状态. """
        self._status = Namespace(
            height=self._cargs.heights[0], width=self._cargs.widths[0], prev_height=self._cargs.heights[0],
            directions_unit_focal=self._directions_unit_focals[0], focal_length=self._focal_lengths[0],
            rays_o=None, rays_d=None, mvp_matrix=None, rgb=None, mask=None, depth=None, Normal=None)
        self._update_rays()
        self._update_images()

    def get_all_images(self):
        return self._status.rgb

    def _update_rays(self):
        # Get directions by dividing directions_unit_focal by focal length
        directions = self._status.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self._status.focal_length
        self._status.rays_o, self._status.rays_d = get_rays(
            directions, self._c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize)
        proj_mtx = get_projection_matrix(
            self._fovy, self._status.width / self._status.height, 0.1, 100.0)
        self._status.mvp_matrix = get_mvp_matrix(self._c2w, proj_mtx)

    def _update_images(self):
        # load image
        assert os.path.exists(self.cfg.image_path), f"Could not find image {self.cfg.image_path}!"
        rgba = cv2.cvtColor(cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        rgba = cv2.resize(rgba, (self._status.width, self._status.height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rgb = rgba[..., :3]
        self._status.rgb = torch.from_numpy(rgb).unsqueeze(0).contiguous().to(get_rank())
        self._status.mask = torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(get_rank())
        print(f"[INFO] single image dataset: load image {self.cfg.image_path} {self._status.rgb.shape}")

        # load depth
        if self.cfg.requires_depth:
            depth_path = self.cfg.image_path.replace("_rgba.png", "_depth.png")
            assert os.path.exists(depth_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(depth, (self._status.width, self._status.height), interpolation=cv2.INTER_AREA)
            self._status.depth = torch.from_numpy(depth.astype(np.float32) / 255.0).unsqueeze(0).to(get_rank())
            print(f"[INFO] single image dataset: load depth {depth_path} {self._status.depth.shape}")
        else:
            self._status.depth = None

        # load normal
        if self.cfg.requires_normal:
            normal_path = self.cfg.image_path.replace("_rgba.png", "_normal.png")
            assert os.path.exists(normal_path)
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = cv2.resize(normal, (self._status.width, self._status.height), interpolation=cv2.INTER_AREA)
            self._status.normal = torch.from_numpy(normal.astype(np.float32) / 255.0).unsqueeze(0).to(get_rank())
            print(f"[INFO] single image dataset: load normal {normal_path} {self._status.normal.shape}")
        else:
            self._status.normal = None

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right([-1] + self._cargs.resolution_milestones, global_step) - 1
        self._status.height = self._cargs.heights[size_ind]
        if self._status.height == self._status.prev_height:
            return

        self._status.prev_height = self._status.height
        self._status.width = self._cargs.widths[size_ind]
        self._status.directions_unit_focal = self._directions_unit_focals[size_ind]
        self._status.focal_length = self._focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self._status.height}, width: {self._status.width}")
        self._update_rays()
        self._update_images()


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self._status.rays_o,
            "rays_d": self._status.rays_d,
            "mvp_mtx": self._status.mvp_matrix,
            "camera_positions": self._camera_position,
            "light_positions": self._light_position,
            "elevation": self._elevation_deg,
            "azimuth": self._azimuth_deg,
            "camera_distances": self._camera_distance,
            "rgb": self._status.rgb,
            "ref_depth": self._status.depth,
            "ref_normal": self._status.normal,
            "mask": self._status.mask,
            "height": self._status.height,
            "width": self._status.width,
            "c2w": self._c2w,
            "fovy": self._fovy,
        }
        if self._cargs.use_random_camera:
            batch["random_camera"] = self._random_camera_generator.get_batch_data() if self._phase == 'train' else \
                self._random_camera_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self._random_camera_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __len__(self):
        return len(self._random_camera_generator)

    def __getitem__(self, index):
        return self._random_camera_generator[index]


@register("single-image-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn)

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.train_dataset.collate)

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
