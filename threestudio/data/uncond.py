import bisect
import math
import random
from argparse import Namespace
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import (get_mvp_matrix, get_projection_matrix, get_ray_directions, get_rays)
from threestudio.utils.typing import *

from custom_pkg.io.config import ConfigArgs


@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (40, 70)  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy

    rays_d_normalize: bool = True


class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self._cargs = ConfigArgs(**cfg)
        self._cargs.resolution_milestones = cfg.resolution_milestones
        self._cargs.distance_range = cfg.camera_distance_range
        delattr(self._cargs, 'camera_distance_range')
        self._cargs.position_perturb = cfg.camera_perturb
        delattr(self._cargs, 'camera_perturb')

        self.cfg = cfg
        self._cargs.heights = [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        delattr(self._cargs, 'height')
        self._cargs.widths = [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        delattr(self._cargs, 'width')
        self._cargs.batch_sizes = [self.cfg.batch_size] if isinstance(self.cfg.batch_size, int) else self.cfg.batch_size
        delattr(self._cargs, 'batch_size')
        delattr(self._cargs, 'rays_d_normalize')
        # --------------------------------------------------------------------------------------------------------------
        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0) for (height, width) in zip(self._cargs.heights, self._cargs.widths)]
        """ 当前状态. """
        self._status = Namespace(
            height=self._cargs.heights[0], width=self._cargs.widths[0], batch_size=self._cargs.batch_sizes[0],
            elevation_range=self._cargs.elevation_range, azimuth_range=self._cargs.azimuth_range,
            directions_unit_focal=self.directions_unit_focals[0], fovy=None, proj_matrix=None)

    def __iter__(self):
        while True:
            yield self.get_batch_data()

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right([-1] + self._cargs.resolution_milestones, global_step) - 1
        self._status.height = self._cargs.heights[size_ind]
        self._status.width = self._cargs.widths[size_ind]
        self._status.batch_size = self._cargs.batch_sizes[size_ind]
        self._status.directions_unit_focal = self.directions_unit_focals[size_ind]
        # progressive view
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self._status.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1]]
        self._status.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0], (1 - r) * 0.0 + r * self.cfg.azimuth_range[1]]

    def get_batch_data(self):
        ################################################################################################################
        """ 相机位置参数. """
        ################################################################################################################
        # --------------------------------------------------------------------------------------------------------------
        # (1) Elevation角度.
        # --------------------------------------------------------------------------------------------------------------
        if random.random() < 0.5:
            elevation_deg = torch.rand(self._status.batch_size) * (self._status.elevation_range[1] - self._status.elevation_range[0]) \
                            + self._status.elevation_range[0]
            elevation = elevation_deg * math.pi / 180
        else:
            elevation_range = [self._status.elevation_range[0] / 180.0 * math.pi, self._status.elevation_range[1] / 180.0 * math.pi]
            # inverse transform sampling
            elevation = torch.asin(
                torch.rand(self._status.batch_size) * (math.sin(elevation_range[1]) - math.sin(elevation_range[0]))
                + math.sin(elevation_range[0]))
            elevation_deg = elevation / math.pi * 180.0
        # --------------------------------------------------------------------------------------------------------------
        # (2) Azimuth角度.
        # --------------------------------------------------------------------------------------------------------------
        if self.cfg.batch_uniform_azimuth:
            azimuth_deg = (torch.rand(self._status.batch_size) + torch.arange(self._status.batch_size)) / self._status.batch_size * \
                          (self._status.azimuth_range[1] - self._status.azimuth_range[0]) + self._status.azimuth_range[0]
        else:
            azimuth_deg = torch.rand(self._status.batch_size) * (self._status.azimuth_range[1] - self._status.azimuth_range[0]) \
                          + self._status.azimuth_range[0]
        azimuth = azimuth_deg * math.pi / 180
        # --------------------------------------------------------------------------------------------------------------
        # (3) 相机距离.
        # --------------------------------------------------------------------------------------------------------------
        camera_distances = torch.rand(self._status.batch_size) * (self._cargs.distance_range[1] - self._cargs.distance_range[0]) \
                           + self._cargs.distance_range[0]
        camera_positions = torch.stack(
            [camera_distances * torch.cos(elevation) * torch.cos(azimuth),
             camera_distances * torch.cos(elevation) * torch.sin(azimuth),
             camera_distances * torch.sin(elevation)], dim=-1)
        camera_perturb = torch.rand(self._status.batch_size, 3) * 2 * self._cargs.position_perturb - self._cargs.position_perturb
        camera_positions = camera_positions + camera_perturb

        ################################################################################################################
        """ 变换矩阵. """
        ################################################################################################################
        # --------------------------------------------------------------------------------------------------------------
        # c2w. 从相机到世界坐标系的变换.
        # --------------------------------------------------------------------------------------------------------------
        center = torch.randn(self._status.batch_size, 3) * self._cargs.center_perturb
        lookat = F.normalize(center - camera_positions, dim=-1)
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self._status.batch_size, 1)
        up = up + torch.randn(self._status.batch_size, 3) * self._cargs.up_perturb
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
        c2w = torch.cat([c2w, torch.zeros_like(c2w[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        # --------------------------------------------------------------------------------------------------------------
        # mvp矩阵. 从世界坐标系到相机坐标系成像平面的变换.
        # --------------------------------------------------------------------------------------------------------------
        fovy_deg = torch.rand(self._status.batch_size) * (self._cargs.fovy_range[1] - self._cargs.fovy_range[0]) + self._cargs.fovy_range[0]
        self._status.fovy = fovy_deg * math.pi / 180
        self._status.proj_matrix = get_projection_matrix(self._status.fovy, self._status.width/self._status.height, 0.01, 100.0)  # FIXME: hard-coded near and far
        mvp_mtx = get_mvp_matrix(c2w, self._status.proj_matrix)

        ################################################################################################################
        """ 光线相关. """
        ################################################################################################################
        light_distances = torch.rand(self._status.batch_size) * (self._cargs.light_distance_range[1] - self._cargs.light_distance_range[0]) \
                          + self._cargs.light_distance_range[0]
        # --------------------------------------------------------------------------------------------------------------
        # DreamFusion
        # --------------------------------------------------------------------------------------------------------------
        if self._cargs.light_sample_strategy == "dreamfusion":
            camera_directions = F.normalize(
                camera_positions + torch.randn(self._status.batch_size, 3) * self._cargs.light_position_perturb, dim=-1)
            light_positions = camera_directions * light_distances[:, None]
        # --------------------------------------------------------------------------------------------------------------
        # magic3d
        # --------------------------------------------------------------------------------------------------------------
        elif self._cargs.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack([local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])], dim=-1), dim=-1)
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (torch.rand(self._status.batch_size) * math.pi * 2 - math.pi)  # [-pi, pi]
            light_elevation = (torch.rand(self._status.batch_size) * math.pi / 3 + math.pi / 6)  # [pi/6, pi/2]
            light_positions_local = torch.stack([
                light_distances * torch.cos(light_elevation) * torch.cos(light_azimuth),
                light_distances * torch.cos(light_elevation) * torch.sin(light_azimuth),
                light_distances * torch.sin(light_elevation)], dim=-1)
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(f"Unknown light sample strategy: {self._cargs.light_sample_strategy}")
        # --------------------------------------------------------------------------------------------------------------
        # 源点和方向.
        # --------------------------------------------------------------------------------------------------------------
        focal_length = 0.5 * self._status.height / torch.tan(0.5 * self._status.fovy)
        directions = self._status.directions_unit_focal[None, :, :, :].repeat(self._status.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)

        # Return.
        return {
            "height": self._status.height, "width": self._status.width,
            # 相机位置相关.
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "camera_positions": camera_positions,
            # 变换矩阵相关.
            "c2w": c2w, "proj_mtx": self._status.proj_matrix, "mvp_mtx": mvp_mtx, "fovy": self._status.fovy,
            # 光线相关
            "light_positions": light_positions, "rays_o": rays_o, "rays_d": rays_d
        }


class RandomCameraDataset(Dataset):

    def __init__(self, cfg, split) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self._phase = split

        self._cargs = ConfigArgs(**cfg)

        self._n_view = getattr(self._cargs, f"n_{split}_views")

        self.azimuth_deg = torch.linspace(0, 360.0, self._n_view + 1)[: self._n_view] if self._phase == "val" else \
            torch.linspace(0, 360.0, self._n_view)
        azimuth = self.azimuth_deg * math.pi / 180
        self.elevation_deg = torch.full_like(azimuth, self.cfg.eval_elevation_deg)
        elevation = self.elevation_deg * math.pi / 180
        self.camera_distances = torch.full_like(azimuth, self.cfg.eval_camera_distance)
        self.camera_positions = torch.stack([
            self.camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            self.camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            self.camera_distances * torch.sin(elevation)], dim=-1)

        center = torch.zeros_like(self.camera_positions)
        lookat = F.normalize(center - self.camera_positions, dim=-1)
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.cfg.eval_batch_size, 1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w = torch.cat([torch.stack([right, up, -lookat], dim=-1), self.camera_positions[:, :, None]], dim=-1)
        c2w = torch.cat([c2w, torch.zeros_like(c2w[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0
        self.c2w = c2w

        fovy_deg = torch.full_like(self.elevation_deg, self.cfg.eval_fovy_deg)
        self.fovy = fovy_deg * math.pi / 180
        self.proj_mtx = get_projection_matrix(self.fovy, self.cfg.eval_width/self.cfg.eval_height, 0.01, 100.0)  # FIXME: hard-coded near and far
        self.mvp_mtx = get_mvp_matrix(c2w, self.proj_mtx)

        self.light_positions = self.camera_positions
        focal_length = (0.5 * self.cfg.eval_height / torch.tan(0.5 * self.fovy))
        directions_unit_focal = get_ray_directions(H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0)
        directions = directions_unit_focal[None, :, :, :].repeat(self._n_view, 1, 1, 1)
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]
        self.rays_o, self.rays_d = get_rays(directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize)

    def __len__(self):
        return self._n_view

    def __getitem__(self, index):
        return {
            "index": index,
            # 相机位置相关.
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "camera_positions": self.camera_positions[index],
            # 变换矩阵.
            "c2w": self.c2w[index], "proj_mtx": self.proj_mtx[index], "mvp_mtx": self.mvp_mtx[index], "fovy": self.fovy[index],
            # 光线相关.
            "light_positions": self.light_positions[index], "rays_o": self.rays_o[index], "rays_d": self.rays_d[index]
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    ####################################################################################################################
    # pl.LightningDataModule 接口.
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # setup.
    # ------------------------------------------------------------------------------------------------------------------

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    # ------------------------------------------------------------------------------------------------------------------
    # prepare_data.
    # ------------------------------------------------------------------------------------------------------------------

    def prepare_data(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # train / val / test / predict dataloader
    # ------------------------------------------------------------------------------------------------------------------

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,  # type: ignore
            batch_size=batch_size, collate_fn=collate_fn)

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(self.train_dataset, batch_size=None)

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate)
