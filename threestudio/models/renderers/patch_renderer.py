from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.typing import *


@threestudio.register("patch-renderer")
class PatchRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        patch_size: int = 128
        base_renderer_type: str = ""
        base_renderer: Optional[VolumeRenderer.Config] = None
        global_detach: bool = False
        global_downsample: int = 4

    cfg: Config

    def configure(self, geometry: BaseImplicitGeometry, material: BaseMaterial, background: BaseBackground) -> None:
        self.base_renderer = threestudio.find(self.cfg.base_renderer_type)(
            self.cfg.base_renderer, geometry=geometry, material=material, background=background)

    def forward(self, rays_o: Float[Tensor, "B H W 3"], rays_d: Float[Tensor, "B H W 3"],
                light_positions: Float[Tensor, "B 3"], bg_color: Optional[Tensor] = None, **kwargs) -> Dict[str, Float[Tensor, "..."]]:
        """
        :param rays_o: (batch, h, w, 3). 每个像素的光线源点（相机位置）[世界坐标系].
        :param rays_d: (batch, h, w, 3). 每个像素的光线方向 [世界坐标系].
        :param light_positions: (batch, 3). 相机光线路径上的一个位置.
        :param bg_color: (batch, 3) or None. 背景颜色.
        :param kwargs:
        """
        B, H, W, _ = rays_o.shape
        ################################################################################################################
        # 训练.
        ################################################################################################################
        if self.base_renderer.training:
            # ----------------------------------------------------------------------------------------------------------
            # 1. 下采样全局渲染.
            # ----------------------------------------------------------------------------------------------------------
            downsample = self.cfg.global_downsample
            # (1) 下采样. (batch, h//downsample, w//downsample, 3).
            global_rays_o = torch.nn.functional.interpolate(
                rays_o.permute(0, 3, 1, 2), (H // downsample, W // downsample), mode="bilinear").permute(0, 2, 3, 1)
            global_rays_d = torch.nn.functional.interpolate(
                rays_d.permute(0, 3, 1, 2), (H // downsample, W // downsample), mode="bilinear").permute(0, 2, 3, 1)
            """ 渲染. """
            out_global = self.base_renderer(global_rays_o, global_rays_d, light_positions, bg_color, **kwargs)
            # ----------------------------------------------------------------------------------------------------------
            # 2. 原分辨率Patch渲染.
            # ----------------------------------------------------------------------------------------------------------
            PS = self.cfg.patch_size
            patch_x = torch.randint(0, W - PS, (1, )).item()
            patch_y = torch.randint(0, H - PS, (1, )).item()
            # (1) 选取一个Patch. (batch, patch_size, patch_size, 3).
            patch_rays_o = rays_o[:, patch_y: patch_y + PS, patch_x: patch_x + PS]
            patch_rays_d = rays_d[:, patch_y: patch_y + PS, patch_x: patch_x + PS]
            """ 渲染 """
            out = self.base_renderer(patch_rays_o, patch_rays_d, light_positions, bg_color, **kwargs)
            # 1> 获取out中形状为(batch, patch_size, patch_size, ...)的tensor的键.
            valid_patch_key = []
            for key in out:
                if torch.is_tensor(out[key]):
                    if len(out[key].shape) == len(out["comp_rgb"].shape):
                        if out[key][..., 0].shape == out["comp_rgb"][..., 0].shape:
                            valid_patch_key.append(key)
            # 2> 将下采样结果插值放大，然后将对应patch设置到对应区域.
            for key in valid_patch_key:
                out_global[key] = F.interpolate(out_global[key].permute(0, 3, 1, 2), (H, W), mode="bilinear").permute(0, 2, 3, 1)
                if self.cfg.global_detach: out_global[key] = out_global[key].detach()
                out_global[key][:, patch_y: patch_y + PS, patch_x: patch_x + PS] = out[key]
            out = out_global
        ################################################################################################################
        # 测试.
        ################################################################################################################
        else:
            out = self.base_renderer(rays_o, rays_d, light_positions, bg_color, **kwargs)

        # Return
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        self.base_renderer.update_step(epoch, global_step, on_load_weights)

    def train(self, mode=True):
        return self.base_renderer.train(mode)

    def eval(self):
        return self.base_renderer.eval()
