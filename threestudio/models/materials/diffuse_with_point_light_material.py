import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("diffuse-with-point-light-material")
class DiffuseWithPointLightMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        ambient_light_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        diffuse_light_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
        ambient_only_steps: int = 1000
        diffuse_prob: float = 0.75
        textureless_prob: float = 0.5
        albedo_activation: str = "sigmoid"
        soft_shading: bool = False

    cfg: Config

    def configure(self) -> None:
        self.requires_normal = True

        self.ambient_light_color: Float[Tensor, "3"]
        self.register_buffer(
            "ambient_light_color",
            torch.as_tensor(self.cfg.ambient_light_color, dtype=torch.float32),
        )
        self.diffuse_light_color: Float[Tensor, "3"]
        self.register_buffer(
            "diffuse_light_color",
            torch.as_tensor(self.cfg.diffuse_light_color, dtype=torch.float32),
        )
        self.ambient_only = False

    def forward(
        self, features: Float[Tensor, "B ... Nf"], positions: Float[Tensor, "B ... 3"], shading_normal: Float[Tensor, "B ... 3"],
        light_positions: Float[Tensor, "B ... 3"], ambient_ratio: Optional[float] = None, shading: Optional[str] = None, **kwargs) -> Float[Tensor, "B ... 3"]:
        """
        :param features: (n_points, feat_channels). 每个采样点的features.
        :param positions: (n_points, 3). 采样点（可见光线interval中心）.
        :param shading_normal: (n_points, 3). 每个采样点的法线.
        :param light_positions: (n_points, 3). 在世界中心与相机位置的连线上.
        :param ambient_ratio: None.
        :param shading: None.
        :param kwargs:
        """
        # --------------------------------------------------------------------------------------------------------------
        """ 反照率：发出辐射与接收辐射之比. 绝对黑体的反照率是0，绝对镜面的反照率为1. """
        # --------------------------------------------------------------------------------------------------------------
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3])

        ################################################################################################################
        """ 计算diffuse light（漫反射光照）color和ambient light（环境光照）color. (3, ).
        @漫反射光照：根据光线入射角度，调整物体亮度——当光线垂直于物体表面时，物体更亮；当光线斜向物体表面时，物体较暗；
        @环境光照：全局光照，不受光源位置和物体表面法线的影响；
        """
        ################################################################################################################
        if ambient_ratio is not None:
            # if ambient ratio is specified, use it
            diffuse_light_color = (1 - ambient_ratio) * torch.ones_like(self.diffuse_light_color)
            ambient_light_color = ambient_ratio * torch.ones_like(self.ambient_light_color)
        elif self.training and self.cfg.soft_shading:
            # otherwise if in training and soft shading is enabled, random a ambient ratio
            diffuse_light_color = torch.full_like(self.diffuse_light_color, random.random())
            ambient_light_color = 1.0 - diffuse_light_color
        else:
            # otherwise use the default fixed values
            diffuse_light_color = self.diffuse_light_color
            ambient_light_color = self.ambient_light_color

        ################################################################################################################
        """ 计算颜色 """
        ################################################################################################################
        # (n_points, 3).
        light_directions: Float[Tensor, "B ... 3"] = F.normalize(light_positions - positions, dim=-1)
        """ 漫反射光照.
        @dot(shading_normal, light_directions): (n_points, 1). 根据法线与入射光线夹角调整亮度；
        @diffuse_light: (n_points, 3).
        """
        diffuse_light: Float[Tensor, "B ... 3"] = dot(shading_normal, light_directions).clamp(min=0.0) * diffuse_light_color
        """ 漫反射光照 + 环境光照. """
        textureless_color = diffuse_light + ambient_light_color
        # clamp albedo to [0, 1] to compute shading
        """ 阴影：颜色乘以反照率 """
        color = albedo.clamp(0.0, 1.0) * textureless_color

        # --------------------------------------------------------------------------------------------------------------
        """ 获取shading. """
        # --------------------------------------------------------------------------------------------------------------
        if shading is None:
            if self.training:
                # adopt the same type of augmentation for the whole batch
                if self.ambient_only or random.random() > self.cfg.diffuse_prob:
                    shading = "albedo"
                elif random.random() < self.cfg.textureless_prob:
                    shading = "textureless"
                else:
                    shading = "diffuse"
            else:
                if self.ambient_only:
                    shading = "albedo"
                else:
                    # return shaded color by default in evaluation
                    shading = "diffuse"

        # --------------------------------------------------------------------------------------------------------------
        """ 根据shading返回不同的颜色. """
        # --------------------------------------------------------------------------------------------------------------
        # multiply by 0 to prevent checking for unused parameters in DDP
        if shading == "albedo":
            return albedo + textureless_color * 0
        elif shading == "textureless":
            return albedo * 0 + textureless_color
        elif shading == "diffuse":
            return color
        else:
            raise ValueError(f"Unknown shading type {shading}")

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if global_step < self.cfg.ambient_only_steps:
            self.ambient_only = True
        else:
            self.ambient_only = False

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3]).clamp(
            0.0, 1.0
        )
        return {"albedo": albedo}
