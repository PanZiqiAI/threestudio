from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(self, geometry: BaseImplicitGeometry, material: BaseMaterial, background: BaseBackground) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(self, mvp_mtx: Float[Tensor, "B 4 4"], camera_positions: Float[Tensor, "B 3"], light_positions: Float[Tensor, "B 3"],
                height: int, width: int, render_rgb: bool = True, **kwargs) -> Dict[str, Any]:
        """
        :param mvp_mtx: (batch, 4, 4). MVP变换矩阵.
        :param camera_positions: (batch, 3). 相机位置.
        :param light_positions: (batch, 3). 光线位置.
        :param height:
        :param width:
        :param render_rgb: bool.
        """
        batch_size = mvp_mtx.shape[0]

        ################################################################################################################
        # 1. 获取模型并光栅化.
        ################################################################################################################
        # 通过等值面获取模型.
        mesh = self.geometry.isosurface()
        # --------------------------------------------------------------------------------------------------------------
        # (1) 将世界坐标系中的模型顶点变换到相机坐标系成像平面上. (batch, n_vertexes, 4).
        # --------------------------------------------------------------------------------------------------------------
        """ @说明：第[i][j]个向量是将第j个点使用第i个矩阵变换的结果. """
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(mesh.v_pos, mvp_mtx)
        # --------------------------------------------------------------------------------------------------------------
        # (2) 将模型光栅化. (batch, height, width, 4).
        # --------------------------------------------------------------------------------------------------------------
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))

        ################################################################################################################
        # 2. 计算渲染结果.
        ################################################################################################################
        # (batch, height, width, 1). 表示每个像素是否对应光栅化三角形.
        mask = rast[..., 3:] > 0
        # --------------------------------------------------------------------------------------------------------------
        # (1) 不透明度：像素对应于三角形，就是不透明的. (batch, height, width, 1).
        # --------------------------------------------------------------------------------------------------------------
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)
        out = {"opacity": mask_aa, "mesh": mesh}
        # --------------------------------------------------------------------------------------------------------------
        # (2) 法线：每个像素根据相关顶点的法线插值而来的结果. (batch, height, width, 3).
        # --------------------------------------------------------------------------------------------------------------
        # 1> 将顶点法线插值并归一化. (batch, height, width, 3).
        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        # 2> 插值并抗锯齿. (batch, height, width, 3).
        gb_normal_aa = torch.lerp(torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float())
        gb_normal_aa = self.ctx.antialias(gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx)
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        # --------------------------------------------------------------------------------------------------------------
        # (2) RGB图像.
        # --------------------------------------------------------------------------------------------------------------
        if render_rgb:
            # (batch, height, width). 每个像素是否是前景（即是否对应光栅化三角形）.
            selector = mask[..., 0]
            # 得到像素点对应的模型表面位置，可以理解为像素中心的光线射到模型表面的位置. (batch, height, width, 3).
            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            # ----------------------------------------------------------------------------------------------------------
            # 1. Geometry: 得到前景模型表面位置的几何输出.
            # ----------------------------------------------------------------------------------------------------------
            # 选择前景像素对应的模型表面位置. (n_fg_pixels, 3).
            positions = gb_pos[selector]
            # (1) 预测前景像素对应模型表面位置的特征. Dict of features. (n_fg_pixels, n_feature_dims).
            geo_out = self.geometry(positions, output_normal=False)
            """ 额外的几何信息. """
            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(mesh.v_tng, rast, mesh.t_pos_idx)
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]
            # ----------------------------------------------------------------------------------------------------------
            # 2. Material: 获取前景颜色.
            # ----------------------------------------------------------------------------------------------------------
            # (batch, height, width, 3). 每一个像素的光线方向.
            gb_viewdirs = F.normalize(gb_pos - camera_positions[:, None, None, :], dim=-1)
            # (1) 预测前景颜色. (n_fg_pixels, 3).
            rgb_fg = self.material(
                viewdirs=gb_viewdirs[selector], positions=positions,
                light_positions=light_positions[:, None, None, :].expand(-1, height, width, -1)[selector],
                **extra_geo_info, **geo_out)
            """ 获取前景颜色，将背景置为黑色. (batch, height, width, 3). """
            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg
            # ----------------------------------------------------------------------------------------------------------
            # 3. Background: 获取背景颜色.
            # ----------------------------------------------------------------------------------------------------------
            # (1) 预测背景颜色. (batch, height, wdith, 3).
            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            """ 前景背景融合. """
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

        # Return
        return out
