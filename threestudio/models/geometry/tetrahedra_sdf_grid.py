
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

import threestudio
from threestudio.models.geometry.base import BaseExplicitGeometry, BaseGeometry, contract_to_unisphere
from threestudio.models.geometry.implicit_sdf import ImplicitSDF
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.isosurface import MarchingTetrahedraHelper
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *


@threestudio.register("tetrahedra-sdf-grid")
class TetrahedraSDFGrid(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        isosurface_resolution: int = 128
        isosurface_deformable_grid: bool = True
        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        geometry_only: bool = False
        fix_geometry: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # --------------------------------------------------------------------------------------------------------------
        # 等值面包围盒. (2, 3).
        # --------------------------------------------------------------------------------------------------------------
        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.register_buffer("isosurface_bbox", self.bbox.clone())
        # --------------------------------------------------------------------------------------------------------------
        # 等值面.
        # --------------------------------------------------------------------------------------------------------------
        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution, f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",)

        # --------------------------------------------------------------------------------------------------------------
        # SDF & deformation. (n_vertexes, 1) & (n_vertexes, 3).
        # --------------------------------------------------------------------------------------------------------------
        self.sdf: Float[Tensor, "Nv 1"]
        self.deformation: Optional[Float[Tensor, "Nv 3"]]
        """ 可训练的SDF和deformation. """
        if not self.cfg.fix_geometry:
            self.register_parameter(
                "sdf", nn.Parameter(torch.zeros((self.isosurface_helper.grid_vertices.shape[0], 1), dtype=torch.float32)))
            if self.cfg.isosurface_deformable_grid:
                self.register_parameter("deformation", nn.Parameter(torch.zeros_like(self.isosurface_helper.grid_vertices)))
            else:
                self.deformation = None
        else:
            self.register_buffer("sdf", torch.zeros((self.isosurface_helper.grid_vertices.shape[0], 1), dtype=torch.float32))
            if self.cfg.isosurface_deformable_grid:
                self.register_buffer("deformation", torch.zeros_like(self.isosurface_helper.grid_vertices))
            else:
                self.deformation = None

        # --------------------------------------------------------------------------------------------------------------
        # Encoding & feature 网络.
        # --------------------------------------------------------------------------------------------------------------
        if not self.cfg.geometry_only:
            self.encoding = get_encoding(self.cfg.n_input_dims, self.cfg.pos_encoding_config)
            self.feature_network = get_mlp(self.encoding.n_output_dims, self.cfg.n_feature_dims, self.cfg.mlp_network_config)

        # --------------------------------------------------------------------------------------------------------------
        # Mesh.
        # --------------------------------------------------------------------------------------------------------------
        self.mesh: Optional[Mesh] = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init: return
        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init: return

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert isinstance(self.cfg.shape_init_params, Sized) and len(self.cfg.shape_init_params) == 3
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(dim=-1, keepdim=True).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh
            mesh = trimesh.load(mesh_path)

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]), "+y": np.array([0, 1, 0]), "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]), "-y": np.array([0, -1, 0]), "-z": np.array([0, 0, -1])}
            if self.cfg.shape_init_mesh_up not in dirs or self.cfg.shape_init_mesh_front not in dirs:
                raise ValueError(f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}.")
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError("shape_init_mesh_up and shape_init_mesh_front must be orthogonal.")
            z_, x_ = (dir2vec[self.cfg.shape_init_mesh_up], dir2vec[self.cfg.shape_init_mesh_front])
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            from pysdf import SDF
            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(points_rand)[..., None]

            get_gt_sdf = func

        else:
            raise ValueError(f"Unknown shape initialization type: {self.cfg.shape_init}")

        sdf_gt = get_gt_sdf(
            scale_tensor(self.isosurface_helper.grid_vertices, self.isosurface_helper.points_range, self.isosurface_bbox))
        self.sdf.data = sdf_gt

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def isosurface(self) -> Mesh:
        # --------------------------------------------------------------------------------------------------------------
        # 1. 当mesh是固定不可优化的，那么直接返回mesh
        # --------------------------------------------------------------------------------------------------------------
        if self.cfg.fix_geometry and self.mesh is not None: return self.mesh
        # --------------------------------------------------------------------------------------------------------------
        # 2. 根据SDF和形变，使用等值面来构建模型.
        # --------------------------------------------------------------------------------------------------------------
        mesh = self.isosurface_helper(self.sdf, self.deformation)
        mesh.v_pos = scale_tensor(mesh.v_pos, self.isosurface_helper.points_range, self.isosurface_bbox)
        if self.cfg.isosurface_remove_outliers:
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
        """ 缓存模型. """
        self.mesh = mesh
        # Return
        return mesh

    def forward(self, points: Float[Tensor, "*N Di"], output_normal: bool = False) -> Dict[str, Float[Tensor, "..."]]:
        """ 计算模型顶点的特征.
        :param points: (n_points, 3). 顶点位置.
        :param output_normal: bool. 是否返回法线.
        """
        if self.cfg.geometry_only: return {}
        assert output_normal is False, f"Normal output is not supported for {self.__class__.__name__}"
        # 1. 将顶点归一化到包围盒中. (n_points, 3).
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        # 2. 由神经网络产生特征. (n_points, n_feature_dims).
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(*points.shape[:-1], self.cfg.n_feature_dims)
        # Return
        return {"features": features}

    @staticmethod
    @torch.no_grad()
    def create_from(other: BaseGeometry, cfg: Optional[Union[dict, DictConfig]] = None, copy_net: bool = True, **kwargs,) -> "TetrahedraSDFGrid":
        """ 从另外的geometry初始化.
        :param other: 另外的geometry.
        :param cfg: 自身的geometry对应的配置.
        :param copy_net: 是否读取另外geometry的的网络参数.
        """
        # --------------------------------------------------------------------------------------------------------------
        # 从TetrahedraSDFGrid初始化.
        # --------------------------------------------------------------------------------------------------------------
        if isinstance(other, TetrahedraSDFGrid):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            assert instance.cfg.isosurface_resolution == other.cfg.isosurface_resolution
            # 拷贝isosurface_bbox.
            instance.isosurface_bbox = other.isosurface_bbox.clone()
            # 拷贝sdf.data / deformation.data
            instance.sdf.data = other.sdf.data.clone()
            if instance.cfg.isosurface_deformable_grid and other.cfg.isosurface_deformable_grid:
                assert instance.deformation is not None and other.deformation is not None
                instance.deformation.data = other.deformation.data.clone()
            # 拷贝网络参数.
            if not instance.cfg.geometry_only and not other.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(other.feature_network.state_dict())
            return instance
        # --------------------------------------------------------------------------------------------------------------
        # 从Implicit Volume初始化.
        # --------------------------------------------------------------------------------------------------------------
        elif isinstance(other, ImplicitVolume):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            """ 检查另外cfg.isosurface_method / isosurface_resolution. """
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(f"Override isosurface_method of the source geometry to 'mt'")
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}")
            # 拷贝isosurface_bbox / sdf.data.
            mesh = other.isosurface()
            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = mesh.extras["grid_level"].to(instance.sdf.data).clamp(-1, 1)
            # 拷贝网络参数.
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(other.feature_network.state_dict())
            return instance
        # --------------------------------------------------------------------------------------------------------------
        # 从ImplicitSDF初始化.
        # --------------------------------------------------------------------------------------------------------------
        elif isinstance(other, ImplicitSDF):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            """ 检查另外cfg.isosurface_method / isosurface_resolution. """
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(f"Override isosurface_method of the source geometry to 'mt'")
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}")
            # 拷贝isosurface_bbox / sdf.data.
            mesh = other.isosurface()
            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = mesh.extras["grid_level"].to(instance.sdf.data)
            # 拷贝deformation.data
            if instance.cfg.isosurface_deformable_grid and other.cfg.isosurface_deformable_grid:
                assert instance.deformation is not None
                instance.deformation.data = mesh.extras["grid_deformation"].to(instance.deformation.data)
            # 拷贝网络参数.
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(other.feature_network.state_dict())
            return instance
        else:
            raise TypeError(f"Cannot create {TetrahedraSDFGrid.__name__} from {other.__class__.__name__}")

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.geometry_only or self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out
