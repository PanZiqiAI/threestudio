from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import threestudio
from threestudio.utils.ops import dot
from threestudio.utils.typing import *


class Mesh:
    def __init__(self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], **kwargs) -> None:
        """
        :param v_pos: (n_vertexes, 3). 模型顶点位置.
        :param t_pos_idx: (n_faces, 3)@Int32. 模型的三角面（面的三个顶点的索引）.
        """
        # 顶点和面.
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        # 1. 顶点法线. (n_vertexes, 3).
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        # 2. 顶点切线.
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None


        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._t_tex_idx: Optional[Float[Tensor, "Nf 3"]] = None
        self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        # 其它.
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items(): self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    ####################################################################################################################
    # 顶点
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # 法线.
    # ------------------------------------------------------------------------------------------------------------------

    def _compute_vertex_normal(self):
        """ 计算顶点法线.
        :return (n_vertexes, 3).
        """
        # (n_faces, ). 每个三角形面的三个顶点的索引.
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]
        # (n_faces, ). 每个三角形的三个顶点的位置.
        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]
        # --------------------------------------------------------------------------------------------------------------
        # 1. 计算面法线（垂直于面）. (n_faces, 3).
        # --------------------------------------------------------------------------------------------------------------
        face_normals = torch.cross(v1 - v0, v2 - v0)
        # --------------------------------------------------------------------------------------------------------------
        # 2. 计算顶点法线. (n_vertexes, 3).
        """ @算法：某个顶点的法线等于它所在的所有面的法线的叠加. """
        # --------------------------------------------------------------------------------------------------------------
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
        # 将退化法线（模长为零）替换为默认的(0, 0, 1)，之后再归一化.
        v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm))
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled(): assert torch.all(torch.isfinite(v_nrm))
        # Return
        return v_nrm

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    # ------------------------------------------------------------------------------------------------------------------
    # 切线.
    # ------------------------------------------------------------------------------------------------------------------

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    # ------------------------------------------------------------------------------------------------------------------
    # UV.
    # ------------------------------------------------------------------------------------------------------------------

    def _unwrap_uv(self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}):
        """ 使用xatlas库自动展UV. """
        threestudio.info("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(self.v_pos.detach().cpu().numpy(), self.t_pos_idx.cpu().numpy())
        co = xatlas.ChartOptions()
        for k, v in xatlas_chart_options.items(): setattr(co, k, v)
        po = xatlas.PackOptions()
        for k, v in xatlas_pack_options.items(): setattr(po, k, v)
        atlas.generate(co, po)
        _, indices, uvs = atlas.get_mesh(0)
        uvs = torch.from_numpy(uvs).to(self.v_pos.device).float()
        indices = torch.from_numpy(
            indices.astype(np.uint64, casting="same_kind").view(np.int64)).to(self.v_pos.device).long()
        return uvs, indices

    def unwrap_uv(self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}):
        self._v_tex, self._t_tex_idx = self._unwrap_uv(xatlas_chart_options, xatlas_pack_options)

    @property
    def v_tex(self):
        if self._v_tex is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self):
        if self._t_tex_idx is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._t_tex_idx

    # ------------------------------------------------------------------------------------------------------------------
    # 颜色.
    # ------------------------------------------------------------------------------------------------------------------

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb

    @property
    def v_rgb(self):
        return self._v_rgb

    ####################################################################################################################
    # 边.
    ####################################################################################################################

    def _compute_edges(self):
        """ 获取所有的边. """
        # 1. 所有三角形面的三条边. (n_edges_from_faces, 2).
        edges = torch.cat([self.t_pos_idx[:, [0, 1]], self.t_pos_idx[:, [1, 2]], self.t_pos_idx[:, [2, 0]]], dim=0)
        # 2. 排序并去重. (n_edges_from_faces, 2) -> (n_edges, 2).
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        # Return
        return edges

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    ####################################################################################################################
    # Utils & Loss.
    ####################################################################################################################

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]) -> Mesh:
        """ 有的时候模型会分成很多组件，而这些组件当中，有的可能是噪音. 在这里，将那些三角形面数小于一个阈值的组件视为噪音，并将它们移除.
        :param outlier_n_faces_threshold: Int/float. 阈值.
            - 如果是int，那么就是三角形面数阈值.
            - 如果是float，那么将三角形面数最多的组件的三角形面数乘上这个float，作为三角形面数阈值.
        """
        if self.requires_grad:
            threestudio.debug("Mesh is differentiable, not removing outliers")
            return self

        # --------------------------------------------------------------------------------------------------------------
        # 1. 使用trimesh将模型分割为多个组件.
        # Use trimesh to first split the mesh into connected components. Then remove the components with less than n_face_threshold faces.
        # --------------------------------------------------------------------------------------------------------------
        import trimesh
        # (1) Construct a trimesh object
        mesh = trimesh.Trimesh(vertices=self.v_pos.detach().cpu().numpy(), faces=self.t_pos_idx.detach().cpu().numpy())
        # (2) Split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # Log the number of faces in each component
        threestudio.debug(
            "Mesh has {} components, with faces: {}".format(len(components), [c.faces.shape[0] for c in components]))

        """ 计算三角形面数阈值. """
        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # Set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold)
        else:
            # Set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold
        # Log the threshold
        threestudio.debug("Removing components with less than {} faces".format(n_faces_threshold))

        # --------------------------------------------------------------------------------------------------------------
        # 2. 移除噪音组件并将剩余组件重新组装成模型.
        # --------------------------------------------------------------------------------------------------------------
        # (1) Remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]
        # log the number of faces in each component after removing outliers
        threestudio.debug(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]))
        # (2) Merge the components
        mesh = trimesh.util.concatenate(components)

        # --------------------------------------------------------------------------------------------------------------
        # 3. 将trimesh再转换回原来的mesh.
        # --------------------------------------------------------------------------------------------------------------
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)
        clean_mesh = Mesh(v_pos, t_pos_idx)
        # 继承原来mesh的extras.
        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            threestudio.debug(f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}")

        # Return
        return clean_mesh

    def normal_consistency(self) -> Float[Tensor, ""]:
        """ 计算每一个边的两个顶点法线的余弦相似度的损失. """
        # 1. 获取每一个边的两个顶点的法线. (n_edges, 2, 3).
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        # 2. 计算边顶点法线一致性：一个边的两个顶点的法线的cosine相似度应该很高.
        nc = (1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)).mean()
        # Return
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
        adj_values = torch.ones(adj.shape[1]).to(verts)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def laplacian(self) -> Float[Tensor, ""]:
        with torch.no_grad():
            L = self._laplacian_uniform()
        loss = L.mm(self.v_pos)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss
