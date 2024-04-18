import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> Float[Tensor, "N 3"]:
        raise NotImplementedError


class MarchingCubeCPUHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        import mcubes

        self.mc_func: Callable = mcubes.marching_cubes
        self._grid_vertices: Optional[Float[Tensor, "N3 3"]] = None
        self._dummy: Float[Tensor, "..."]
        self.register_buffer(
            "_dummy", torch.zeros(0, dtype=torch.float32), persistent=False
        )

    @property
    def grid_vertices(self) -> Float[Tensor, "N3 3"]:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            threestudio.warn(
                f"{self.__class__.__name__} does not support deformation. Ignoring."
            )
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(
            level.detach().cpu().numpy(), 0.0
        )  # transform to numpy
        v_pos, t_pos_idx = (
            torch.from_numpy(v_pos).float().to(self._dummy.device),
            torch.from_numpy(t_pos_idx.astype(np.int64)).long().to(self._dummy.device),
        )  # transform back to torch tensor on CUDA
        v_pos = v_pos / (self.resolution - 1.0)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)


class MarchingTetrahedraHelper(IsosurfaceHelper):
    """ Marching Tetrahedra (MT) 算法.
    @说明：这是一种用于生成等值面（isovalue surfaces）的算法，通常用于可视化三维数据集中的特定值。例如，我们可能对某个特定密度的体素感兴
    趣，Marching Tetrahedra算法可以帮助我们找到这些体素的表面. 该算法概述如下：
        - MT算法将三维空间划分为一系列四面体（tetrahedra）；
        - 输入可以是点云或粗体素数据，然后生成一个包围输入数据的立方体；
        - 然后，将这个立方体网格化为一组四面体，每个四面体由4个顶点和4个面组成；
        - 根据4个顶点的SDF（有符号距离函数）值，计算网格表面的顶点位置（使用线性插值公式计算）.
        - 最终，可以得到网格表面.
    """
    def __init__(self, resolution: int, tets_path: str):
        super().__init__()
        self.resolution = resolution
        self.tets_path = tets_path

        # --------------------------------------------------------------------------------------------------------------
        # triangle_table. (16, 6).
        # todo：关于每一行6个值的解释，还需要进一步研究.
        """ @说明：在MT算法中，需要判断哪些四面体与物体的的表面相交，以便提取出网格表面，这就是"triangle table"发挥作用的地方. 一个四
        面体有4个顶点，每个顶点可能在物体表面的内部或外部，因此，总共可能有2^4=16种可能的情况（表的每行对应于一个情况）. 可以进一步将这
        些情况分为三类：
            - (1) 4个顶点都在表面内或表面外：这些四面体被丢弃；
            - (2) 有2个顶点在表面外或表面内：这些四面体与物体表面有4个交点，可以用2个三角面片表示；
            - (3) 有1个顶点在表面外，另外3个顶点在表面内或相反的情形：这些四面体与物体表面有3个交点，可以用1个三角面片表示.
        对于每一行来说，它共有6个值，对应于两个三角面片（可以表示上述三类情况）. 具体意义如下：
            - 前3个值代表第一个三角面片的三条边的索引（四面体一共有6条边，0/1/2/3/4/5）；
            - 后3个值代表第二个三角面片的三条边的索引；
            - 若值为-1,-1,-1，表示这个三角形无效.
        这些值的组合决定了四面体与物体表面的交点，从而帮助我们生成精确的三维模型. """
        # --------------------------------------------------------------------------------------------------------------
        self.triangle_table: Float[Tensor, "..."]
        self.register_buffer(
            "triangle_table", torch.as_tensor([
                [-1, -1, -1, -1, -1, -1],   # 所有顶点都在表面内.
                [1, 0, 2, -1, -1, -1],      # 1个顶点在表面外 (第1个顶点)
                [4, 0, 3, -1, -1, -1],      # 1个顶点在表面外 (第2个顶点)
                [1, 4, 2, 1, 3, 4],         # 2个顶点在表面外 (第1/2个顶点)
                [3, 1, 5, -1, -1, -1],      # 1个顶点在表面外 (第3个顶点)
                [2, 3, 0, 2, 5, 3],         # 2个顶点在表面外 (第1/3个顶点)
                [1, 4, 0, 1, 5, 4],         # 2个顶点在表面外 (第2/3个顶点)
                [4, 2, 5, -1, -1, -1],      # 3个顶点在表面外 (第1/2/3个顶点)
                [4, 5, 2, -1, -1, -1],      # 1个顶点在表面外 (第4个顶点)
                [4, 1, 0, 4, 5, 1],         # 2个顶点在表面外 (第1/4个顶点)
                [3, 2, 0, 3, 5, 2],         # 2个顶点在表面外 (第2/4个顶点)
                [1, 3, 5, -1, -1, -1],      # 3个顶点在表面外 (第1/2/4个顶点)
                [4, 1, 2, 4, 3, 1],         # 2个顶点在表面外 (第2/3个顶点)
                [3, 0, 4, -1, -1, -1],      # 3个顶点在表面外 (第1/2/3个顶点)
                [2, 0, 1, -1, -1, -1],      # 3个顶点在表面外 (第2/3/4个顶点)
                [-1, -1, -1, -1, -1, -1]    # 4个顶点都在表面外
            ], dtype=torch.long), persistent=False)

        # --------------------------------------------------------------------------------------------------------------
        # num_triangles_table. (16, ). triangle_table的每一行可以用几个三角面表示.
        # --------------------------------------------------------------------------------------------------------------
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer("num_triangles_table", torch.as_tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long), persistent=False)
        # --------------------------------------------------------------------------------------------------------------
        # base_tet_edges. (12, ). 四面体6个边，每两个元素对应于一条边的两个顶点的索引 (0/1/2/3).
        # --------------------------------------------------------------------------------------------------------------
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer("base_tet_edges", torch.as_tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long), persistent=False)

        # --------------------------------------------------------------------------------------------------------------
        # _grid_vertices. (n_grid_vertexes, 3). 所有四面体的顶点的集合.
        # --------------------------------------------------------------------------------------------------------------
        tets = np.load(self.tets_path)
        self._grid_vertices: Float[Tensor, "..."]   # E.g., (277410, 3).
        self.register_buffer("_grid_vertices", torch.from_numpy(tets["vertices"]).float(), persistent=False)
        # --------------------------------------------------------------------------------------------------------------
        # Indices. (n_tets, 4). 每一个四面体的4个顶点在grid_vertexes中的索引.
        # --------------------------------------------------------------------------------------------------------------
        self.indices: Integer[Tensor, "..."]        # E.g., (1524684, 4).
        self.register_buffer("indices", torch.from_numpy(tets["indices"]).long(), persistent=False)

        # --------------------------------------------------------------------------------------------------------------
        # _all_edges. (n_edges, 2). 所有四面体的边（顶点索引对）的集合.
        # --------------------------------------------------------------------------------------------------------------
        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

    def normalize_grid_deformation(self, grid_vertex_offsets: Float[Tensor, "Nv 3"]) -> Float[Tensor, "Nv 3"]:
        """ 将grid_vertex的形变归一化.
        :param grid_vertex_offsets: (n_grid_vertexes, 3). 每一个grid_vertex的形变.
        :return (n_grid_vertexes, 3). 归一化后结果.
        """
        return ((self.points_range[1] - self.points_range[0]) / self.resolution  # half tet size is approximately 1 / self.resolution
                * torch.tanh(grid_vertex_offsets))  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            # ----------------------------------------------------------------------------------------------------------
            # 边. (12, )@torch.long. 每两个元素为一组，代表边的两个顶点的索引.
            # ----------------------------------------------------------------------------------------------------------
            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=self.indices.device)
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        """
        :param edges_ex2: (n_edges, 2).
        :return (n_edges, 1, 2), 其中2对应于排序后的边的两个顶点.
        """
        with torch.no_grad():
            # (n_edges, ). 边的第一个顶点索引是否大于第二个顶点索引.
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            # (n_edges, ) -> (n_edges, 1).
            order = order.unsqueeze(dim=1)
            # (n_edges, 1). 对于每一个边的两个顶点，把小的那个放前面.
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        """
        :param pos_nx3: (n_grid_vertexes, 3). 每一个grid_vertex的位置.
        :param sdf_n: (n_grid_vertexes, 1). 每一个grid_vertex的场值的level.
        :param tet_fx4: (n_tets, 4). 每一个四面体的4个顶点在grid_vertexes中的索引.
        :return
            - verts: (n_unique_valid_edges_cross, 3). 横跨边插值得到的顶点.
            - faces: (n_valid_tets_cross, 3). 横跨四面体的表示三角形面的边.
                * 若某条边为非横跨，值为-1；
                * 否则，值为在横跨边中的索引0/1/2/...；
        """
        with torch.no_grad():
            # (n_grid_vertexes, 1). 每一个grid_vertex是否在模型表面外.
            occ_n = sdf_n > 0
            # (n_tets, 4). 表示每一个四面体的每一个顶点是否在模型表面外.
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            # (n_tets, ). 计算每一个四面体有多少个点在模型表面外.
            occ_sum = torch.sum(occ_fx4, -1)
            # (n_tets, ). 表示每一个四面体是否有效：当所有点都在模型外/内时，这个四面体无效.
            valid_tets = (occ_sum > 0) & (occ_sum < 4)

            # (n_valid_tets, 12) -> (n_valid_tets*6, 2). 所有有效四面的边（两个顶点在n_grid_vertexes中的索引）.
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            # (n_valid_edges=n_valid_tets*6, 1, 2). 对于all_edges的每一个边，把顶点索引小的那个放在前面.
            all_edges = self.sort_edges(all_edges)
            # (n_unique_valid_edges, 1, 2). 去除重合的边.
            # (n_valid_edges, ). 原来边所对应的去重后边的索引.
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
            unique_edges = unique_edges.long()

            # (n_unique_valid_edges, ). 每一个边是否是横跨（指一个顶点在模型表面外，而另一个在模型表面内）.
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            # (n_unique_valid_edges, ). 全是-1.
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device) * -1
            # (n_unique_valid_edges, ). 对于横跨的那些边，将mapping的值依次设为0/1/2/...
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=pos_nx3.device)
            # (n_valid_edges, ). 将idx_map设为所对应unique_edge的mapping值（非横跨为-1，横跨为0/1/2/...）；
            idx_map = mapping[idx_map]

            # (n_unique_valid_edges_cross, 1, 2). 横跨的unique_edges.
            interp_v = unique_edges[mask_edges]

        # --------------------------------------------------------------------------------------------------------------
        # 得到横跨的unique_valid_edges，用于后续插值. (n_unique_valid_edges_cross, 2, 3).
        # --------------------------------------------------------------------------------------------------------------
        # (n_unique_valid_edges_cross, 2, 3). unique_valid_edges_out边的两个顶点的位置.
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        # --------------------------------------------------------------------------------------------------------------
        # 获取插值权重. (n_unique_valid_edges_cross, 2, 1).
        """ 假设边的两个顶点所对应的场值的level分别为w1和w2，那么从以下的计算中可以知道，两个顶点的插值权重分别为
        - (-w2)/(-w1-w2) = w2/(w1+w2).
        - (-w1)/(-w1-w2) = w1/(w1+w2).
        """
        # --------------------------------------------------------------------------------------------------------------
        # (n_unique_valid_edges_cross, 2, 1). unique_valid_edges_out边的两个顶点的场值的level.
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        # 将场值的level反号.
        edges_to_interp_sdf[:, -1] *= -1
        # (n_unique_valid_edges_cross, 1, 1). 将edges_to_interp_sdf的第二维度相加并保持维度.
        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        # (n_unique_valid_edges_cross, 2, 1). 将edges_to_interp_sdf的第二维度翻转，再处以denominator.
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        # --------------------------------------------------------------------------------------------------------------
        """ 将edges_to_interp的两个边的顶点插值得到新的顶点. (n_unique_valid_edges_cross, 3). """
        # --------------------------------------------------------------------------------------------------------------
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        # (n_valid_tets, 6). 每一个valid_tets的6条边的idx_map值.
        idx_map = idx_map.reshape(-1, 6)

        # (4, ). [1, 2, 4, 8].
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        """ 得到有效四面体所对应的triangle_table情况. (n_valid_tets, ).
        ----------------------------------------------------------------------------------------------------------------
        四个顶点是否在表面外(第一个顶点是否在表面外)     对应triangle_table的索引
        ----------------------------------------------------------------------------------------------------------------
            0 0 0 0                                     0   (+1 = 1)
            0 0 0 1                                     8   (+1 = 9)
            0 0 1 0                                     4   (+1 = 5)
            0 0 1 1                                     12  (+1 = 13)
            0 1 0 0                                     2   (+1 = 3)
            0 1 0 1                                     10  (+1 = 11)
            0 1 1 0                                     6   (+1 = 7)
            0 1 1 1                                     14  (+1 = 15)
        ----------------------------------------------------------------------------------------------------------------
        """
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        # 获取每一个有效四面体的可以用几个三角形面表示. (n_valid_tets, ).
        num_triangles = self.num_triangles_table[tetindex]

        # (n_valid_tets_cross, 3). 有效四面体的表示三角形面的边.
        #   其中n_valid_tets_cross = n_valid_tets_with_1/3_verts_out + n_valid_tets_with_2_verts_out*2.
        faces = torch.cat([
            # (1) 有1个顶点或3个顶点在表面外. (n_valid_tets_with_1/3_verts_out, 3).
            #   input: (n_valid_tets_with_1/3_verts_out, 6). 对于6来说，
            #       - 如果某条边为非横跨，那么值为-1.
            #       - 否则，值为0/1/2/...
            #   index: 有效四面体的表示三角形面的边. (n_valid_tets_with_1/3_verts_out, 3).
            #   结果: (n_valid_tets_with_1/3_verts_out, 3). 有效四面体的表示三角形面的边，对于3来说，
            #       - 若边为非横跨，值为-1；否则，为在横跨边中的索引0/1/2/...
            torch.gather(input=idx_map[num_triangles == 1], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            # (2) 有2个顶点在表面外. (n_valid_tets_with_2_verts_out*2, 3).
            #   input: (n_valid_tets_with_2_verts_out, 6). 对于6来说，同上.
            #   index: 有效四面体的表示三角形面的边. (n_valid_tets_with_2_verts_out, 6).
            #   结果: (n_valid_tets_with_2_verts_out, 6). 有效四面体的表示三角形面的边，对于6来说，同上. 之后，会reshape.
            torch.gather(input=idx_map[num_triangles == 2], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3)], dim=0)

        return verts, faces

    def forward(self, level: Float[Tensor, "N3 1"], deformation: Optional[Float[Tensor, "N3 3"]] = None) -> Mesh:
        """
        :param level: (n_points, 1).
        :param deformation: (n_points, 3).
        """
        # --------------------------------------------------------------------------------------------------------------
        # 对grid_vertexes进行形变.
        # --------------------------------------------------------------------------------------------------------------
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(deformation)
        else:
            grid_vertices = self.grid_vertices

        # --------------------------------------------------------------------------------------------------------------
        # 进行MT算法.
        """ 返回值说明：
        @v_pos: (n_unique_valid_edges_cross, 3). 横跨边插值得到的顶点.
        @t_pos_idx: (n_valid_tets_cross, 3). 横跨四面体的表示三角形面的边.
            - 若某条边为非横跨，值为-1；
            - 否则，值为在横跨边中的索引0/1/2/...；
        """
        # --------------------------------------------------------------------------------------------------------------
        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        # --------------------------------------------------------------------------------------------------------------
        # 基于MT算法测定的表面来构建Mesh.
        # --------------------------------------------------------------------------------------------------------------
        mesh = Mesh(
            v_pos=v_pos, t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices, tet_edges=self.all_edges, grid_level=level, grid_deformation=deformation)

        # Return
        return mesh
