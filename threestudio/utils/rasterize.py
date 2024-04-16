import nvdiffrast.torch as dr
import torch

from threestudio.utils.typing import *


""" 光栅化（Rasterization）：首先，使用透视投影法将模型的三角形“投射”到屏幕上（将构成三角形的三维顶点投射到屏幕上）；然后，对图像中的所
有像素进行循环，测试它们是否位于所产生的2D三角形内——如果是的话，我们就用三角形的颜色来填充这个像素。
[参考]: https://www.zhihu.com/p/544088415.
"""


class NVDiffRasterizerContext:
    """ NvDiff Rasterizer Context. """
    def __init__(self, context_type: str, device: torch.device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)

    def initialize_context(self, context_type: str, device: torch.device) -> Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]:
        if context_type == "gl":
            return dr.RasterizeGLContext(device=device)
        elif context_type == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown rasterizer context type: {context_type}")

    def vertex_transform(self, verts: Float[Tensor, "Nv 3"], mvp_mtx: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B Nv 4"]:
        """ 将顶点变换使用MVP矩阵变换到相机坐标系的成像平面上.
        :param verts: (n_vertexes, 3). 顶点位置.
        :param mvp_mtx: (batch, 4, 4). MVP矩阵，将世界坐标系中的点变换到相机坐标系的成像平面上.
        :return (batch, batch, 4). 第[i][j]个向量是将第j个点使用第i个矩阵进行变换的结果.
        """
        # (batch, 4).
        verts_homo = torch.cat([verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1)
        # (batch, batch, 4) <= (1, batch, 4)@(batch, 4, 4).
        """ @说明：结果的第[i][j]个向量是将第j个点使用第i个矩阵进行变换的结果. """
        ret = torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))
        # Return
        return ret

    def rasterize(self, pos: Float[Tensor, "B Nv 4"], tri: Integer[Tensor, "Nf 3"], resolution: Union[int, Tuple[int, int]]):
        """ TODO: 用于干吗的？
        :param pos: (batch, n_vertexes, 4). 顶点位置.
        :param tri: (n_faces, 3)@Int32. 三角形面.
        :param resolution: Int or (height, width). 输出图像分辨率.
        :return 如下元组：
            - (batch, h, w, 4)，其中4对应于u, v, z/w, triangle_id. 其中，
                * u和表示纹理坐标，对应于输出图像（屏幕）的像素；
                * z/w是深度值与透视除法的结果。在投影变换后，三维物体的坐标被映射到裁剪空间。z/w表示深度值除以透视投影后的齐次坐标的w分
                量。这个值用于深度测试，以确定哪些像素在前面，哪些在后面。
            - (batch, h, w, 0). 空的.
        """
        # rasterize in instance mode (single topology)
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(self, pos: Float[Tensor, "Nv 4"], tri: Integer[Tensor, "Nf 3"], resolution: Union[int, Tuple[int, int]]):
        """
        :param pos: (n_vertexes, 4). 顶点位置.
        :param tri: (n_faces, 3)@Int32. 三角形面.
        :param resolution: Int or (height, width). 输出图像分辨率.
        """
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(self, color: Float[Tensor, "B H W C"], rast: Float[Tensor, "B H W 4"], pos: Float[Tensor, "B Nv 4"],
                  tri: Integer[Tensor, "Nf 3"]) -> Float[Tensor, "B H W C"]:
        """ 抗锯齿.
        :param color: (batch, h, w, c). 需要进行抗锯齿处理的图像.
        :param rast: (batch, h, w, 4).
        :param pos: (batch, n_vertexes, 4). 顶点位置.
        :param tri: (n_faces, 3)@Int32. 三角形面.
        :return (batch, h, w, c). 抗锯齿处理后的图像.
        """
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(self, attr: Float[Tensor, "B Nv C"], rast: Float[Tensor, "B H W 4"], tri: Integer[Tensor, "Nf 3"],
                    rast_db=None, diff_attrs=None) -> Float[Tensor, "B H W C"]:
        """
        :param attr: (batch, n_vertexes, n_attributes). 顶点属性.
        :param rast: (batch, h, w, 4).
        :param tri: (batch, n_faces, 3)@Int32. 三角形面.
        :param rast_db:
        :param diff_attrs:
        :return 如下元组：
            - (batch, h, w, n_attributes). 插值后的属性.
            - ...
        """
        return dr.interpolate(attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs)

    def interpolate_one(self, attr: Float[Tensor, "Nv C"], rast: Float[Tensor, "B H W 4"], tri: Integer[Tensor, "Nf 3"],
                        rast_db=None, diff_attrs=None) -> Float[Tensor, "B H W C"]:
        """
        :param attr:
        :param rast:
        :param tri:
        :param rast_db:
        :param diff_attrs:
        """
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)
