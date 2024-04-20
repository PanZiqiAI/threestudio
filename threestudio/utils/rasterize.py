import nvdiffrast.torch as dr
import torch

from threestudio.utils.typing import *


""" 光栅化（Rasterization）：首先，使用透视投影法将模型的三角形“投射”到屏幕上（将构成三角形的三维顶点投射到屏幕上）；然后，对图像中的所
有像素进行循环，测试它们是否位于所产生的2D三角形内——如果是的话，我们就用三角形的颜色来填充这个像素。
[1]: https://www.zhihu.com/p/544088415.
[2]: https://zhuanlan.zhihu.com/p/671493698 """


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
        :return (batch, n_vertexes, 4). 第[i][j]个向量是将第j个点使用第i个矩阵进行变换的结果.
        """
        # (n_vertexes, 4).
        verts_homo = torch.cat([verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1)
        # (batch, n_vertexes, 4) <= (1, n_vertexes, 4)@(batch, 4, 4).
        """ @说明：结果的第[i][j]个向量是将第j个点使用第i个矩阵进行变换的结果. """
        ret = torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))
        # Return
        return ret

    def rasterize(self, pos: Float[Tensor, "B Nv 4"], tri: Integer[Tensor, "Nf 3"], resolution: Union[int, Tuple[int, int]]):
        """ 光栅化.
        :param pos: (batch, n_vertexes, 4). 在相机坐标系中变换到裁剪空间中的位置.
        :param tri: (n_faces, 3)@Int32. 三角形面（三个顶点在上述pos的n_vertexes中的索引）.
        :param resolution: Int or (height, width). 输出图像分辨率.
        :return 如下元组：
            - (batch, h, w, 4)，其中4对应于u, v, z/w, triangle_id. 其中，
                * triangle_id表示对应图像像素是由哪个三角形光栅化而来的（偏移1）. 其为沿着穿过像素中心的光线的最近三角形.
                * u和v表示像素在光栅化三角形内的坐标：它的第一个顶点(u, v)=(1, 0)，第二个顶点(u, v)=(0, 1)，第三个顶点(u, v)=(0, 0).
                这样就可以根据uv坐标以及三角形三个顶点的特征来计算得到像素的特征.
                * z/w是深度值与透视除法的结果。在投影变换后，三维物体的坐标被映射到裁剪空间。z/w表示深度值除以透视投影后的齐次坐标的w分
                量。这个值用于深度测试，以确定哪些像素在前面，哪些在后面。
            - (batch, h, w, 0). 空的.
        """
        # Rasterize in instance mode (single topology)
        """ @说明：Nvdiffrast中的所有操作都有效支持小批量轴. 与此相关的是，我们支持两种表示几何图形的方式：范围（range）模式和实例
        化（instanced）模式. 如果要在每个小批量索引中渲染不同的网格，则需要使用范围模式. 但是，如果您在每个小批量索引中渲染相同的网格，
        但具有可能不同的视点、顶点位置、属性、纹理等，则实例化模式将更加方便. [来自：https://zhuanlan.zhihu.com/p/671493698]
        """
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(self, pos: Float[Tensor, "Nv 4"], tri: Integer[Tensor, "Nf 3"], resolution: Union[int, Tuple[int, int]]):
        """ 光栅化.
        :param pos: (n_vertexes, 4). 在相机坐标系中变换到裁剪空间中的位置.
        :param tri: (n_faces, 3)@Int32. 三角形面（三个顶点在上述pos的n_vertexes中的索引）.
        :param resolution: Int or (height, width). 输出图像分辨率.
        :return 如下元组：解释见上.
            - (h, w, 4)，其中4对应于u, v, z/w, triangle_id.
            - (h, w, 0). 空的.
        """
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(self, color: Float[Tensor, "B H W C"], rast: Float[Tensor, "B H W 4"], pos: Float[Tensor, "B Nv 4"],
                  tri: Integer[Tensor, "Nf 3"]) -> Float[Tensor, "B H W C"]:
        """ 抗锯齿.
        :param color: (batch, h, w, c). 需要进行抗锯齿处理的图像.
        :param rast: (batch, h, w, 4). 光栅化结果.
        :param pos: (batch, n_vertexes, 4). 裁剪空间中的顶点位置.
        :param tri: (n_faces, 3)@Int32. 三角形面.
        :return (batch, h, w, c). 抗锯齿处理后的图像.
        """
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(self, attr: Float[Tensor, "B Nv C"], rast: Float[Tensor, "B H W 4"], tri: Integer[Tensor, "Nf 3"],
                    rast_db=None, diff_attrs=None) -> Float[Tensor, "B H W C"]:
        """
        :param attr: (batch, n_vertexes, n_attributes). 顶点属性.
        :param rast: (batch, h, w, 4). 模型光栅化的结果.
        :param tri: (batch, n_faces, 3)@Int32. 三角形面.
        :param rast_db:
        :param diff_attrs:
        :return 如下元组：
            - (batch, h, w, n_attributes). 插值后的属性.
            @说明：对于背景像素（没有被光栅化三角形覆盖的像素），插值结果为零向量.
            - ...
        """
        return dr.interpolate(attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs)

    def interpolate_one(self, attr: Float[Tensor, "Nv C"], rast: Float[Tensor, "B H W 4"], tri: Integer[Tensor, "Nf 3"],
                        rast_db=None, diff_attrs=None) -> Float[Tensor, "B H W C"]:
        """
        :param attr: (n_vertexes, n_attributes). 顶点属性.
        :param rast: (1, h, w, 4). 模型光栅化的结果. 第一个1代表batch.
        :param tri: (1, n_faces, 3)@Int32. 三角形面. 第一个1代表batch.
        :param rast_db:
        :param diff_attrs:
        :return 如下元组：
            - (1, h, w, n_attributes). 插值后的属性. 第一个1代表batch.
            - ...
        """
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)
