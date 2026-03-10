import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def get_contour_from_stl(mesh, plane_z, plane):
    """
    使用 trimesh 计算 STL 文件与平面 z=plane_z 的交线轮廓
    mesh: trimesh 网格对象
    plane_z: 平面 z 坐标
    plane: 平面类型
    返回: 交线段列表（每段为两个点的数组）
    """
    # 定义平面：法向量 (0, 1, 0)，原点 (0, plane_z, 0)
    if plane == 'xz':
        plane_normal = np.array([0, 1, 0])  # XZ 平面的法向量
        plane_origin = np.array([0, plane_z, 0])  # 平面的原点
    else:
        raise ValueError("无效的平面参数，请选择 'xz'")

    # 计算网格与平面的交线
    intersections = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin
    )
    points_crossing_plane = []
    for intersection in intersections:
        x = [intersection[0][0], intersection[1][0]]
        z = [intersection[0][2], intersection[1][2]]
        # 添加交点（投影到XZ平面，即y=0）
        points_crossing_plane.append(np.array([x[0], plane_z, z[0]]))
        points_crossing_plane.append(np.array([x[1], plane_z, z[1]]))

    return points_crossing_plane


def calculate(stl_path):
    Z = None
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标
    vertices = mesh.vertices
    if vertices.size == 0:
        raise ValueError("模型中没有有效的顶点数据")

    # 找到z轴最小值
    z_min = np.min(vertices[:, 2])

    # 筛选满足条件的点：z轴大于z轴最小值300mm的点
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    if len(filtered_vertices) == 0:
        raise ValueError("没有找到Z轴大于最小值+300的点")

    # 在这些筛选后的点中找y轴最小值对应的点（点1）
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 筛选点
    filtered_vertices2 = vertices[(vertices[:, 2] > (point1[2] - 80)) & (vertices[:, 1] < (point1[1] + 230))]
    if len(filtered_vertices2) == 0:
        raise ValueError("第二次筛选未找到符合条件的点")

    # 投影到yz平面（保留y和z坐标，x坐标设为0）
    projected_points = filtered_vertices2[:, 1:]
    projected_points1 = projected_points[
        (projected_points[:, 1] > (point1[2] - 30)) & (projected_points[:, 0] < (point1[1] + 230))]
    if len(projected_points1) == 0:
        raise ValueError("投影后未找到符合条件的点")

    # 找出投影到XY平面的X向的最大值和最小值
    x_max = np.max(projected_points1[:, 0])
    x_min = np.min(projected_points[:, 0])

    # 计算x_max和x_min的中间值
    x_mid = (x_max + x_min) / 2

    # 设置平面 z 坐标
    plane_z = x_mid  # XZ 平面的 y 坐标

    # 获取与 XZ 平面的交点
    contour_points = get_contour_from_stl(mesh, plane_z, 'xz')
    if contour_points is None or len(contour_points) == 0:
        raise ValueError("未获取到有效的交线点")

    # 转换为 NumPy 数组以便处理
    contour_points = np.array(contour_points)

    # 筛选z轴大于（点1 z轴值-150mm）的点
    contour_points1 = contour_points[
        (contour_points[:, 2] > (point1[2] - 180)) & (contour_points[:, 0] < (point1[0] + 100))]
    if len(contour_points1) == 0:
        raise ValueError("筛选后未找到符合条件的交线点")

    # 找出投影到XZ平面的Z向的最大值和最小值
    z_max = np.max(contour_points1[:, 2])
    z_min = np.min(contour_points1[:, 2])

    # 找出投影到XZ平面的Z向的最大值和最小值对应的点
    z_max_point = contour_points1[np.argmax(contour_points1[:, 2])]
    z_min_point = contour_points1[np.argmin(contour_points1[:, 2])]

    # 后视镜Z向尺寸
    Z = abs(z_max - z_min)
    #print(f"后视镜Z向尺寸: {Z:.2f}")

    # # 可视化结果
    # point_size = 2  # 设置点的大小
    # plt.figure()
    # plt.plot(contour_points1[:, 0], contour_points1[:, 2], 'ro', markersize=point_size)  # 绘制红色点，设置点的大小
    # plt.plot(z_max_point[0], z_max_point[2], 'go', markersize=10, label='Z max')  # 绿色圆点标记Z最大值点
    # plt.plot(z_min_point[0], z_min_point[2], 'bo', markersize=10, label='Z min')  # 蓝色圆点标记Z最小值点
    # plt.title('Intersection Points with XZ Plane')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return Z


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)