import trimesh
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def get_contour_from_stl(mesh, plane_z, plane):
    """
    使用 trimesh 计算 STL 文件与平面 z=plane_z 的交线轮廓
    mesh: trimesh网格对象
    plane_z: 平面坐标
    返回: 交线段列表（每段为两个点的数组）
    """
    # 定义平面
    if plane == 'xy':
        plane_normal = np.array([0, 0, 1])
        plane_origin = np.array([0, 0, plane_z])
    elif plane == 'xz':
        plane_normal = np.array([0, 1, 0])
        plane_origin = np.array([0, plane_z, 0])
    elif plane == 'yz':
        plane_normal = np.array([1, 0, 0])
        plane_origin = np.array([plane_z, 0, 0])
    else:
        raise ValueError("无效的平面参数，请选择 'xy', 'xz' 或 'yz'")

    # 计算网格与平面的交线
    intersections = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin
    )
    points_crossing_plane = []
    for intersection in intersections:
        x = [intersection[0][0], intersection[1][0]]
        y = [intersection[0][1], intersection[1][1]]
        z = [intersection[0][2], intersection[1][2]]
        points_crossing_plane.append(np.array([x[0], y[0], z[0]]))
        points_crossing_plane.append(np.array([x[1], y[1], z[1]]))

    return points_crossing_plane, intersections


def line_plane_intersection(intersections, plane):
    """计算点云交线和平面的交点"""
    plane_normal = np.array(plane[:3])
    plane_d = plane[3]
    if abs(plane_normal[2]) > 1e-6:
        plane_origin = np.array([0, 0, -plane_d / plane_normal[2]])
    elif abs(plane_normal[1]) > 1e-6:
        plane_origin = np.array([0, -plane_d / plane_normal[1], 0])
    else:
        plane_origin = np.array([-plane_d / plane_normal[0], 0, 0])

    def line_plane_intersection_inner(line_start, line_end):
        line_dir = line_end - line_start
        denom = np.dot(line_dir, plane_normal)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(plane_normal, plane_origin - line_start) / denom
        if 0 <= t <= 1:
            intersection_point = line_start + t * line_dir
            if abs(np.dot(plane_normal, intersection_point) + plane_d) < 1e-6:
                return intersection_point
        return None

    intersection_points = []
    for line_segment in intersections:
        line_start = line_segment[0]
        line_end = line_segment[1]
        intersection = line_plane_intersection_inner(line_start, line_end)
        if intersection is not None:
            intersection_points.append(intersection)

    return np.array(intersection_points)


def get_points_by_indices(points):
    """
    1. 按contour_points1和contour_points2的范围筛选交点
    2. 对筛选后的交点按y坐标从小到大排序
    3. 取排序后倒数第1、2个点（y坐标最大的两个点）
    points: 原始二次交线交点
    point1: 基准点，用于计算筛选边界（全局变量）
    """
    # 先获取筛选边界（与contour_points1、contour_points2保持一致）
    y_upper1 = point1[1] + 240
    x_lower1 = point1[0] - 250
    z_lower1 = point1[2] - 20
    y_upper2 = point1[1] + 400
    x_upper2 = point1[0] + 200

    # 筛选在两个contour范围内的交点
    filtered_points = []
    for p in points:
        # 满足contour_points1 或 contour_points2的筛选条件
        cond1 = (p[1] < y_upper1) & (p[0] > x_lower1) & (p[2] > z_lower1)
        cond2 = (p[1] < y_upper2) & (p[0] < x_upper2)
        if cond1 or cond2:
            filtered_points.append(p)

    if len(filtered_points) == 0:
        raise ValueError("筛选后没有找到有效的交点")

    # 去重（避免浮点误差导致重复）
    unique_points = []
    seen = set()
    for point in filtered_points:
        point_tuple = tuple(np.round(point, 6))  # 四舍五入到6位小数去重
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)

    if len(unique_points) < 2:
        raise ValueError(f"去重后仅找到{len(unique_points)}个点，不足2个无法提取倒数第1、2个点")

    # 按y坐标从小到大排序
    sorted_unique_points = sorted(unique_points, key=lambda p: p[1])

    # 取倒数第1、2个点（y最大的两个点）
    point_last1 = sorted_unique_points[-1]  # 倒数第1个（y最大）
    point_last2 = sorted_unique_points[-2]  # 倒数第2个（y次大）

    return point_last1, point_last2


# 全局基准点（供get_points_by_indices使用）
point1 = None


def calculate(stl_path):
    y_distance = None
    global point1  # 声明使用全局变量point1，解决作用域问题
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标及处理
    vertices = mesh.vertices
    z_min = np.min(vertices[:, 2])
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    if len(filtered_vertices) == 0:
        raise ValueError("没有找到z轴大于最小值+300的点")
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 设置平面坐标并获取交线
    plane_z = point1[2]
    contour_points, intersections = get_contour_from_stl(mesh, plane_z, 'xy')
    if contour_points is None or intersections is None:
        raise ValueError("未能获取有效的交线数据")
    contour_points = np.array(contour_points)

    # 筛选交线点
    contour_points1 = contour_points[
        (contour_points[:, 1] < (point1[1] + 240)) &
        (contour_points[:, 0] > (point1[0] - 250)) &
        (contour_points[:, 2] > (point1[2] - 20))
        ]
    contour_points2 = contour_points[
        (contour_points[:, 1] < (point1[1] + 400)) &
        (contour_points[:, 0] < (point1[0] + 200))
        ]
    if len(contour_points1) == 0:
        raise ValueError("筛选后未找到有效的contour_points1")
    y_max_point = contour_points1[np.argmax(contour_points1[:, 1])]

    # 计算二次交线及特征点
    plane2 = (1, 0, 0, -y_max_point[0])
    intersection_point = line_plane_intersection(intersections, plane2)
    if intersection_point is None or len(intersection_point) == 0:
        raise ValueError("未能计算有效的二次交线点")
    # 提取筛选、排序后的倒数第1、2个点
    point_last1, point_last2 = get_points_by_indices(intersection_point)
    if point_last1 is None or point_last2 is None:
        raise ValueError("未能提取到有效的倒数第1、2个点")

    # 计算并输出距离
    y_distance = abs(point_last1[1] - point_last2[1])
    # print(f"后视镜到车身距离: {y_distance:.2f}")

    # # 可视化结果
    # point_size = 2
    # plt.figure(figsize=(10, 8))
    # # 绘制筛选后的交线点
    # plt.plot(contour_points2[:, 0], contour_points2[:, 1], 'ro', markersize=point_size, label='交线点')
    # # 绘制特征点
    # plt.plot(point_last1[0], point_last1[1], 'yo', markersize=12, label='倒数第1个点（y最大）')
    # plt.plot(point_last2[0], point_last2[1], 'bo', markersize=12, label='倒数第2个点（y次大）')
    # plt.title('交线点与特征点可视化')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.axis('equal')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.show()

    return y_distance


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)
