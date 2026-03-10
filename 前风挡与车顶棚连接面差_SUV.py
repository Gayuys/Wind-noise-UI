import trimesh
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def get_contour_from_stl(mesh, plane_z, plane):
    """
    使用 trimesh 计算 STL 文件与平面的交线轮廓
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
        points_crossing_plane.append(np.array([intersection[0][0], intersection[0][1], intersection[0][2]]))
        points_crossing_plane.append(np.array([intersection[1][0], intersection[1][1], intersection[1][2]]))

    return points_crossing_plane


def find_target_point1(points):
    """
    查找前挡上端点
    """
    if points is None or len(points) == 0:
        return None
    if isinstance(points, np.ndarray):
        points = points.tolist()
    sorted_points = sorted(points, key=lambda p: p[0])
    prev_point = sorted_points[0]
    for current_point in sorted_points[2:]:
        if current_point[2] < prev_point[2]:
            return prev_point
        prev_point = current_point
    return sorted_points[-1]


def get_top3_points_by_x(points):
    """
    获取按x排序的前3个不同点
    """
    sorted_points = sorted(points, key=lambda point: point[0])
    seen = set()
    result = []
    for point in sorted_points:
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)
            if len(result) == 3:
                break
    return result


def linear_fit_xy(points_3d):
    """
    对三维点集进行XZ平面的线性拟合
    """
    points_3d = np.array(points_3d)
    x = points_3d[:, 0]
    z = points_3d[:, 2]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, z, rcond=None)[0]
    y_fit = slope * x + intercept
    return (slope, intercept), y_fit


def calculate_z_distances_to_line(points_3d, line_params):
    """
    计算三维点的z值到拟合直线的垂直距离
    """
    slope, intercept = line_params
    points_3d = np.array(points_3d)
    x = points_3d[:, 0]
    z = points_3d[:, 2]
    distances = np.abs(z - (slope * x + intercept)) / np.sqrt(1 + slope ** 2)
    return distances


def find_closest_point(points_3d, distances):
    """
    找出距离拟合直线最近的点
    """
    min_index = np.argmin(distances)
    closest_point = points_3d[min_index]
    min_distance = distances[min_index]
    return min_index, closest_point, min_distance


def calculate(stl_path):
    """
    核心计算函数：计算前风挡与车顶棚连接面差
    """
    min_distance = None  # 最终返回的面差值

    # 1. 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 2. 前置数据处理（为计算距离最小点做准备）
    vertices = mesh.vertices
    z_min = np.min(vertices[:, 2])
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    y_min = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    y_max = filtered_vertices[np.argmax(filtered_vertices[:, 1])]
    y_mid = (y_max[1] + y_min[1]) / 2

    # 3. 获取交线轮廓点
    contour_points = get_contour_from_stl(mesh, y_mid, 'xz')
    contour_points = np.array(contour_points)

    # 4. 筛选前挡部分并找到前挡上端点
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])
    contour_points1 = contour_points[
        (contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] < (x_min + x_max) / 2)]
    target1 = find_target_point1(contour_points1)

    # 5. 线性拟合并计算距离（增加空值判断，保持语法合规）
    if target1 is not None:
        top1 = get_top3_points_by_x(contour_points1[(contour_points1[:, 0] > (target1[0] + 10))])
        if len(top1) > 0:
            (slope, intercept), y_fit = linear_fit_xy(top1)
            points_3d = contour_points1[(contour_points1[:, 0] < target1[0])]
            if len(points_3d) > 0:
                distances = calculate_z_distances_to_line(points_3d, (slope, intercept))
                # 6. 找出距离最小的点（核心输出部分）
                min_index, closest_point, min_distance = find_closest_point(points_3d, distances)
                # print(f"前风挡与车顶棚连接面差: {min_distance:.2f}")
        else:
            pass  # 空操作，保持语法合规
    else:
        pass  # 空操作，保持语法合规

    return min_distance


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)