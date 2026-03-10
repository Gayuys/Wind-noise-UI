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
    stl_file: STL 文件路径
    plane_z: 平面 z 坐标
    返回: 交线段列表（每段为两个点的数组）
    """
    # 加载 STL 文件
    mesh = mesh

    # 定义平面：法向量 (0, 0, 1)，原点 (0, 0, plane_z)
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
    # 返回交线段（每段为两个点 [start, end]）
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

    return points_crossing_plane


def find_target_point1(points):
    """
    从y值最小点开始向上查找，当x值首次增大时返回上一个点，仅考虑y值<=最小y值的点
    :param points: 三维点列表或数组，每个点为(x, y, z)
    :return: 目标点坐标，如果没有找到则返回None
    """
    # 处理空输入
    if points is None or len(points) == 0:
        return None

    # 将NumPy数组转换为列表
    if isinstance(points, np.ndarray):
        points = points.tolist()

    # 按x值升序排序
    sorted_points = sorted(points, key=lambda p: p[0])

    # 初始化前一个点为x值最小的点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[2:]:
        if current_point[2] < prev_point[2]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递增，则返回最后一个点
    return sorted_points[-1]


def find_target_point2(points):
    """
    从y值最小点开始向上查找，当x值首次增大时返回上一个点，仅考虑y值<=最小y值的点
    :param points: 三维点列表或数组，每个点为(x, y, z)
    :return: 目标点坐标，如果没有找到则返回None
    """
    # 处理空输入
    if points is None or len(points) == 0:
        return None

    # 将NumPy数组转换为列表
    if isinstance(points, np.ndarray):
        points = points.tolist()

    # 按x值降序排序
    sorted_points = sorted(points, key=lambda p: p[0], reverse=True)

    # 初始化前一个点为x值最大的点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[2:]:
        if current_point[2] < prev_point[2]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递增，则返回最后一个点
    return sorted_points[-1]


def max_z_difference(point1, point2, point3, target_index):
    """
    计算三个点中目标点的z值与其他两点z值的最大绝对差值

    参数:
    point1, point2, point3: 三维点，格式为(x, y, z)
    target_index: 目标点的索引 (0, 1, 或 2)，指定哪个点作为基准

    返回:
    float: 最大绝对差值
    """
    # 根据索引选择目标点
    points = [point1, point2, point3]
    target_point = points[target_index]

    # 提取目标点的z值
    target_z = target_point[2]

    # 计算与其他两点z值的绝对差值
    diffs = []
    for i, point in enumerate(points):
        if i != target_index:  # 跳过目标点自身
            diffs.append(abs(target_z - point[2]))

    # 返回最大差值
    return max(diffs)


def get_top3_points_by_x(points):
    # 按x坐标从小到大排序
    sorted_points = sorted(points, key=lambda point: point[0])

    # 使用集合记录已出现的点
    seen = set()
    result = []

    # 遍历排序后的点，只添加未出现过的点
    for point in sorted_points:
        point_tuple = tuple(point)  # 转换为元组以便放入集合
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)

            # 已收集到3个不同的点，提前结束
            if len(result) == 3:
                break

    return result


def linear_fit_xy(points_3d):
    """
    对三维点集进行XY平面的线性拟合

    参数:
    points_3d (numpy.ndarray): 形状为(n, 3)的三维点集数组

    返回:
    tuple: 包含斜率和截距的元组，以及拟合直线上的Y值数组
    """
    # 将Python列表转换为NumPy数组
    points_3d = np.array(points_3d)

    # 提取X和Z坐标
    x = points_3d[:, 0]
    z = points_3d[:, 2]

    # 进行线性拟合
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, z, rcond=None)[0]

    # 计算拟合的Y值
    y_fit = slope * x + intercept

    return (slope, intercept), y_fit


def calculate_z_distances_to_line(points_3d, line_params):
    """
    计算三维点的z值到拟合直线的垂直距离

    参数:
    points_3d (numpy.ndarray): 形状为(n, 3)的三维点集数组
    line_params (tuple): 由linear_fit_xy返回的(slope, intercept)元组

    返回:
    numpy.ndarray: 包含每个点到直线的垂直距离的数组
    """
    # 提取直线参数
    slope, intercept = line_params

    # 提取X和Z坐标
    x = points_3d[:, 0]
    z = points_3d[:, 2]

    # 直线的点斜式方程：z = slope * x + intercept
    # 点到直线的距离公式（在 x-z 平面，二维情况）：
    # 距离 = |z - (slope * x + intercept)| / sqrt(1 + slope^2)
    distances = np.abs(z - (slope * x + intercept)) / np.sqrt(1 + slope ** 2)

    return distances


def find_closest_point(points_3d, distances):
    """
    找出距离拟合直线最近的点

    参数:
    points_3d (numpy.ndarray): 形状为(n, 3)的三维点集数组
    distances (numpy.ndarray): 包含每个点到直线的垂直距离的数组

    返回:
    tuple: 包含最小距离点的索引、坐标和最小距离
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

    # 2. 前置数据处理
    vertices = mesh.vertices
    # 找到z轴最小值 # 筛选满足条件的点：z轴大于z轴最小值300mm的点 # 在这些筛选后的点中找y轴最小值对应的点（点1）
    z_min = np.min(vertices[:, 2])
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    y_min = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    y_max = filtered_vertices[np.argmax(filtered_vertices[:, 1])]
    y_mid = (y_max[1] + y_min[1]) / 2
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])

    # 3. 获取交线轮廓点
    plane_z = y_mid
    contour_points = get_contour_from_stl(mesh, plane_z, 'xz')
    contour_points = np.array(contour_points)

    # 4. 筛选前挡部分并找到前挡上端点
    contour_points1 = contour_points[
        (contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] < (x_min + x_max) / 2)]
    points1 = contour_points1
    target1 = find_target_point1(points1)

    # 5. 筛选后挡部分并找到后挡上端点（保留逻辑，未参与最终计算）
    contour_points2 = contour_points[
        (contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] > (x_min + x_max) / 2)]
    points2 = contour_points2
    target2 = find_target_point2(points2)

    # 6. 找出截面的最高点(顶棚)（保留逻辑，未参与最终计算）
    if target1 is not None and target2 is not None:
        contour_points3 = contour_points[
            (contour_points[:, 0] > target1[0] + 100) & (contour_points[:, 0] < (target2[0] - 600))]
        if len(contour_points3) > 0:
            z_max = contour_points3[np.argmax(contour_points3[:, 2])]
        else:
            z_max = None
    else:
        z_max = None

    # 7. 拟合顶棚直线并计算最小距离
    if target1 is not None:
        top1 = get_top3_points_by_x(contour_points1[(contour_points1[:, 0] > (target1[0] + 10))])
        if len(top1) > 0:
            (slope, intercept), y_fit = linear_fit_xy(top1)
            points_3d = contour_points1[(contour_points1[:, 0] < target1[0])]
            if len(points_3d) > 0:
                distances = calculate_z_distances_to_line(points_3d, (slope, intercept))
                # 找出距离最小的点
                min_index, closest_point, min_distance = find_closest_point(points_3d, distances)
                # print(f"前风挡与车顶棚连接面差: {min_distance:.2f}")
        else:
            pass  # 空操作，保持语法合规
    else:
        pass  # 空操作，保持语法合规

    # # 可视化结果（保留注释，匹配示例风格）
    # point_size = 2  # 设置点的大小
    # plt.figure()
    # plt.plot(contour_points[:, 0], contour_points[:, 2], 'o', color='g', markersize=point_size)
    # plt.plot(target1[0], target1[2], 'o', color='r', markersize=5, label='前挡上端点')
    # plt.plot(target2[0], target2[2], 'o', color='b', markersize=5, label='后挡上端点')
    # plt.plot(z_max[0], z_max[2], 'o', color='c', markersize=5, label='截面的最高点(顶棚)')
    # plt.plot(closest_point[0], closest_point[2], 'o', color='k', markersize=5, label='距离最小的点')
    # plt.title('Intersection Points with XZ Plane')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return min_distance


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\小米.stl")
    print(rear_windshield_angle)