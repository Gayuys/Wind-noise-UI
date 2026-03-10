import trimesh
import numpy as np
import matplotlib.pyplot as plt
import math

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

    return points_crossing_plane, intersections

def find_target_point(points):
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

    # 计算最小y值
    min_y = min(p[1] for p in points)
    # 找出最小y值的点坐标
    min_y_point = next((p for p in points if p[1] == min_y), None)

    # 筛选出所有x值小于等于最小y值的点x值
    filtered_points = [p for p in points if p[0] <= min_y_point[0]]

    # 如果没有这样的点，返回None
    if not filtered_points:
        return None

    # 按y值升序排序
    sorted_points = sorted(filtered_points, key=lambda p: p[1])

    # 初始化前一个点为y值最小的点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[1:]:
        if current_point[0] > prev_point[0]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递增，则返回最后一个点
    return sorted_points[-1]

def find_target_point1(points):
    """
    从y值最大点开始向下查找，当x值首次减小时返回上一个点，仅考虑所有输入点（按y值从大到小遍历）
    :param points: 三维点列表或数组，每个点为(x, y, z)
    :return: 目标点坐标（找到符合条件的点时），None（输入无效或未找到符合条件的点时）
    """
    # 处理空输入，返回空值None
    if points is None or len(points) == 0:
        return None

    # 将NumPy数组转换为列表
    if isinstance(points, np.ndarray):
        points = points.tolist()

    # 按y值降序排序（从大到小）
    sorted_points = sorted(points, key=lambda p: p[1], reverse=True)

    # 初始化前一个点为y值最大的点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[1:]:
        # 判断当前点x值是否比上一个点x值小，首次满足时返回上一个点
        if current_point[0] < prev_point[0]:
            return prev_point
        prev_point = current_point

    # 关键修改：遍历完所有点都未找到符合条件的点，返回空值None
    # 替代了原逻辑中返回sorted_points[-1]的行为
    return None

def find_target_point2(points):
    """
    从x值最小点开始向上查找，当x值首次增大时返回上一个点，仅考虑y值>=最小y值的点
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
    for current_point in sorted_points[1:]:
        if current_point[1] > prev_point[1]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递增，则返回最后一个点
    return sorted_points[-1]

def find_target_point3(points):
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

    # 按y值升序排序
    sorted_points = sorted(points, key=lambda p: p[1])

    # 初始化前一个点为y值最小的点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[1:]:
        if current_point[0] > prev_point[0]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递增，则返回最后一个点
    return sorted_points[-1]

def find_target_point4(points):
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

    # 按y值升序排序
    sorted_points = sorted(points, key=lambda p: p[1])

    # 初始化前一个点为y值最小的点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[1:]:
        if current_point[0] < prev_point[0]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递增，则返回最后一个点
    return sorted_points[-1]

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

    # 提取X和Y坐标
    x =  points_3d[:, 0]
    y =  points_3d[:, 1]

    # 进行线性拟合
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # 计算拟合的Y值
    y_fit = slope * x + intercept

    return (slope, intercept), y_fit

def calculate_angle_between_lines(slope1, slope2, degrees=True):
    """
    计算两条直线之间的夹角

    参数:
    slope1 (float): 第一条直线的斜率
    slope2 (float): 第二条直线的斜率
    degrees (bool): 是否以角度制返回结果（默认为True）

    返回:
    float: 两条直线之间的夹角
    """
    # 计算夹角的正切值
    tan_angle = abs((slope2 - slope1) / (1 + slope1 * slope2))

    # 计算夹角（弧度）
    angle_rad = np.arctan(tan_angle)

    # 转换为角度（如果需要）
    if degrees:
        return np.degrees(angle_rad)
    else:
        return angle_rad

def get_top3_points_by_x1(points):
    # 按x坐标从大到小排序
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

def get_top3_points_by_x2(points):
    # 按y坐标从小到大排序
    sorted_points = sorted(points, key=lambda point: point[1])

    # 使用集合记录已出现的点
    seen = set()
    result = []

    # 遍历排序后的点，只添加未出现过的点
    for point in sorted_points:
        point_tuple = tuple(point)  # 转换为元组以便放入集合
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)

            # 已收集到10个不同的点，提前结束
            if len(result) == 5:
                break

    return result

def get_top3_points_by_x3(points):
    # 按x坐标从大到小排序
    sorted_points = sorted(points, key=lambda point: point[0], reverse=True)

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

def get_top3_points_by_x4(points):
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
            if len(result) == 5:
                break

    return result


def get_angle_from_slope(slope):
    """将斜率转换为角度（度）"""
    return np.degrees(np.arctan(slope))


def calculate_line_length(points):
    """计算点集首尾两点形成的直线段长度"""
    if len(points) < 2:
        return 0.0
    first_point = np.array(points[0][:2])  # 取前两列(X,Y)
    last_point = np.array(points[-1][:2])
    return np.linalg.norm(last_point - first_point)


def get_best_fit_points(points):
    """
    逐步增加点数量进行直线拟合，直到角度变化超过阈值，并计算最佳拟合直线的长度

    参数:
    points (list): 待拟合的点列表，每个点是一个包含三个坐标的列表或元组

    返回:
    tuple: 包含最佳拟合的点列表和拟合直线的长度
    """
    # 按x坐标从小到大排序
    sorted_points = sorted(points, key=lambda point: point[0])

    if len(sorted_points) < 3:
        raise ValueError("至少需要3个点进行拟合")

    # 初始化
    best_points = []
    prev_angle = None
    threshold = 0.5  # 角度变化阈值（度）

    # 从3个点开始，逐步增加点数量
    for n in range(3, len(sorted_points) + 1):
        current_points = sorted_points[:n]

        # 拟合直线
        (slope, _), _ = linear_fit_xy(current_points)
        current_angle = get_angle_from_slope(slope)

        # 计算与初始角度（3个点）的差值
        if prev_angle is not None:
            angle_diff = abs(current_angle - prev_angle)

            # 如果角度变化超过阈值，返回上一组点
            if angle_diff > threshold:
                line_length = calculate_line_length(best_points)
                return best_points, line_length

        # 更新最佳点集和角度
        best_points = current_points
        prev_angle = current_angle if n == 3 else prev_angle  # 仅在第一次记录3个点的角度

    # 如果所有点的角度变化都未超过阈值，返回所有点
    line_length = calculate_line_length(best_points)
    return best_points, line_length

def calculate_min_angle_with_y_axis(point1, point2):
    """
    计算三维两点在XY平面投影的连线与Y轴的**最小绝对夹角**（0°~90°）

    参数:
        point1: 第一个点的三维坐标，元组格式 (x1, y1, z1)
        point2: 第二个点的三维坐标，元组格式 (x2, y2, z2)

    返回:
        min_angle_deg: 与Y轴的最小绝对夹角（角度制，范围：0° ~ 90°）
    """
    # 提取XY平面坐标，忽略z轴
    x1, y1, _ = point1
    x2, y2, _ = point2

    # 计算XY平面的向量分量
    delta_x = x2 - x1
    delta_y = y2 - y1

    # 处理特殊情况：两点XY坐标重合（无意义的直线）
    if delta_x == 0 and delta_y == 0:
        raise ValueError("两点在XY平面的投影重合，无法计算角度")

    # 1. 先计算与X轴的夹角（弧度制）
    angle_with_x_rad = math.atan2(delta_y, delta_x)

    # 2. 转换为与Y轴的夹角（弧度制）
    angle_with_y_rad = math.pi / 2 - angle_with_x_rad

    # 3. 转换为角度制，并调整到0°~360°范围
    angle_with_y_deg = math.degrees(angle_with_y_rad) % 360

    # 4. 核心修正：取与Y轴的最小绝对夹角（0°~90°）
    # 逻辑：若角度>90°，则取180°-角度；若角度>180°，先取模180再判断
    min_angle_deg = angle_with_y_deg % 180  # 先将角度归到0°~180°
    if min_angle_deg > 90:
        min_angle_deg = 180 - min_angle_deg  # 取最小夹角（如173°→7°，100°→80°）

    return min_angle_deg

def calculate(stl_path):
    A_X = None
    # 读取STL文件，需填入你的STL文件路径
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标
    vertices = mesh.vertices

    # 找到z轴最小值 # 筛选满足条件的点：z轴大于z轴最小值300mm的点 # 在这些筛选后的点中找y轴最小值对应的点（点1）
    z_min = np.min(vertices[:, 2])
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 设置平面 z 坐标
    plane_z = point1[2]
    # 获取与 XZ 平面的交点
    contour_points, _ = get_contour_from_stl(mesh, plane_z, 'xy')
    # 转换为 NumPy 数组以便处理
    contour_points = np.array(contour_points)

    # 找出点1（A柱与前风挡交点）
    x_min = np.min(contour_points[:, 0])
    x_min_point = contour_points[np.argmin(contour_points[:, 0])]
    y_min = np.min(contour_points[:, 1])
    y_max = np.max(contour_points[:, 1])
    contour_points1 = contour_points[(contour_points[:, 1] < ((y_min + y_max)/2 - 600)) & (contour_points[:, 0] < (x_min + 600)) & (contour_points[:, 1] > (point1[1] + 250))]
    points = contour_points1
    dian_1 = find_target_point1(points)

    # 找出点2（A柱的上端X向的最小点）
    contour_points2 = contour_points1[(contour_points1[:, 1] <= dian_1[1])]
    dian_2 = contour_points2[np.argmin(contour_points2[:, 0])]

    # 判断点3（若A柱存在亮条）
    contour_points3 = contour_points2[(contour_points2[:, 1] <= dian_2[1]) & (contour_points2[:, 1] >= (dian_2[1] - 25)) & (contour_points2[:, 0] <= (dian_2[0] + 30))]
    points = contour_points3
    dian_3 = find_target_point1(points)

    # 根据点3是否找到执行不同逻辑
    if dian_3 is not None:
        # 点3找到，执行套1逻辑
        #print("点3已找到，执行套1逻辑")

        # 找出点4（若A柱存在亮条，A柱上端与亮条X向的最小点）
        contour_points4 = contour_points2[(contour_points2[:, 1] < dian_3[1])]
        dian_4_1 = contour_points4[np.argmin(contour_points4[:, 0])]

        # 找出点5（A柱Y向的最小点）
        contour_points5 = contour_points2[(contour_points2[:, 1] < dian_4_1[1])]
        points = contour_points5
        dian_5_1 = find_target_point2(points)

        # 找出点6（亮条Y点）
        contour_points6 = contour_points2[(contour_points2[:, 0] > (dian_5_1[0] + 30)) & (contour_points2[:, 0] < (dian_5_1[0] + 100))]
        dian_6_1 = contour_points6[np.argmin(contour_points6[:, 1])]

        # 找出点7（侧窗）点
        contour_points7 = contour_points2[(contour_points2[:, 0] > dian_6_1[0])]
        dian_7_1 = contour_points7[np.argmax(contour_points7[:, 1])]

        # 找出A柱与前挡夹角的直线的点
        top1_1 = get_top3_points_by_x1(contour_points1[(contour_points1[:, 1] <= dian_2[1])])
        top1_2 = get_top3_points_by_x2(contour_points1[(contour_points1[:, 1] >= (dian_1[1] + 10))])
        # 第一条直线
        points_3d_line1 = top1_1
        # 第二条直线
        points_3d_line2 = top1_2
        # 拟合两条直线
        (slope1, intercept1), y_fit1 = linear_fit_xy(points_3d_line1)
        (slope2, intercept2), y_fit2 = linear_fit_xy(points_3d_line2)
        # 计算两条直线之间的夹角
        angle1 = calculate_angle_between_lines(slope1, slope2)

        # 找出A柱与侧窗夹角的直线的点
        top2_1 = get_top3_points_by_x3(contour_points1[contour_points1[:, 0] <= dian_5_1[0]])
        top2_2 = get_top3_points_by_x4(contour_points1[(contour_points1[:, 0] >= (dian_7_1[0] + 10))])
        # 第一条直线
        points_3d_line3 = top2_1
        # 第二条直线
        points_3d_line4 = top2_2
        # 拟合两条直线
        (slope3, intercept3), y_fit3 = linear_fit_xy(points_3d_line3)
        (slope4, intercept4), y_fit4 = linear_fit_xy(points_3d_line4)
        # 计算两条直线之间的夹角
        angle2 = calculate_angle_between_lines(slope3, slope4)

        # 平行段
        contour_points7 = contour_points1[(contour_points1[:, 1] <= dian_2[1])]
        points = contour_points7
        best_points, line_length = get_best_fit_points(points)

        # A柱上端X向尺寸
        A_X = abs(dian_4_1[0] - dian_5_1[0])

        # A柱上端Y向尺寸
        A_Y = abs(dian_4_1[1] - dian_5_1[1])

        # A柱上端与前风挡阶差
        A_Q = abs(dian_2[0] - dian_1[0])

        # A柱上端与亮条挡阶差
        A_L = abs(dian_5_1[1] - dian_6_1[1])

        # 亮条上端与侧窗挡阶差
        L_C = abs(dian_6_1[1] - dian_7_1[1])

        # 下端R角
        point_a = x_min_point
        point_b = dian_2
        angle = calculate_min_angle_with_y_axis(point_a, point_b)
        #print(f"两点XY连线与Y轴的夹角：{angle:.2f}°")

        # # 可视化结果
        # point_size = 2  # 设置点的大小
        # plt.figure()
        # plt.plot(contour_points1[:, 0], contour_points1[:, 1], 'o', color='g', markersize=point_size)
        # plt.plot(dian_1[0], dian_1[1], 'o', color='c', markersize=5, label='点 1')
        # plt.plot(dian_2[0], dian_2[1], 'o', color='m', markersize=5, label='点 2')
        # plt.plot(dian_3[0], dian_3[1], 'o', color='r', markersize=5, label='点 3')
        # plt.plot(dian_4_1[0], dian_4_1[1], 'o', color='b', markersize=5, label='点 4')
        # plt.plot(dian_5_1[0], dian_5_1[1], 'o', color='orange', markersize=5, label='点 5')
        # plt.plot(dian_6_1[0], dian_6_1[1], 'o', color='y', markersize=5, label='点 6')
        # plt.plot(dian_7_1[0], dian_7_1[1], 'o', color='k', markersize=5, label='点 7')
        # # 提取拟合直线的X坐标范围
        # x_range1 = np.array([min(p[0] for p in points_3d_line1), max(p[0] for p in points_3d_line1)])
        # x_range2 = np.array([min(p[0] for p in points_3d_line2), max(p[0] for p in points_3d_line2)])
        # # 计算拟合直线的Y值
        # y_range1 = slope1 * x_range1 + intercept1
        # y_range2 = slope2 * x_range2 + intercept2
        # # 绘制拟合直线
        # plt.plot(x_range1, y_range1, 'b--', linewidth=2,
        #         label=f'拟合直线1: y = {slope1:.2f}x + {intercept1:.2f}')
        # plt.plot(x_range2, y_range2, 'k--', linewidth=2,
        #         label=f'拟合直线2: y = {slope2:.2f}x + {intercept2:.2f}')
        # # 提取拟合直线的X坐标范围
        # x_range3 = np.array([min(p[0] for p in points_3d_line3), max(p[0] for p in points_3d_line3)])
        # x_range4 = np.array([min(p[0] for p in points_3d_line4), max(p[0] for p in points_3d_line4)])
        # # 计算拟合直线的Y值
        # y_range3 = slope3 * x_range3 + intercept3
        # y_range4 = slope4 * x_range4 + intercept4
        # # 绘制拟合直线
        # plt.plot(x_range3, y_range3, 'b--', linewidth=2,
        #         label=f'拟合直线1: y = {slope3:.2f}x + {intercept3:.2f}')
        # plt.plot(x_range4, y_range4, 'k--', linewidth=2,
        #         label=f'拟合直线2: y = {slope4:.2f}x + {intercept4:.2f}')
        # plt.title('Intersection Points with XZ Plane')
        # plt.xlabel('X-axis')
        # plt.ylabel('Z-axis')
        # plt.axis('equal')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

    else:
        # 点3未找到，执行套2逻辑
        #print("点3未找到，执行套2逻辑")

        # 找出点5（A柱Y向的最小点）
        contour_points5 = contour_points2[(contour_points2[:, 1] < dian_2[1])]
        points = contour_points5
        dian_5_2 = find_target_point2(points)

        # 找出点6（亮条Y点）
        contour_points6 = contour_points2[(contour_points2[:, 0] > (dian_5_2[0] + 30)) & (contour_points2[:, 0] < (dian_5_2[0] + 100))]
        dian_6_2 = contour_points6[np.argmin(contour_points6[:, 1])]

        # 找出点7（侧窗）点
        contour_points7 = contour_points2[(contour_points2[:, 0] > dian_6_2[0])]
        dian_7_2 = contour_points7[np.argmax(contour_points7[:, 1])]

        # 找出A柱与前挡夹角的直线的点
        top1_1 = get_top3_points_by_x1(contour_points1[(contour_points1[:, 1] <= dian_2[1])])
        top1_2 = get_top3_points_by_x2(contour_points1[(contour_points1[:, 1] >= (dian_1[1] + 10))])
        # 第一条直线
        points_3d_line1 = top1_1
        # 第二条直线
        points_3d_line2 = top1_2
        # 拟合两条直线
        (slope1, intercept1), y_fit1 = linear_fit_xy(points_3d_line1)
        (slope2, intercept2), y_fit2 = linear_fit_xy(points_3d_line2)
        # 计算两条直线之间的夹角
        angle1 = calculate_angle_between_lines(slope1, slope2)

        # 找出A柱与侧窗夹角的直线的点
        top2_1 = get_top3_points_by_x3(contour_points1[contour_points1[:, 0] <= dian_5_2[0]])
        top2_2 = get_top3_points_by_x4(contour_points1[(contour_points1[:, 0] >= (dian_7_2[0] + 10))])
        # 第一条直线
        points_3d_line3 = top2_1
        # 第二条直线
        points_3d_line4 = top2_2
        # 拟合两条直线
        (slope3, intercept3), y_fit3 = linear_fit_xy(points_3d_line3)
        (slope4, intercept4), y_fit4 = linear_fit_xy(points_3d_line4)
        # 计算两条直线之间的夹角
        angle2 = calculate_angle_between_lines(slope3, slope4)

        # 平行段
        contour_points7 = contour_points1[(contour_points1[:, 1] <= dian_2[1])]
        points = contour_points7
        best_points, line_length = get_best_fit_points(points)

        # A柱上端X向尺寸
        A_X = abs(dian_2[0] - dian_5_2[0])

        # A柱上端Y向尺寸
        A_Y = abs(dian_2[1] - dian_5_2[1])

        # A柱上端与前风挡阶差
        A_Q = abs(dian_2[0] - dian_1[0])

        # A柱上端与亮条挡阶差
        A_L = abs(dian_5_2[1] - dian_6_2[1])

        # 亮条上端与侧窗挡阶差
        L_C = abs(dian_6_2[1] - dian_7_2[1])

        # 下端R角
        point_a = x_min_point
        point_b = dian_2
        angle = calculate_min_angle_with_y_axis(point_a, point_b)
        #print(f"两点XY连线与Y轴的夹角：{angle:.2f}°")

        # # 可视化结果
        # point_size = 2  # 设置点的大小
        # plt.figure()
        # plt.plot(contour_points1[:, 0], contour_points1[:, 1], 'o', color='g', markersize=point_size)
        # plt.plot(dian_1[0], dian_1[1], 'o', color='c', markersize=5, label='点 1')
        # plt.plot(dian_2[0], dian_2[1], 'o', color='m', markersize=5, label='点 2')
        # plt.plot(dian_5_2[0], dian_5_2[1], 'o', color='b', markersize=5, label='点 4')
        # plt.plot(dian_6_2[0], dian_6_2[1], 'o', color='orange', markersize=5, label='点 5')
        # plt.plot(dian_7_2[0], dian_7_2[1], 'o', color='y', markersize=5, label='点 6')
        # # 提取拟合直线的X坐标范围
        # x_range1 = np.array([min(p[0] for p in points_3d_line1), max(p[0] for p in points_3d_line1)])
        # x_range2 = np.array([min(p[0] for p in points_3d_line2), max(p[0] for p in points_3d_line2)])
        # # 计算拟合直线的Y值
        # y_range1 = slope1 * x_range1 + intercept1
        # y_range2 = slope2 * x_range2 + intercept2
        # # 绘制拟合直线
        # plt.plot(x_range1, y_range1, 'b--', linewidth=2,
        #         label=f'拟合直线1: y = {slope1:.2f}x + {intercept1:.2f}')
        # plt.plot(x_range2, y_range2, 'k--', linewidth=2,
        #         label=f'拟合直线2: y = {slope2:.2f}x + {intercept2:.2f}')
        # # 提取拟合直线的X坐标范围
        # x_range3 = np.array([min(p[0] for p in points_3d_line3), max(p[0] for p in points_3d_line3)])
        # x_range4 = np.array([min(p[0] for p in points_3d_line4), max(p[0] for p in points_3d_line4)])
        # # 计算拟合直线的Y值
        # y_range3 = slope3 * x_range3 + intercept3
        # y_range4 = slope4 * x_range4 + intercept4
        # # 绘制拟合直线
        # plt.plot(x_range3, y_range3, 'b--', linewidth=2,
        #         label=f'拟合直线1: y = {slope3:.2f}x + {intercept3:.2f}')
        # plt.plot(x_range4, y_range4, 'k--', linewidth=2,
        #         label=f'拟合直线2: y = {slope4:.2f}x + {intercept4:.2f}')
        # plt.title('Intersection Points with XZ Plane')
        # plt.xlabel('X-axis')
        # plt.ylabel('Z-axis')
        # plt.axis('equal')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

    # 输出A柱下端X向尺寸
    #print(f"A柱下端X向尺寸: {A_X:.2f}")
    return A_X

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)