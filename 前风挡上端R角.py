import trimesh
import numpy as np
import math

def project_3d_to_2d(points, plane):
    """将三维点投影到二维平面上"""
    if plane == 'xy':
        return np.array([point[:2] for point in points])
    elif plane == 'xz':
        return np.array([point[[0, 2]] for point in points])
    elif plane == 'yz':
        return np.array([point[1:] for point in points])
    else:
        raise ValueError("无效的平面参数，请选择 'xy', 'xz' 或 'yz'")

def get_contour_from_stl(mesh, plane_z, plane):
    """使用 trimesh 计算 STL 文件与平面的交线轮廓"""
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

def find_target_point1(points):
    """从y值最大点开始向下查找，当x值首次减小时返回上一个点"""
    if points is None or len(points) == 0:
        return None

    if isinstance(points, np.ndarray):
        points = points.tolist()

    sorted_points = sorted(points, key=lambda p: p[1], reverse=True)
    prev_point = sorted_points[0]

    for current_point in sorted_points[1:]:
        if current_point[0] < prev_point[0]:
            return prev_point
        prev_point = current_point

    return None

def find_target_point2(points):
    """从x值最小点开始向上查找，当x值首次增大时返回上一个点"""
    if points is None or len(points) == 0:
        return None

    if isinstance(points, np.ndarray):
        points = points.tolist()

    sorted_points = sorted(points, key=lambda p: p[0])
    prev_point = sorted_points[0]

    for current_point in sorted_points[1:]:
        if current_point[1] > prev_point[1]:
            return prev_point
        prev_point = current_point

    return sorted_points[-1]

def find_target_point3(points):
    """从x值最小点开始向上查找，当x值首次增大时返回上一个点"""
    if points is None or len(points) == 0:
        return None

    if isinstance(points, np.ndarray):
        points = points.tolist()

    sorted_points = sorted(points, key=lambda p: p[0])
    prev_point = sorted_points[0]

    for current_point in sorted_points[1:]:
        if current_point[1] < prev_point[1]:
            return prev_point
        prev_point = current_point

    return sorted_points[-1]

def get_top3_points_by_x1(points):
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

def get_top3_points_by_x2(points):
    sorted_points = sorted(points, key=lambda point: point[1])
    seen = set()
    result = []
    for point in sorted_points:
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)
            if len(result) == 5:
                break
    return result

def get_top3_points_by_x3(points):
    sorted_points = sorted(points, key=lambda point: point[0], reverse=True)
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

def get_top3_points_by_x4(points):
    sorted_points = sorted(points, key=lambda point: point[0])
    seen = set()
    result = []
    for point in sorted_points:
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)
            if len(result) == 5:
                break
    return result

def linear_fit_xy(points_3d):
    """对三维点集进行XY平面的线性拟合"""
    points_3d = np.array(points_3d)
    x =  points_3d[:, 0]
    y =  points_3d[:, 1]

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_fit = slope * x + intercept

    return (slope, intercept), y_fit

def calculate_angle_between_lines(slope1, slope2, degrees=True):
    """计算两条直线之间的夹角"""
    tan_angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
    angle_rad = np.arctan(tan_angle)

    if degrees:
        return np.degrees(angle_rad)
    else:
        return angle_rad

def get_angle_from_slope(slope):
    """将斜率转换为角度（度）"""
    return np.degrees(np.arctan(slope))

def calculate_line_length(points):
    """计算点集首尾两点形成的直线段长度"""
    if len(points) < 2:
        return 0.0
    first_point = np.array(points[0][:2])
    last_point = np.array(points[-1][:2])
    return np.linalg.norm(last_point - first_point)

def get_best_fit_points(points):
    """逐步增加点数量进行直线拟合，直到角度变化超过阈值"""
    sorted_points = sorted(points, key=lambda point: point[0])

    if len(sorted_points) < 3:
        raise ValueError("至少需要3个点进行拟合")

    best_points = []
    prev_angle = None
    threshold = 1

    for n in range(3, len(sorted_points) + 1):
        current_points = sorted_points[:n]
        (slope, _), _ = linear_fit_xy(current_points)
        current_angle = get_angle_from_slope(slope)

        if prev_angle is not None:
            angle_diff = abs(current_angle - prev_angle)
            if angle_diff > threshold:
                line_length = calculate_line_length(best_points)
                return best_points, line_length

        best_points = current_points
        prev_angle = current_angle if n == 3 else prev_angle

    line_length = calculate_line_length(best_points)
    return best_points, line_length

def calculate_min_angle_with_y_axis(point1, point2):
    """计算三维两点在XY平面投影的连线与Y轴的最小绝对夹角（0°~90°）"""
    x1, y1, _ = point1
    x2, y2, _ = point2

    delta_x = x2 - x1
    delta_y = y2 - y1

    if delta_x == 0 and delta_y == 0:
        raise ValueError("两点在XY平面的投影重合，无法计算角度")

    angle_with_x_rad = math.atan2(delta_y, delta_x)
    angle_with_y_rad = math.pi / 2 - angle_with_x_rad
    angle_with_y_deg = math.degrees(angle_with_y_rad) % 360

    min_angle_deg = angle_with_y_deg % 180
    if min_angle_deg > 90:
        min_angle_deg = 180 - min_angle_deg

    return min_angle_deg

def calculate(stl_path):
    upper_r_angle = None
    # 核心计算逻辑
    mesh = trimesh.load_mesh(stl_path)
    points = mesh.vertices
    middle_point_y = (max(points[:, 1]) + min(points[:, 1])) / 2
    plane_z = (max(points[:, 1]) + min(points[:, 1])) / 2
    y_min_point = points[np.argmin(points[:, 1])]
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    z = abs(z_max - z_min)

    # 获取交线轮廓
    segments, intersections = get_contour_from_stl(mesh, plane_z, 'xz')
    segments = np.unique(segments, axis=0)
    segments = np.array(segments)

    # 投影到XZ平面
    xz_projection = project_3d_to_2d(segments, 'xz')
    xz_projection = np.unique(xz_projection, axis=0)

    point_front = min(xz_projection[:, 0])
    point_back = max(xz_projection[:, 0])
    middle = (point_front + point_back) / 2

    point_settle = segments[(segments[:, 0] >= y_min_point[0] - 1000) & (segments[:, 0] <= middle) & (segments[:, 2] > (z_min + 300))]

    # 找出引擎盖后端点
    max_point = find_target_point3(point_settle)

    # 筛选满足条件的点
    filtered_vertices = point_settle[point_settle[:, 2] > (z_min + 300)]
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 设置平面 z 坐标
    plane_z1 = max_point[2] + z * 0.23
    # 获取与 XY 平面的交点
    contour_points,_ = get_contour_from_stl(mesh, plane_z1, 'xy')
    contour_points = np.array(contour_points)

    # 找出关键点位
    x_min = np.min(contour_points[:, 0])
    x_min_point = contour_points[np.argmin(contour_points[:, 0])]
    y_min = np.min(contour_points[:, 1])
    y_max = np.max(contour_points[:, 1])
    contour_points1 = contour_points[(contour_points[:, 1] < ((y_min + y_max)/2 - 500)) & (contour_points[:, 0] < (x_min + 600))]
    dian_1 = find_target_point1(contour_points1)

    # 找出点2（A柱的上端X向的最小点）
    contour_points2 = contour_points1[(contour_points1[:, 1] <= dian_1[1])]
    dian_2 = contour_points2[np.argmin(contour_points2[:, 0])]

    # 上端R角计算和输出
    point_a = x_min_point
    point_b = dian_2
    upper_r_angle = calculate_min_angle_with_y_axis(point_a, point_b)

    # 输出上端R角
    #print(f"上端R角：{upper_r_angle:.2f}°")
    return upper_r_angle

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)