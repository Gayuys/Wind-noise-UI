import trimesh
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def project_3d_to_2d(points, plane):
    if plane == 'xy':
        return np.array([point[:2] for point in points])
    elif plane == 'xz':
        return np.array([point[[0, 2]] for point in points])
    elif plane == 'yz':
        return np.array([point[1:] for point in points])
    else:
        raise ValueError("无效的平面参数，请选择 'xy', 'xz' 或 'yz'")

def get_contour_from_stl(mesh, plane_z, plane):
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

def linear_fit_xy(points_3d):
    points_3d = np.array(points_3d)
    x =  points_3d[:, 0]
    y =  points_3d[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_fit = slope * x + intercept
    return (slope, intercept), y_fit

def get_angle_from_slope(slope):
    return np.degrees(np.arctan(slope))

def calculate_line_length(points):
    if len(points) < 2:
        return 0.0
    first_point = np.array(points[0][:2])
    last_point = np.array(points[-1][:2])
    return np.linalg.norm(last_point - first_point)

def get_best_fit_points(points):
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

def calculate(stl_path):
    line_length = None
    # 加载 STL 文件并获取点云数据
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
    points = point_settle
    max_point = find_target_point2(points)

    # 筛选点并找point1
    filtered_vertices = points[points[:, 2] > (z_min + 300)]
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 设置平面 z 坐标并获取轮廓点
    plane_z1 = max_point[2] + z * 0.23
    contour_points,_ = get_contour_from_stl(mesh, plane_z1, 'xy')
    contour_points = np.array(contour_points)

    # 找出点1（A柱与前风挡交点）
    x_min = np.min(contour_points[:, 0])
    x_min_point = contour_points[np.argmin(contour_points[:, 0])]
    y_min = np.min(contour_points[:, 1])
    y_max = np.max(contour_points[:, 1])
    contour_points1 = contour_points[(contour_points[:, 1] < ((y_min + y_max)/2 - 500)) & (contour_points[:, 0] < (x_min + 600))]
    points = contour_points1
    dian_1 = find_target_point1(points)

    # 找出点2（A柱的上端X向的最小点）
    contour_points2 = contour_points1[(contour_points1[:, 1] <= dian_1[1])]
    dian_2 = contour_points2[np.argmin(contour_points2[:, 0])]

    # 判断点3（若A柱存在亮条）
    contour_points3 = contour_points2[(contour_points2[:, 1] <= dian_2[1]) & (contour_points2[:, 1] >= (dian_2[1] - 25)) & (contour_points2[:, 0] <= (dian_2[0] + 30))]
    points = contour_points3
    dian_3 = find_target_point1(points)

    # 初始化核心变量
    best_points = []
    line_length = 0.0
    if dian_3 is not None:
        #print("点3已找到，执行套1逻辑")
        # 找点4-7
        contour_points4 = contour_points2[(contour_points2[:, 1] < dian_3[1])]
        dian_4_1 = contour_points4[np.argmin(contour_points4[:, 0])]
        contour_points5 = contour_points2[(contour_points2[:, 1] < dian_4_1[1])]
        points = contour_points5
        dian_5_1 = find_target_point2(points)
        contour_points6 = contour_points2[(contour_points2[:, 0] > (dian_5_1[0] + 30)) & (contour_points2[:, 0] < (dian_5_1[0] + 100))]
        dian_6_1 = contour_points6[np.argmin(contour_points6[:, 1])]
        contour_points7 = contour_points2[(contour_points2[:, 0] > dian_6_1[0]) & (contour_points2[:, 0] < (dian_6_1[0] + 80))]
        dian_7_1 = contour_points7[np.argmax(contour_points7[:, 1])]
        # 计算平行段
        contour_points7 = contour_points1[(contour_points1[:, 1] <= dian_2[1])]
        points = contour_points7
        best_points, line_length = get_best_fit_points(points)
        # 可视化赋值
        d4, d5, d6, d7 = dian_4_1, dian_5_1, dian_6_1, dian_7_1
    else:
        #print("点3未找到，执行套2逻辑")
        # 找点5-7
        contour_points5 = contour_points2[(contour_points2[:, 1] < dian_2[1])]
        points = contour_points5
        dian_5_2 = find_target_point2(points)
        contour_points6 = contour_points2[(contour_points2[:, 0] > (dian_5_2[0] + 30)) & (contour_points2[:, 0] < (dian_5_2[0] + 100))]
        dian_6_2 = contour_points6[np.argmin(contour_points6[:, 1])]
        contour_points7 = contour_points2[(contour_points2[:, 0] > dian_6_2[0]) & (contour_points2[:, 0] < (dian_6_2[0] + 80))]
        dian_7_2 = contour_points7[np.argmax(contour_points7[:, 1])]
        # 计算平行段
        contour_points7 = contour_points1[(contour_points1[:, 1] <= dian_2[1])]
        points = contour_points7
        best_points, line_length = get_best_fit_points(points)
        # 可视化赋值
        d4, d5, d6, d7 = dian_5_2, dian_6_2, dian_7_2, [0,0,0]

    # # 可视化结果（与原代码完全一致）
    # point_size = 2
    # plt.figure()
    # plt.plot(contour_points1[:, 0], contour_points1[:, 1], 'o', color='g', markersize=point_size)
    # plt.plot(dian_1[0], dian_1[1], 'o', color='c', markersize=5, label='点 1')
    # plt.plot(dian_2[0], dian_2[1], 'o', color='m', markersize=5, label='点 2')
    # if dian_3 is not None:
    #     plt.plot(dian_3[0], dian_3[1], 'o', color='r', markersize=5, label='点 3')
    #     plt.plot(d4[0], d4[1], 'o', color='b', markersize=5, label='点 4')
    #     plt.plot(d5[0], d5[1], 'o', color='orange', markersize=5, label='点 5')
    #     plt.plot(d6[0], d6[1], 'o', color='y', markersize=5, label='点 6')
    #     plt.plot(d7[0], d7[1], 'o', color='k', markersize=5, label='点 7')
    # else:
    #     plt.plot(d4[0], d4[1], 'o', color='b', markersize=5, label='点 4')
    #     plt.plot(d5[0], d5[1], 'o', color='orange', markersize=5, label='点 5')
    #     plt.plot(d6[0], d6[1], 'o', color='y', markersize=5, label='点 6')
    # plt.title('Intersection Points with XZ Plane')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # 仅输出：A柱上端平行段
    #print(f"最佳拟合点数量: {len(best_points)}")
    #print(f"A柱上端平行段: {line_length:.2f}")
    
    return line_length

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)