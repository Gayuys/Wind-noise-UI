import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================== 全局配置 ========================
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# ======================== 通用函数定义 ========================
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
    """从x值最小点开始向上查找，当x值首次增大时返回上一个点，仅考虑y值>=最小y值的点"""
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

    def _line_plane_intersection(line_start, line_end):
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
        intersection = _line_plane_intersection(line_start, line_end)
        if intersection is not None:
            intersection_points.append(intersection)
    return np.array(intersection_points)


def calculate_a_pillar_angles(point1, point2):
    """计算A柱的空间方向角和方向余弦"""
    point1 = np.array(point1)
    point2 = np.array(point2)
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]

    magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if magnitude < 1e-9:
        raise ValueError("两点坐标相同，无法计算方向角")

    cos_alpha = dx / magnitude
    cos_beta = dy / magnitude
    cos_gamma = dz / magnitude

    alpha = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))

    length = magnitude
    tilt_angle = np.degrees(np.arcsin(np.clip(dz / magnitude, -1.0, 1.0)))

    return {
        'direction_cosines': (round(cos_alpha, 4), round(cos_beta, 4), round(cos_gamma, 4)),
        'direction_angles': (round(alpha, 2), round(beta, 2), round(gamma, 2)),
        'a_pillar_length': round(length, 2),
        'tilt_angle': round(tilt_angle, 2)
    }


def visualize_a_pillar_3d(lower_point, upper_point,
                          contour_points_lower=None, contour_points_upper=None):
    """三维可视化A柱端点、空间角和交线轮廓"""
    angle_results = calculate_a_pillar_angles(lower_point, upper_point)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制截面轮廓点
    if contour_points_lower is not None and len(contour_points_lower) > 0:
        contour_points_lower = np.array(contour_points_lower)
        ax.scatter(
            contour_points_lower[:, 0],
            contour_points_lower[:, 1],
            contour_points_lower[:, 2],
            c='blue',
            s=5,
            alpha=0.6,
            label='下端截面交线点'
        )

    if contour_points_upper is not None and len(contour_points_upper) > 0:
        contour_points_upper = np.array(contour_points_upper)
        ax.scatter(
            contour_points_upper[:, 0],
            contour_points_upper[:, 1],
            contour_points_upper[:, 2],
            c='green',
            s=5,
            alpha=0.6,
            label='上端截面交线点'
        )

    # 绘制A柱端点
    ax.scatter(
        lower_point[0],
        lower_point[1],
        lower_point[2],
        c='yellow',
        s=150,
        marker='o',
        edgecolors='black',
        linewidth=2,
        label='A柱下端点'
    )
    ax.scatter(
        upper_point[0],
        upper_point[1],
        upper_point[2],
        c='red',
        s=150,
        marker='o',
        edgecolors='black',
        linewidth=2,
        label='A柱上端点'
    )

    # 绘制A柱轴线
    ax.plot(
        [lower_point[0], upper_point[0]],
        [lower_point[1], upper_point[1]],
        [lower_point[2], upper_point[2]],
        'k--',
        linewidth=3,
        alpha=0.8,
        label='A柱轴线'
    )

    # 添加角度信息标注（突出Y轴夹角）
    if angle_results is not None:
        info_text = (
            f"【核心输出】A柱与Y轴夹角: {angle_results['direction_angles'][1]}°\n"
            f"A柱长度: {angle_results['a_pillar_length']} mm\n"
            f"与X轴夹角: {angle_results['direction_angles'][0]}°\n"
            f"与Z轴夹角: {angle_results['direction_angles'][2]}°\n"
            f"倾斜角(与XY平面): {angle_results['tilt_angle']}°"
        )
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # 设置坐标轴和标题
    ax.set_xlabel('X 轴 (mm)', fontsize=11)
    ax.set_ylabel('Y 轴 (mm)', fontsize=11)
    ax.set_zlabel('Z 轴 (mm)', fontsize=11)
    ax.set_title('A柱空间位置与角度分析（聚焦Y轴夹角）', fontsize=14, pad=20)

    # 美化设置
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


# ======================== 主程序（输出A柱与Y轴夹角） ========================
def calculate(stl_path):
    kongjiabnjiao_Y = None
    # 1. 加载STL文件
    mesh = trimesh.load_mesh(stl_path)
    points = mesh.vertices

    a_column_lower_point = None
    a_column_upper_point = None
    max_point = None
    contour_points1_basic = None
    contour_points1_optimized = None

    # 2. 计算车高并找到引擎盖后端点
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    car_height = abs(z_max - z_min)

    middle_point_y = (max(points[:, 1]) + min(points[:, 1])) / 2
    plane_z = middle_point_y
    segments, intersections = get_contour_from_stl(mesh, plane_z, 'xz')

    if segments is not None and intersections is not None:
        segments = np.unique(segments, axis=0)
        xz_projection = project_3d_to_2d(segments, 'xz')
        xz_projection = np.unique(xz_projection, axis=0)

        point_front = min(xz_projection[:, 0])
        point_back = max(xz_projection[:, 0])
        middle = (point_front + point_back) / 2
        point_settle = segments[(segments[:, 0] >= point_front + 500) & (segments[:, 0] <= middle)]

        Step1 = 10
        settle_max = 0
        start = min(point_settle[:, 0])
        for i in range(0, 5000):
            plane1 = (1, 0, 0, (-1 * (start + i * Step1)))
            result1 = line_plane_intersection(intersections, plane1)
            if len(result1) > 0:
                settle_max1 = max(result1[:, 2])
                if settle_max1 - settle_max >= 0:
                    settle_max = settle_max1
                else:
                    Step2 = 0.5
                    settle_max2 = 0
                    result3 = None
                    for a in range(0, 20):
                        start1 = start + (i - 1) * Step1
                        plane2 = (1, 0, 0, (-1 * (start1 + a * Step2)))
                        plane3 = (1, 0, 0, (-1 * (start1 + (a - 1) * Step2)))
                        result2 = line_plane_intersection(intersections, plane2)
                        if len(result2) > 0:
                            settle_max3 = max(result2[:, 2])
                            if settle_max3 - settle_max2 >= 0:
                                settle_max2 = settle_max3
                            else:
                                result3 = line_plane_intersection(intersections, plane3)
                                break
                    if result3 is not None and len(result3) > 0:
                        yz_coords = result3[:, 2:]
                        max_index = np.argmax(yz_coords, axis=0)
                        max_point = result3[max_index][0]
                    break

    # 3. 查找A柱下端点
    filtered_vertices = points[points[:, 2] > (z_min + 300)]
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    plane_z_basic = point1[2]
    contour_points_basic, _ = get_contour_from_stl(mesh, plane_z_basic, 'xy')
    contour_points_basic = np.array(contour_points_basic)

    x_min_basic = np.min(contour_points_basic[:, 0])
    y_min_basic = np.min(contour_points_basic[:, 1])
    y_max_basic = np.max(contour_points_basic[:, 1])
    contour_points1_basic = contour_points_basic[
        (contour_points_basic[:, 1] < ((y_min_basic + y_max_basic) / 2 - 600)) &
        (contour_points_basic[:, 0] < (x_min_basic + 600)) &
        (contour_points_basic[:, 1] > (point1[1] + 250))
        ]
    dian_1_basic = find_target_point1(contour_points1_basic)

    if dian_1_basic is not None:
        contour_points2_basic = contour_points1_basic[(contour_points1_basic[:, 1] <= dian_1_basic[1])]
        dian_2_basic = contour_points2_basic[np.argmin(contour_points2_basic[:, 0])]

        contour_points3_basic = contour_points2_basic[
            (contour_points2_basic[:, 1] <= dian_2_basic[1]) &
            (contour_points2_basic[:, 1] >= (dian_2_basic[1] - 25)) &
            (contour_points2_basic[:, 0] <= (dian_2_basic[0] + 30))
            ]
        dian_3_basic = find_target_point1(contour_points3_basic)

        if dian_3_basic is not None:
            contour_points4_basic = contour_points2_basic[(contour_points2_basic[:, 1] < dian_3_basic[1])]
            dian_4_1_basic = contour_points4_basic[np.argmin(contour_points4_basic[:, 0])]
            contour_points5_basic = contour_points2_basic[(contour_points2_basic[:, 1] < dian_4_1_basic[1])]
            a_column_lower_point = find_target_point2(contour_points5_basic)
        else:
            contour_points5_basic = contour_points2_basic[(contour_points2_basic[:, 1] < dian_2_basic[1])]
            a_column_lower_point = find_target_point2(contour_points5_basic)

    # 4. 查找A柱上端点
    if max_point is not None:
        plane_z1_optimized = max_point[2] + car_height * 0.23
        contour_points_optimized, _ = get_contour_from_stl(mesh, plane_z1_optimized, 'xy')
        contour_points_optimized = np.array(contour_points_optimized)

        x_min_optimized = np.min(contour_points_optimized[:, 0])
        y_min_optimized = np.min(contour_points_optimized[:, 1])
        y_max_optimized = np.max(contour_points_optimized[:, 1])
        contour_points1_optimized = contour_points_optimized[
            (contour_points_optimized[:, 1] < ((y_min_optimized + y_max_optimized) / 2 - 300)) &
            (contour_points_optimized[:, 0] < (x_min_optimized + 600))
            ]
        dian_1_optimized = find_target_point1(contour_points1_optimized)

        if dian_1_optimized is not None:
            contour_points2_optimized = contour_points1_optimized[
                (contour_points1_optimized[:, 1] <= dian_1_optimized[1])]
            dian_2_optimized = contour_points2_optimized[np.argmin(contour_points2_optimized[:, 0])]

            contour_points3_optimized = contour_points2_optimized[
                (contour_points2_optimized[:, 1] <= dian_2_optimized[1]) &
                (contour_points2_optimized[:, 1] >= (dian_2_optimized[1] - 25)) &
                (contour_points2_optimized[:, 0] <= (dian_2_optimized[0] + 30))
                ]
            dian_3_optimized = find_target_point1(contour_points3_optimized)

            if dian_3_optimized is not None:
                contour_points4_optimized = contour_points2_optimized[
                    (contour_points2_optimized[:, 1] < dian_3_optimized[1])]
                dian_4_1_optimized = contour_points4_optimized[np.argmin(contour_points4_optimized[:, 0])]
                contour_points5_optimized = contour_points2_optimized[
                    (contour_points2_optimized[:, 1] < dian_4_1_optimized[1])]
                a_column_upper_point = find_target_point2(contour_points5_optimized)
            else:
                contour_points5_optimized = contour_points2_optimized[
                    (contour_points2_optimized[:, 1] < dian_2_optimized[1])]
                a_column_upper_point = find_target_point2(contour_points5_optimized)

    # 5. 计算并输出A柱与Y轴的夹角 + 可视化
    if a_column_lower_point is not None and a_column_upper_point is not None:
        angle_results = calculate_a_pillar_angles(a_column_lower_point, a_column_upper_point)
        if angle_results is not None:
            # 仅输出核心的Y轴夹角结果
            # print("\n========================")
            # print(f"A柱与Y轴的空间夹角: {angle_results['direction_angles'][1]}°")
            # print("========================")

            # 调用可视化函数
            # visualize_a_pillar_3d(
            #     a_column_lower_point,
            #     a_column_upper_point,
            #     contour_points1_basic,
            #     contour_points1_optimized
            # )
            kongjiabnjiao_Y = angle_results['direction_angles'][1]
        else:
            # print("无法计算夹角：A柱端点未找到")
            kongjiabnjiao_Y = None
    else:
        # 补充A柱端点未找到的情况处理
        kongjiabnjiao_Y = None

    return kongjiabnjiao_Y


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)