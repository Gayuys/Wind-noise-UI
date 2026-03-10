import trimesh
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.path import Path

# ===================== 全局设置 =====================
# 设置 Matplotlib 显示中文
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# ===================== 工具函数 =====================
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
    """计算STL模型与平面的交线轮廓"""
    # 定义平面法向量和原点
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

    # 计算交线
    intersections = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin
    )
    points_crossing_plane = []
    for intersection in intersections:
        points_crossing_plane.append(np.array([intersection[0][0], intersection[0][1], intersection[0][2]]))
        points_crossing_plane.append(np.array([intersection[1][0], intersection[1][1], intersection[1][2]]))

    return points_crossing_plane, intersections


def line_plane_intersection(intersections, plane):
    """计算点云交线和平面的交点"""
    plane_normal = np.array(plane[:3])
    plane_d = plane[3]
    # 确定平面上的一个点
    if abs(plane_normal[2]) > 1e-6:
        plane_origin = np.array([0, 0, -plane_d / plane_normal[2]])
    elif abs(plane_normal[1]) > 1e-6:
        plane_origin = np.array([0, -plane_d / plane_normal[1], 0])
    else:
        plane_origin = np.array([-plane_d / plane_normal[0], 0, 0])

    # 线段与平面交点计算函数
    def segment_intersection(line_start, line_end):
        line_dir = line_end - line_start
        denom = np.dot(line_dir, plane_normal)
        if abs(denom) < 1e-6:  # 平行无交点
            return None
        t = np.dot(plane_normal, plane_origin - line_start) / denom
        if 0 <= t <= 1:  # 交点在线段上
            intersection_point = line_start + t * line_dir
            if abs(np.dot(plane_normal, intersection_point) + plane_d) < 1e-6:
                return intersection_point
        return None

    # 计算所有交点
    intersection_points = []
    for line_segment in intersections:
        if len(line_segment) < 2:
            continue
        line_start = line_segment[0]
        line_end = line_segment[1]
        intersection = segment_intersection(line_start, line_end)
        if intersection is not None:
            intersection_points.append(intersection)

    return np.array(intersection_points) if intersection_points else np.array([])


def linear_fit_xy(points_3d):
    """对三维点集进行XZ平面的线性拟合"""
    points_3d = np.array(points_3d)
    if len(points_3d) < 2:
        raise ValueError("线性拟合需要至少2个点")
    x = points_3d[:, 0]
    z = points_3d[:, 2]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, z, rcond=None)[0]
    y_fit = slope * x + intercept
    return (slope, intercept), y_fit


def curve_fit_xy(points_3d, degree=3):
    """对三维点集进行XZ平面的多项式曲线拟合"""
    points_3d = np.array(points_3d)
    if len(points_3d) == 0:
        raise ValueError("曲线拟合输入点集为空")
    x = points_3d[:, 0]
    z = points_3d[:, 2]
    if len(x) < degree + 1:
        raise ValueError(f"{degree}次拟合需要至少{degree + 1}个点")
    coefficients = np.polyfit(x, z, degree)
    poly_func = np.poly1d(coefficients)
    z_fit = poly_func(x)
    return coefficients, poly_func, x, z, z_fit


def calculate_angle_statistics(points_3d, degree=3):
    """计算拟合曲线与水平方向夹角的统计值"""
    coefficients, poly_func, x, z_original, z_fit = curve_fit_xy(points_3d, degree)
    if poly_func is None or x is None:
        raise ValueError("曲线拟合失败，无法计算角度")
    x_dense = np.linspace(min(x), max(x), 1000)
    deriv_func = poly_func.deriv()
    slopes = deriv_func(x_dense)
    angles = np.arctan(slopes) * 180 / np.pi
    return {
        'max_angle': np.max(angles),
        'min_angle': np.min(angles),
        'mean_angle': np.mean(angles),
        'angles': angles,
        'x_dense': x_dense,
        'x_original': x,
        'z_original': z_original,
        'z_fit': z_fit,
        'poly_func': poly_func
    }, coefficients


# ===================== 主程序：引擎盖角度计算 =====================
def calculate(stl_path):
    """
    计算引擎盖后端点及指定范围内的线性拟合角度（对齐参考示例风格）

    参数:
        stl_path (str): STL文件路径

    返回:
        float: 引擎盖线性拟合与水平方向的夹角（°），失败返回None
    """
    hood_angle = None  # 最终返回的引擎盖角度

    # 1. 加载网格文件
    mesh = trimesh.load_mesh(stl_path)
    points = mesh.vertices
    middle_point_y = (max(points[:, 1]) + min(points[:, 1])) / 2
    plane_z = middle_point_y  # 简化重复计算

    # 2. 获取交线轮廓
    segments, intersections = get_contour_from_stl(mesh, plane_z, 'xz')
    segments = np.unique(segments, axis=0) if segments else np.array([])

    # 3. 投影到XZ平面并筛选点集
    xz_projection = project_3d_to_2d(segments, 'xz')
    xz_projection = np.unique(xz_projection, axis=0) if len(xz_projection) > 0 else np.array([])
    if len(xz_projection) == 0:
        return hood_angle

    point_front = min(xz_projection[:, 0])
    point_back = max(xz_projection[:, 0])
    middle = (point_front + point_back) / 2
    point_settle = segments[
        (segments[:, 0] >= point_front + 500) &
        (segments[:, 0] <= middle)
        ] if len(segments) > 0 else np.array([])
    if len(point_settle) == 0:
        return hood_angle

    # 4. 找出引擎盖后端点（核心逻辑）
    Step1 = 10
    settle_max = 0
    start = min(point_settle[:, 0])
    result3 = np.array([])

    for i in range(0, 5000):
        plane1 = (1, 0, 0, (-1 * (start + i * Step1)))
        result1 = line_plane_intersection(intersections, plane1)
        if len(result1) == 0:
            continue

        settle_max1 = max(result1[:, 2])
        if settle_max1 >= settle_max:
            settle_max = settle_max1
        else:
            # 精细搜索
            Step2 = 0.5
            settle_max2 = 0
            for a in range(0, 20):
                start1 = start + (i - 1) * Step1
                plane2 = (1, 0, 0, (-1 * (start1 + a * Step2)))
                result2 = line_plane_intersection(intersections, plane2)
                if len(result2) == 0:
                    continue

                settle_max3 = max(result2[:, 2])
                if settle_max3 >= settle_max2:
                    settle_max2 = settle_max3
                else:
                    plane3 = (1, 0, 0, (-1 * (start1 + (a - 1) * Step2)))
                    result3 = line_plane_intersection(intersections, plane3)
                    break
            break

    if len(result3) == 0:
        return hood_angle

    # 提取引擎盖后端点
    max_z_index = np.argmax(result3[:, 2])
    hood_back_point = result3[max_z_index]
    if hood_back_point is None:
        return hood_angle

    # 5. 筛选顶点并计算中间值
    z_min = np.min(points[:, 2])
    filtered_vertices = points[points[:, 2] > (z_min + 300)]
    if len(filtered_vertices) == 0:
        return hood_angle

    y_min = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    y_max = filtered_vertices[np.argmax(filtered_vertices[:, 1])]
    y_mid = (y_max[1] + y_min[1]) / 2

    # 6. 获取XZ平面切片轮廓（用于拟合引擎盖角度）
    plane_z = y_mid - 20
    contour_points, _ = get_contour_from_stl(mesh, plane_z, 'xz')
    contour_points = np.array(contour_points) if contour_points else np.array([])
    if len(contour_points) == 0:
        return hood_angle

    # 7. 以引擎盖后端点为基准，向前偏移400mm确定拟合范围
    fit_start_x = hood_back_point[0] - 400
    fit_end_x = hood_back_point[0]

    # 8. 筛选该范围内的引擎盖点集
    hood_fit_points = contour_points[
        (contour_points[:, 0] >= fit_start_x) &
        (contour_points[:, 0] <= fit_end_x) &
        (contour_points[:, 2] > (z_min + 300))
        ]
    if len(hood_fit_points) == 0:
        return hood_angle

    # 9. 线性拟合计算引擎盖角度（核心返回值）
    (slope, intercept), y_fit = linear_fit_xy(hood_fit_points)
    if slope is None:
        return hood_angle

    # 计算与水平方向的夹角
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    hood_angle = angle_deg

    # 10. 可选：打印详细结果（对齐参考示例的注释风格）
    # print(f"引擎盖后端点坐标: X={hood_back_point[0]:.2f}, Y={hood_back_point[1]:.2f}, Z={hood_back_point[2]:.2f}")
    # print(f"引擎盖拟合范围: X ∈ [{fit_start_x:.2f}, {fit_end_x:.2f}] (后端点向前400mm)")
    # print(f"拟合范围内有效点数: {len(hood_fit_points)}")
    #
    # # 多项式拟合统计（可选输出）
    # angle_stats, coefficients = calculate_angle_statistics(hood_fit_points, degree=4)
    # if angle_stats is not None:
    #     print("\n=== 引擎盖角度统计（4次多项式拟合）===")
    #     print(f"与水平方向夹角最大值: {angle_stats['max_angle']:.2f}°")
    #     print(f"与水平方向夹角最小值: {angle_stats['min_angle']:.2f}°")
    #     print(f"与水平方向夹角平均值: {angle_stats['mean_angle']:.2f}°")
    #
    # print("\n=== 引擎盖线性拟合结果 ===")
    # print(f"拟合直线方程: Z = {slope:.4f} * X + {intercept:.2f}")
    # print(f"与水平方向的夹角: {hood_angle:.2f}°")

    # 11. 可视化（对齐参考示例的注释风格）
    # point_size = 2
    # plt.figure(figsize=(12, 8))
    #
    # # 绘制所有切片轮廓点（灰色）
    # plt.scatter(contour_points[:, 0], contour_points[:, 2], color='lightgray', s=1, alpha=0.5, label='切片所有点')
    #
    # # 绘制引擎盖拟合范围内的点（红色）
    # plt.scatter(hood_fit_points[:, 0], hood_fit_points[:, 2], color='red', s=point_size * 2,
    #             label='拟合范围点（后端点向前400mm）')

    # # 绘制多项式拟合曲线（蓝色）
    # if angle_stats is not None:
    #     plt.plot(angle_stats['x_dense'], angle_stats['poly_func'](angle_stats['x_dense']),
    #              'b-', linewidth=2, label=f'4次多项式拟合曲线')
    #
    # # 绘制线性拟合直线（绿色虚线）
    # x_range = np.array([fit_start_x, fit_end_x])
    # y_range = slope * x_range + intercept
    # plt.plot(x_range, y_range, 'g--', linewidth=2,
    #          label=f'线性拟合直线 (夹角{hood_angle:.2f}°)')

    # 标记引擎盖后端点（黄色五角星）
    # # plt.scatter(hood_back_point[0], hood_back_point[2],
    #             color='gold', marker='*', s=200, label='引擎盖后端点', zorder=10)

    # 标记拟合范围边界（垂直虚线）
    # plt.axvline(x=fit_start_x, color='orange', linestyle=':', alpha=0.8, label='拟合范围起点（后端点-400mm）')
    # plt.axvline(x=fit_end_x, color='orange', linestyle=':', alpha=0.8, label='拟合范围终点（后端点）')
    #
    # plt.title('引擎盖角度拟合（后端点向前400mm范围）')
    # plt.xlabel('X轴 (mm)')
    # plt.ylabel('Z轴 (mm)')
    # plt.axis('equal')
    # plt.grid(True, alpha=0.6)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

    return hood_angle


# ===================== 运行程序 =====================
if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:/风噪/小米.stl")
    print(rear_windshield_angle)