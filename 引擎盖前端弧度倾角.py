import numpy as np
import trimesh
import matplotlib.pyplot as plt
import math

# ===================== 全局设置 =====================
# 设置 Matplotlib 显示中文
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# ===================== 工具函数 =====================
def get_contour_from_stl(mesh, plane_z, plane='xz'):
    """计算 STL 与指定平面的交线点云"""
    if plane == 'xy':
        normal = [0, 0, 1]
        origin = [0, 0, plane_z]
    elif plane == 'xz':
        normal = [0, 1, 0]
        origin = [0, plane_z, 0]  # 切 Y 轴
    elif plane == 'yz':
        normal = [1, 0, 0]
        origin = [plane_z, 0, 0]
    else:
        raise ValueError("Invalid plane")

    lines = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    if len(lines) == 0:
        return np.array([])
    points = lines.reshape(-1, 3)
    return points


def find_expanded_vertical_segment(points, x_front_global, z_bottom_limit, window_size=60, x_diff_limit=15.0):
    """
    升级版算法V2: 深度锁定法
    先找到最直的核心区域, 然后只允许 X 坐标偏差在 x_diff_limit (mm) 内的点加入。
    """
    # 1. ROI 初步过滤
    mask = (
            (points[:, 0] < x_front_global + 1500) &  # 稍微放宽范围防止漏点
            (points[:, 0] > x_front_global - 100) &
            (points[:, 2] > z_bottom_limit + 400)
    )
    roi_points = points[mask]

    if len(roi_points) < window_size:
        return roi_points if len(roi_points) > 0 else points

    # 2. 按高度 (Z轴) 排序
    roi_points = roi_points[roi_points[:, 2].argsort()]

    # 3. 寻找最直的核心窗口 (方差最小)
    min_variance = float('inf')
    best_window_idx = -1

    num_windows = len(roi_points) - window_size
    for i in range(num_windows):
        segment = roi_points[i: i + window_size]
        v = np.var(segment[:, 0])
        if v < min_variance:
            min_variance = v
            best_window_idx = i

    if best_window_idx == -1:
        return roi_points  # 没找到, 兜底

    # 获取核心区域的平均 X 坐标 (作为基准深度)
    core_segment = roi_points[best_window_idx: best_window_idx + window_size]
    ref_x = np.mean(core_segment[:, 0])

    # 4. 向上下扩展 (深度一致性检查)
    start_idx = best_window_idx
    end_idx = best_window_idx + window_size

    # (A) 向下搜索 (索引减小)
    final_start = start_idx
    for i in range(start_idx - 1, -1, -1):
        curr_x = roi_points[i, 0]
        if abs(curr_x - ref_x) > x_diff_limit:
            break
        final_start = i

    # (B) 向上搜索 (索引增加)
    final_end = end_idx
    for i in range(end_idx, len(roi_points)):
        curr_x = roi_points[i, 0]
        if abs(curr_x - ref_x) > x_diff_limit:
            break
        final_end = i + 1

    # 5. 提取最终区域
    expanded_segment = roi_points[final_start: final_end]

    return expanded_segment


def fit_inverse_xz_line(points_3d):
    """
    【修改版】反向拟合函数
    拟合方程: x = m * z + c
    解决垂直线斜率无穷大的问题
    """
    # 提取坐标
    x_coords = points_3d[:, 0]
    z_coords = points_3d[:, 2]

    # 记录 Z 坐标范围, 用于绘图
    z_range = (np.min(z_coords), np.max(z_coords))

    if len(x_coords) < 2:
        raise ValueError("至少需要2个点才能拟合直线")

    # 使用 Z 作为自变量, X 作为因变量进行拟合
    m, c = np.polyfit(z_coords, x_coords, deg=1)

    # 计算角度:
    # m = dx / dz = tan(theta_z)
    # theta_z 就是直线与 Z 轴 (竖直方向) 的夹角
    angle_rad = math.atan(m)
    angle_deg = math.degrees(angle_rad)

    # 取绝对值, 通常我们要的是 0~90度
    angle_with_z_deg = abs(angle_deg)

    return (m, c), angle_with_z_deg, z_range


def fit_hood_angle(hood_points):
    """
    拟合引擎盖前端弧度的倾角（水平方向为基准）
    拟合方程: z = k * x + b
    返回: 倾角(°), 斜率k, 截距b
    """
    x_coords = hood_points[:, 0]
    z_coords = hood_points[:, 2]

    if len(x_coords) < 2:
        raise ValueError("引擎盖拟合需要至少2个点")

    # 线性拟合 z = k*x + b
    k, b = np.polyfit(x_coords, z_coords, deg=1)

    # 计算与水平方向的夹角 (arctan(斜率))
    angle_rad = math.atan(k)
    angle_deg = math.degrees(angle_rad)

    # 调整角度范围到 0~180°
    if angle_deg < 0:
        angle_deg += 180

    return angle_deg, k, b


# ===================== 主程序：格栅与引擎盖角度计算 =====================
def calculate(stl_path):
    """
    核心计算函数：
    1. 提取前格栅与Z轴的夹角
    2. 找到引擎盖前端点并拟合400mm范围内的倾角
    参数: stl_path - STL文件路径
    返回: 引擎盖倾角（主返回值，对齐参考示例单值返回风格）
    """
    # 初始化返回值（对齐参考示例）
    hood_angle = None

    # --------------- 核心参数配置 ---------------
    offset_mm = 300
    search_window_size = 60  # 基础拟合窗口大小(点数)
    x_diff_limit = 3.0
    grill_offset_x = 50  # 格栅拟合直线向后偏移距离
    hood_fit_range = 400  # 引擎盖拟合范围（前端点向后400mm）
    # -------------------------------------------

    # 1. 加载网格文件（对齐参考示例步骤编号）
    mesh_data = trimesh.load_mesh(stl_path)
    if isinstance(mesh_data, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh_data.dump())
    else:
        mesh = mesh_data

    # 2. 获取几何信息
    vertices = mesh.vertices
    x_min = np.min(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    y_mid = (y_max + y_min) / 2

    # 3. 切片获取XZ平面轮廓
    target_y = y_mid - offset_mm
    contour_points = get_contour_from_stl(mesh, target_y, plane='xz')

    if len(contour_points) == 0:
        # print(f"错误: 在 Y={target_y} 处切片未获取到点。")
        return hood_angle

    # 4. 自动寻找最佳并扩展垂直段（前格栅区域）
    target_points = find_expanded_vertical_segment(
        contour_points,
        x_front_global=x_min,
        z_bottom_limit=z_min,
        window_size=search_window_size,
        x_diff_limit=x_diff_limit
    )

    if target_points is None or len(target_points) < 2:
        # print("未找到合适的格栅拟合区域。")
        return hood_angle

    # 5. 反向拟合前格栅直线
    (m, c), grill_angle, z_range = fit_inverse_xz_line(target_points)
    # print(f"前格栅与Z轴夹角: {grill_angle:.2f}°")
    # print(f"前格栅拟合直线方程: x = {m:.4f} * z + {c:.4f}")

    # 6. 格栅直线向后偏移50mm（X轴正方向）
    c_offset = c + grill_offset_x  # 偏移后的截距
    # print(f"偏移50mm后直线方程: x = {m:.4f} * z + {c_offset:.4f}")

    # 7. 找到偏移后直线上Z值最大的点（引擎盖前端点）
    # 遍历切片点，计算每个点到偏移直线的距离，筛选距离足够近的点
    distance_threshold = 10.0  # 距离直线10mm内的点视为在直线上
    hood_front_candidates = []

    for point in contour_points:
        x_p, z_p = point[0], point[2]
        # 计算点到直线 x - m*z - c_offset = 0 的垂直距离
        distance = abs(x_p - m * z_p - c_offset) / math.sqrt(m ** 2 + 1)
        if distance < distance_threshold:
            hood_front_candidates.append(point)

    if len(hood_front_candidates) == 0:
        # print("未找到引擎盖前端点候选")
        return hood_angle

    hood_front_candidates = np.array(hood_front_candidates)
    # 找到Z值最大的点
    max_z_idx = np.argmax(hood_front_candidates[:, 2])
    hood_front_point = hood_front_candidates[max_z_idx]
    hood_front_x, hood_front_y, hood_front_z = hood_front_point
    # print(f"引擎盖前端点坐标: X={hood_front_x:.2f}, Y={hood_front_y:.2f}, Z={hood_front_z:.2f}")

    # 8. 筛选引擎盖拟合范围（前端点向后400mm）
    hood_fit_start_x = hood_front_x
    hood_fit_end_x = hood_front_x + hood_fit_range  # 向后（X增大）400mm
    # print(f"引擎盖拟合范围: X ∈ [{hood_fit_start_x:.2f}, {hood_fit_end_x:.2f}]")

    # 9. 筛选该范围内的引擎盖点（过滤底部无效点）
    hood_mask = (
            (contour_points[:, 0] >= hood_fit_start_x) &
            (contour_points[:, 0] <= hood_fit_end_x) &
            (contour_points[:, 2] > z_min + 500)  # 过滤底部点
    )
    hood_fit_points = contour_points[hood_mask]

    if len(hood_fit_points) < 2:
        # print(f"引擎盖拟合范围内点数不足: {len(hood_fit_points)}")
        return hood_angle

    # 10. 拟合引擎盖倾角（核心返回值）
    hood_angle, hood_k, hood_b = fit_hood_angle(hood_fit_points)
    # print(f"引擎盖前端倾角（与水平方向）: {hood_angle:.2f}°")
    # print(f"引擎盖拟合直线方程: z = {hood_k:.4f} * x + {hood_b:.2f}")

    #11. 可视化
    # plt.figure(figsize=(14, 10))
    #
    # # 绘制切片所有点（灰色）
    # plt.scatter(contour_points[:, 0], contour_points[:, 2], s=2, c='lightgray', label='切片轮廓点')
    #
    # # 绘制前格栅拟合点（红色）
    # plt.scatter(target_points[:, 0], target_points[:, 2], s=20, c='red', label='前格栅拟合点')
    #
    # # 绘制原始格栅拟合直线（蓝色）
    # z_grill = np.linspace(z_range[0]-100, z_range[1]+100, 100)
    # x_grill = m * z_grill + c
    # plt.plot(x_grill, z_grill, 'b-', linewidth=2, label=f'前格栅拟合直线 (与Z轴{grill_angle:.2f}°)')
    #
    # # 绘制偏移后格栅直线（绿色虚线）
    # x_grill_offset = m * z_grill + c_offset
    # plt.plot(x_grill_offset, z_grill, 'g--', linewidth=2, label=f'格栅直线后移50mm')
    #
    # # 标记引擎盖前端点（黄色五角星）
    # plt.scatter(hood_front_x, hood_front_z, s=200, c='gold', marker='*', zorder=10, label='引擎盖前端点')
    #
    # # 绘制引擎盖拟合点（橙色）
    # plt.scatter(hood_fit_points[:, 0], hood_fit_points[:, 2], s=20, c='orange', label='引擎盖拟合点')
    #
    # # 绘制引擎盖拟合直线（紫色）
    # x_hood = np.linspace(hood_fit_start_x, hood_fit_end_x, 100)
    # z_hood = hood_k * x_hood + hood_b
    # plt.plot(x_hood, z_hood, 'purple', linewidth=2, label=f'引擎盖拟合直线 (倾角{hood_angle:.2f}°)')
    #
    # # 标记引擎盖拟合范围（垂直虚线）
    # plt.axvline(x=hood_fit_start_x, color='orange', linestyle=':', alpha=0.8, label='引擎盖拟合起点')
    # plt.axvline(x=hood_fit_end_x, color='orange', linestyle=':', alpha=0.8, label='引擎盖拟合终点')
    #
    # plt.title(f"前格栅与引擎盖角度分析")
    # plt.xlabel("X (车长方向, mm)")
    # plt.ylabel("Z (高度方向, mm)")
    # plt.axis('equal')
    # plt.grid(visible=True, linestyle='--', alpha=0.6)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

    # 最终仅返回引擎盖倾角（对齐参考示例单值返回风格）
    return hood_angle


# ===================== 运行程序 =====================
if __name__ == "__main__":
    rear_windshield_angle = calculate('F:/风噪/小米.stl')
    print(rear_windshield_angle)