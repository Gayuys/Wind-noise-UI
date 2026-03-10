import numpy as np
import trimesh
import matplotlib.pyplot as plt
import math

# ===================== 绘图配置区 =====================
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

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
        # print(f"警告: ROI区域点数不足 ({len(roi_points)}), 返回原始 ROI")
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
    # print(f"    [锁定深度] 核心区域 X = {ref_x:.2f}, 允许偏差 ±{x_diff_limit}mm")

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

    # print(f"    [最终结果] 核心 {window_size} 点 -> 扩展为 {len(expanded_segment)} 点")
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

    # ===================== 核心修改 =====================
    # 使用 Z 作为自变量, X 作为因变量进行拟合
    # polyfit(x, y, deg) -> 这里输入 (z, x, 1)
    m, c = np.polyfit(z_coords, x_coords, deg=1)

    # 计算角度:
    # m = dx / dz = tan(theta_z)
    # theta_z 就是直线与 Z 轴 (竖直方向) 的夹角
    angle_rad = math.atan(m)
    angle_deg = math.degrees(angle_rad)

    # 取绝对值, 通常我们要的是 0~90度
    angle_with_z_deg = abs(angle_deg)
    # ====================================================

    return (m, c), angle_with_z_deg, z_range

# ===================== 核心计算函数 =====================
def calculate(stl_path):
    """
    核心计算函数：提取STL模型中前格栅与Z轴的夹角
    参数: stl_path - STL文件路径
    返回: angle_with_z_deg - 格栅与Z轴的夹角（角度制，0~90°）
    """
    # --------------- 核心参数配置 ---------------
    offset_mm = 300
    search_window_size = 60  # 基础拟合窗口大小(点数)
    x_diff_limit = 3.0
    # -------------------------------------------

    # print(f"正在加载模型: {stl_path}")

    # 模型加载逻辑
    mesh_data = trimesh.load(stl_path)
    if isinstance(mesh_data, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh_data.dump())
    else:
        mesh = mesh_data

    # 1. 获取几何信息
    vertices = mesh.vertices
    x_min = np.min(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min = np.min(vertices[:, 2])

    y_mid = (y_max + y_min) / 2

    # 2. 切片
    target_y = y_mid - offset_mm
    # print(f"正在进行切片... 中心Y={y_mid:.2f}, 目标Y={target_y:.2f} (偏移 {offset_mm}mm)")

    contour_points = get_contour_from_stl(mesh, target_y, plane='xz')

    if len(contour_points) == 0:
        # print(f"错误: 在 Y={target_y} 处切片未获取到点。")
        return None

    # 3. 自动寻找最佳并扩展垂直段
    # print("正在自动搜索垂直格栅区域...")

    target_points = find_expanded_vertical_segment(
        contour_points,
        x_front_global=x_min,
        z_bottom_limit=z_min,
        window_size=search_window_size,
        x_diff_limit=x_diff_limit
    )

    if target_points is None or len(target_points) < 2:
        # print("未找到合适的拟合区域。")
        return None

    # ===================== 数据打印 =====================
    # print("\n" + "=" * 40)
    # print(f"【拟合数据详情】 共锁定 {len(target_points)} 个点")
    # print(f"{'Index':<6} | {'X (mm)':<10} | {'Z (mm)':<10}")
    # print("-" * 40)
    # for i, p in enumerate(target_points):
    #     print(f"{i:<6} | {p[0]:<10.4f} | {p[2]:<10.4f}")
    # print("=" * 40 + "\n")

    # 4. 反向拟合与计算
    (m, c), angle_with_z_deg, z_range = fit_inverse_xz_line(target_points)
    # print(f"反向斜率 m (dx/dz) = {m:.6f}")
    #
    # print("-" * 30)
    # print(f"反向拟合结果 (Fit X against Z): ")
    # print(f"直线方程: x = {m:.4f} * z + {c:.4f}")
    # print(f"前格栅角度: {angle_with_z_deg:.2f}°")
    # print("-" * 30)

    # 5. 可视化
    # plt.figure(figsize=(12, 8))

    # (A) 画所有轮廓点 (灰色)
    # x_data = target_points[:, 0]
    # z_data = target_points[:, 2]
    #
    # margin_view = 500
    # mask_bg = (contour_points[:, 0] > np.min(x_data) - margin_view) & \
    #           (contour_points[:, 0] < np.max(x_data) + margin_view)
    # bg_points = contour_points[mask_bg]
    #
    # plt.scatter(bg_points[:, 0], bg_points[:, 2], s=2, c='lightgray', label=f'切片轮廓 (Offset {offset_mm})')
    #
    # # 画一条参考线表示整车最前端
    # plt.axvline(x=x_min, color='orange', linestyle='--', alpha=0.5, label='整车最前端 (X_min)')
    #
    # # (B) 画被选中的拟合区域 (红色)
    # plt.scatter(x_data, z_data, s=20, c='red', zorder=5,
    #             label=f'自动锁定的格栅区域 ({len(target_points)}点)')

    # (C) 画拟合直线 (蓝色)
    # z_min_fit, z_max_fit = z_range
    # z_length = z_max_fit - z_min_fit
    # z_plot = np.linspace(z_min_fit - z_length * 0.5, z_max_fit + z_length * 0.5, num=100)
    # x_plot = m * z_plot + c
    #
    # plt.plot(x_plot, z_plot, color='blue', linewidth=2, zorder=10,
    #          label=f'反向拟合直线 (与Z轴夹角{angle_with_z_deg:.2f}°)')
    #
    # plt.title(f"格栅角度提取 (反向拟合 x=mz+c): {angle_with_z_deg:.2f}°")
    # plt.xlabel("X (车长方向)")
    # plt.ylabel("Z (高度方向)")
    # plt.axis('equal')
    # plt.grid(visible=True, linestyle='--', alpha=0.6)
    # plt.legend()
    # plt.show()

    return angle_with_z_deg

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    rear_windshield_angle = calculate(r"D:\baocun1\红旗车型点云数据\红旗点云文件stl版\红旗点云文件stl版\E009.stl")
    print(rear_windshield_angle)