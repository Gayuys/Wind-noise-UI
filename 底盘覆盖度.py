import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def project_3d_to_2d(points, plane):
    if plane == 'xy':
        return np.array([point[:2] for point in points])
    elif plane == 'xz':
        return np.array([point[[0, 2]] for point in points])
    elif plane == 'yz':
        return np.array([point[1:] for point in points])
    else:
        raise ValueError("无效的平面参数")


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
        raise ValueError("无效的平面参数")

    intersections = trimesh.intersections.mesh_plane(
        mesh=mesh, plane_normal=plane_normal, plane_origin=plane_origin
    )
    points_crossing_plane = []
    for line in intersections:
        points_crossing_plane.append(line[0])
        points_crossing_plane.append(line[1])
    return points_crossing_plane, intersections


def calculate_area_ratio_monte_carlo(all_points, chassis_points, num_samples=500000, search_radius=50):
    """
    使用蒙特卡洛法计算面积比值
    :param all_points: 整车投影点集 (N, 2)
    :param chassis_points: 底盘投影点集 (M, 2)
    :param num_samples: 随机撒点数量
    :param search_radius: 判定随机点是否在图形内的半径阈值 (mm)
    """
    # print("\n--- 开始蒙特卡洛面积计算 ---")

    # 1. 确定包围盒 (Bounding Box)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])

    # 稍微扩大一点范围，防止边缘判定丢失
    margin = 50
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    box_area = (x_max - x_min) * (y_max - y_min)
    # print(f"采样区域大小: {box_area / 1e6:.2f} m^2")

    # 2. 生成随机采样点
    # 在包围盒内均匀分布
    rand_x = np.random.uniform(x_min, x_max, num_samples)
    rand_y = np.random.uniform(y_min, y_max, num_samples)
    random_points = np.column_stack((rand_x, rand_y))

    # 3. 构建 KDTree 用于快速查找
    # 这一步是关键：我们需要快速判断随机点离真实点有多远
    # print("构建空间索引 (KDTree)...")
    tree_all = cKDTree(all_points)
    tree_chassis = cKDTree(chassis_points)

    # 4. 判定归属 (查询最近邻距离)
    # query 返回 (distances, indices)
    # print(f"正在判定 {num_samples} 个采样点归属 (搜索半径={search_radius}mm)...")

    # 判定是否在整车范围内
    dists_all, _ = tree_all.query(random_points, k=1)
    mask_in_all = dists_all <= search_radius
    count_all = np.sum(mask_in_all)

    # 判定是否在底盘范围内
    dists_chassis, _ = tree_chassis.query(random_points, k=1)
    mask_in_chassis = dists_chassis <= search_radius
    count_chassis = np.sum(mask_in_chassis)

    # 5. 计算面积与比值
    # 面积 = (命中点数 / 总撒点数) * 包围盒面积
    area_all = (count_all / num_samples) * box_area
    area_chassis = (count_chassis / num_samples) * box_area

    ratio = 0
    if area_all > 0:
        ratio = area_chassis / area_all

    # print(f"结果统计:")
    # print(f"  - 命中整车点数: {count_all}")
    # print(f"  - 命中底盘点数: {count_chassis}")
    # print(f"  - 整车投影估算面积: {area_all / 1e6:.4f} m^2")
    # print(f"  - 底盘投影估算面积: {area_chassis / 1e6:.4f} m^2")
    # print(f"  - 覆盖度 (Area Ratio): {ratio:.2%} (即 {ratio:.4f})")

    return ratio, random_points[mask_in_all], random_points[mask_in_chassis]


def plot_comparison_and_calc(all_points_2d, chassis_points_2d):
    """
    绘制对比图并显示计算结果
    """
    current_dir = r"F:\Users\liph\Desktop\自动提取新增\自动0130版本"
    if not os.path.exists(current_dir): os.makedirs(current_dir)
    output_path = os.path.join(current_dir, "comparison_monte_carlo.jpg")

    # --- 调用蒙特卡洛计算 ---
    # search_radius 越小越精确，但如果点云稀疏，会有空洞。
    # 建议设为 降采样后点间距的 2-3 倍。
    ratio, sample_pts_all, sample_pts_chassis = calculate_area_ratio_monte_carlo(
        all_points_2d, chassis_points_2d, num_samples=2000000, search_radius=30
    )
    return ratio
    # # --- 绘图 ---
    # x_min, x_max = np.min(all_points_2d[:, 0]), np.max(all_points_2d[:, 0])
    # y_min, y_max = np.min(all_points_2d[:, 1]), np.max(all_points_2d[:, 1])
    # margin = 100
    # xlims = (x_min - margin, x_max + margin)
    # ylims = (y_min - margin, y_max + margin)

    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # # 图1：整车 + 蒙特卡洛命中的点（绿色）
    # axes[0].set_title(f"图1: 整车投影 (估算面积: {len(sample_pts_all)} 采样点)")
    # axes[0].scatter(all_points_2d[:, 0], all_points_2d[:, 1], s=1, c='gray', alpha=0.3)
    # # 可视化蒙特卡洛命中的区域（这一步可以直观看到计算了哪些面积）
    # # axes[0].scatter(sample_pts_all[:, 0], sample_pts_all[:, 1], s=0.5, c='green', alpha=0.1)

    # axes[0].set_aspect('equal', adjustable='box')
    # axes[0].set_xlim(xlims)
    # axes[0].set_ylim(ylims)
    # axes[0].axis('off')

    # # 图2：底盘 + 蒙特卡洛命中的点（红色）
    # axes[1].set_title(f"图2: 底盘投影 (覆盖度: {ratio:.2%})")
    # axes[1].scatter(all_points_2d[:, 0], all_points_2d[:, 1], s=1, c='lightgray', alpha=0.1)  # 背景参考
    # axes[1].scatter(chassis_points_2d[:, 0], chassis_points_2d[:, 1], s=5, c='blue', label='Chassis Data')
    # # 可视化蒙特卡洛命中的区域
    # # axes[1].scatter(sample_pts_chassis[:, 0], sample_pts_chassis[:, 1], s=0.5, c='red', alpha=0.1, label='MC Hit')

    # # 在图上添加文字说明
    # plt.figtext(0.5, 0.05, f"覆盖面积比值 = {ratio:.4f}", ha="center", fontsize=14,
    #             bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # axes[1].set_aspect('equal', adjustable='box')
    # axes[1].set_xlim(xlims)
    # axes[1].set_ylim(ylims)
    # axes[1].axis('off')

    # plt.tight_layout()
    # plt.savefig(output_path, dpi=300)
    # print(f"图像已保存至: {output_path}")
    # plt.show()


# --- 主程序 ---
def calculate(stl_path):
    ratio = None

    if not os.path.exists(stl_path):
        print(f"错误：找不到文件 {stl_path}")
        exit()

    # print("正在处理...")
    mesh_or_scene = trimesh.load_mesh(stl_path)

    # 降采样
    original_num_points = len(mesh_or_scene.vertices)
    sample_count = original_num_points // 2
    indices = np.random.choice(original_num_points, sample_count, replace=False)
    points = mesh_or_scene.vertices[indices]

    # 初始交线
    plane_z = min(points[:, 2]) + 500
    segments, _ = get_contour_from_stl(mesh_or_scene, plane_z, 'xy')
    if len(segments) == 0: exit()
    segments = np.array(segments)

    # 投影并计算中心
    xy_projection = project_3d_to_2d(segments, 'xy')
    middle_point_y = (np.max(xy_projection[:, 1]) + np.min(xy_projection[:, 1])) / 2
    middle_point_x = (np.max(xy_projection[:, 0]) + np.min(xy_projection[:, 0])) / 2

    # 计算轮心
    plane_z2 = min(points[:, 2]) + 100
    segments2, _ = get_contour_from_stl(mesh_or_scene, plane_z2, 'xy')
    segments2 = np.array(segments2)
    mask_zuoqian = (segments2[:, 0] <= middle_point_x) & (segments2[:, 1] <= middle_point_y)
    point_zuoqian = segments2[mask_zuoqian]

    if len(point_zuoqian) > 0:
        point_qianlunzhongxing = (np.max(point_zuoqian[:, 0]) + np.min(point_zuoqian[:, 0])) / 2
    else:
        point_qianlunzhongxing = 0

    # 寻找底盘高度
    plane_znew = min(points[:, 2]) + 100
    plane_dipan = plane_znew

    for i in range(100):
        segments_iter, _ = get_contour_from_stl(mesh_or_scene, plane_znew, 'xy')
        segments_iter = np.array(segments_iter)
        if len(segments_iter) == 0:
            plane_znew += 10
            continue

        mask_iter = (segments_iter[:, 0] <= middle_point_x) & (segments_iter[:, 1] <= middle_point_y)
        point_zuoqiannew = segments_iter[mask_iter]

        if len(point_zuoqiannew) > 0:
            point_qianlunzhongxingnew = (np.max(point_zuoqiannew[:, 0]) + np.min(point_zuoqiannew[:, 0])) / 2
            if abs(point_qianlunzhongxingnew - point_qianlunzhongxing) > 100:
                plane_dipan = plane_znew
                break
        plane_znew += 10

    # 最终筛选
    plane_cheke = plane_dipan + 60
    points_chassis = points[(points[:, 2] <= plane_cheke)]
    xy_projection_chassis = project_3d_to_2d(points_chassis, 'xy')
    xy_projection_all = project_3d_to_2d(points, 'xy')

    # 计算与绘图
    if len(xy_projection_chassis) > 0:
        ratio = plot_comparison_and_calc(xy_projection_all, xy_projection_chassis)
    # print(f"底盘覆盖度: {ratio:.2%}")
    #ratio= f"{ratio * 100:.2f}%"
    return ratio * 100

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\Users\liph\Desktop\自动提取新增\自动0130版本\CAR9.stL")
    print(rear_windshield_angle)
