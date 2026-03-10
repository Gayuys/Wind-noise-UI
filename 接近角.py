import trimesh
import numpy as np
import matplotlib.pyplot as plt
import math

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
        return np.array([[point[0], point[2]] for point in points])
    elif plane == 'yz':
        return np.array([point[1:] for point in points])
    else:
        return None


def get_contour_from_stl(mesh, plane_z, plane):
    """计算STL文件与平面的交线轮廓"""
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
        return None, None

    intersections = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin
    )

    points_crossing_plane = []
    for intersection in intersections:
        points_crossing_plane.append(np.array(intersection[0]))
        points_crossing_plane.append(np.array(intersection[1]))
    return points_crossing_plane, intersections


def calculate_minimum_ground_clearance(mesh):
    """内联最小离地间隙计算逻辑（原外部文件逻辑）"""
    vertices = mesh.vertices
    # 找到全局z轴最小值的点
    z_min_1 = vertices[np.argmin(vertices[:, 2])]
    # 计算y轴最大、最小值并取中值
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])
    y_mid = (y_max + y_min) / 2
    # 筛选y轴中值±500范围内的点
    filtered_vertices = vertices[(vertices[:, 1] < (y_mid + 500)) & (vertices[:, 1] > (y_mid - 500))]
    # 找到筛选后z轴最小值的点
    z_min_2 = filtered_vertices[np.argmin(filtered_vertices[:, 2])]
    # 计算最小离地间隙
    clearance = abs(z_min_1[2] - z_min_2[2])

    # 最小离地间隙可视化（原逻辑保留）
    point_size = 2
    # plt.figure()
    # plt.plot(vertices[:, 0], vertices[:, 2], 'ro', markersize=point_size)
    # plt.plot(z_min_1[0], z_min_1[2], 'go', markersize=10, label='全局Z最小值')
    # plt.plot(z_min_2[0], z_min_2[2], 'bo', markersize=10, label='Y中值区域Z最小值')
    # plt.title('最小离地间隙分析')
    # plt.xlabel('X-axis (mm)')
    # plt.ylabel('Z-axis (mm)')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()

    return clearance


# ===================== 主程序：接近角计算 =====================
def calculate(stl_path):
    angle_x = None
    # 1. 加载网格文件
    mesh = trimesh.load_mesh(stl_path)
    points = mesh.vertices

    # 2. 计算最小离地间隙（内联逻辑，带可视化）
    distance = calculate_minimum_ground_clearance(mesh)

    # 3. 预处理：计算轮心X坐标
    point_qianlunzhongxing = None
    xz_projection = None
    # 3.1 获取XY平面投影计算前轮位置
    plane_z = min(points[:, 2]) + 500
    segments, _ = get_contour_from_stl(mesh, plane_z, plane='xy')
    if segments is not None:
        xy_proj = project_3d_to_2d(segments, plane='xy')
        middle_x = (max(xy_proj[:, 0]) + min(xy_proj[:, 0])) / 2
        middle_y = (max(xy_proj[:, 1]) + min(xy_proj[:, 1])) / 2

        # 截取左前部分计算前轮
        plane_z2 = min(points[:, 2]) + 100
        segments2, _ = get_contour_from_stl(mesh, plane_z2, plane='xy')
        segments2 = np.array(segments2)
        # 筛选左前区域的点
        point_zuoqian = segments2[(segments2[:, 0] <= middle_x) & (segments2[:, 1] <= middle_y)]
        # 简单估算前轮中心X
        point_qianlunzhongxing = (max(point_zuoqian[:, 0]) + min(point_zuoqian[:, 0])) / 2

    # 3.2 全车投影到XZ平面并去重
    xz_projection = project_3d_to_2d(points, plane='xz')
    xz_projection = np.unique(xz_projection, axis=0)

    # 4. 筛选车头点云 (Point_chetou)
    Point_chetou = None
    p1 = point_qianlunzhongxing
    if p1 is not None and xz_projection is not None:
        global_min_z = min(xz_projection[:, 1])
        height_threshold = global_min_z + distance + 300
        mask = (xz_projection[:, 0] <= p1) & (xz_projection[:, 1] <= height_threshold)
        Point_chetou = xz_projection[mask]

    # 5. 计算接近角（逻辑：预处理去噪 -> 迭代逼近）
    point_wheel = None  # 最终, 轮胎接地点
    point_target = None  # 最终, 车身切点
    angle_x = None  # 最终: 接近角
    line_equation = None  # 最终: 直线方程 [m, b]
    valid_points_for_plot = None

    if Point_chetou is not None and len(Point_chetou) > 0:
        # 步骤 A: 地面噪点预处理（升级版：百分位抗噪 + 全局切除）
        x_tolerance = 20.0
        wheel_slice_mask = (xz_projection[:, 0] > (p1 - x_tolerance)) & (xz_projection[:, 0] < (p1 + x_tolerance))
        wheel_slice_points = xz_projection[wheel_slice_mask]
        if len(wheel_slice_points) == 0:
            wheel_slice_points = xz_projection
        ref_ground_z = np.percentile(wheel_slice_points[:, 1], 2)  # 取 2% 分位点

        # 全局"剃刀"过滤
        global_clean_mask = Point_chetou[:, 1] >= (ref_ground_z - 1.0)
        Point_chetou = Point_chetou[global_clean_mask]
        xz_clean_mask = xz_projection[:, 1] >= (ref_ground_z - 1.0)
        xz_projection = xz_projection[xz_clean_mask]

        # 保险杠离地间隙过滤
        bumper_zone_threshold_x = p1 - 350.0
        min_bumper_clearance = 50.0
        keep_mask = (Point_chetou[:, 0] >= bumper_zone_threshold_x) | \
                    ((Point_chetou[:, 0] < bumper_zone_threshold_x) & (
                            Point_chetou[:, 1] > (ref_ground_z + min_bumper_clearance)))
        Point_chetou = Point_chetou[keep_mask]

        if len(Point_chetou) == 0:
            pass

        # 步骤 B: 迭代逼近法
        step_size = 1.0  # 移动步长
        max_iter = 1500  # 最大迭代次数
        tire_radius_exclusion = 350.0  # 保险杠判断区域
        current_contact_x = p1  # 初始接地点 X

        # 循环寻找最佳接地点
        for i in range(max_iter):
            # 1. 确定当前X坐标下的轮胎Z值（找最低点）
            x_tol = 5.0
            slice_mask = (xz_projection[:, 0] > (current_contact_x - x_tol)) & (
                    xz_projection[:, 0] < (current_contact_x + x_tol))
            slice_points = xz_projection[slice_mask]
            if len(slice_points) == 0:
                break
            current_contact_z = np.min(slice_points[:, 1])
            current_wheel_point = np.array([current_contact_x, current_contact_z])

            # 2. 计算切线（基于保险杠核心区域）
            calc_mask = Point_chetou[:, 0] < (p1 - tire_radius_exclusion)
            calc_points = Point_chetou[calc_mask]
            if len(calc_points) == 0:
                break

            # 向量化计算角度
            dx = np.abs(calc_points[:, 0] - current_wheel_point[0])
            dz = calc_points[:, 1] - current_wheel_point[1]
            # 过滤异常点
            idx_safe = dx > 1e-4
            dx = dx[idx_safe]
            dz = dz[idx_safe]
            curr_calc_pts = calc_points[idx_safe]
            if len(curr_calc_pts) == 0:
                break

            # 找最小角度
            angles = np.arctan2(dz, dx)
            min_angle_idx = np.argmin(angles)
            min_angle_rad = angles[min_angle_idx]

            # 当前的临时切点和斜率
            temp_target = curr_calc_pts[min_angle_idx]
            temp_m = (temp_target[1] - current_wheel_point[1]) / (temp_target[0] - current_wheel_point[0])
            temp_b = current_wheel_point[1] - temp_m * current_wheel_point[0]

            # 3. 碰撞检测（判断切线下面是否还有点）
            check_mask = (Point_chetou[:, 0] < (current_contact_x - 50)) & (Point_chetou[:, 0] > temp_target[0])
            check_points = Point_chetou[check_mask]
            points_below_line = False
            if len(check_points) > 0:
                line_z = temp_m * check_points[:, 0] + temp_b
                real_z = check_points[:, 1]
                if np.any(real_z < (line_z - 1.0)):
                    points_below_line = True

            # 4. 决策
            if not points_below_line:
                point_wheel = current_wheel_point
                point_target = temp_target
                angle_x = np.degrees(min_angle_rad)
                line_equation = [temp_m, temp_b]
                valid_points_for_plot = calc_points
                break
            else:
                current_contact_x -= step_size

    # # 6. 接近角可视化（原逻辑保留）
    # if line_equation is not None and point_target is not None and angle_x is not None:
    #     plt.figure(figsize=(12, 6))
    #     plt.axis('equal')

    #     # 1. 绘制整体轮廓（画清洗过后的）
    #     plt.scatter(xz_projection[:, 0][::10], xz_projection[:, 1][::10],
    #                 s=1, c='lightgray', label="整车截面(去噪后)")

    #     # 2. 绘制车头点
    #     plt.scatter(Point_chetou[:, 0], Point_chetou[:, 1],
    #                 s=1, c='blue', alpha=0.3, label="车头区域")

    #     # 3. 绘制关键点
    #     plt.scatter(point_wheel[0], point_wheel[1], s=80, c='green', marker='o', zorder=10, label="最终接地点")
    #     plt.scatter(point_target[0], point_target[1], s=80, c='red', marker='o', zorder=10, label="关键切点")

    #     # 4. 绘制接近角直线
    #     line_x = np.linspace(point_target[0] - 200, point_wheel[0] + 100, num=100)
    #     line_y = line_equation[0] * line_x + line_equation[1]
    #     plt.plot(line_x, line_y, 'k--', linewidth=1.5, label=f"接近角线 ({angle_x:.1f}°)")

    #     # 辅助线
    #     plt.axvline(x=p1, color='orange', linestyle=':', label="初始轮心X")

    #     plt.xlabel("X (mm)")
    #     plt.ylabel("Z (mm)")
    #     plt.title(f"接近角分析结果 (去噪+迭代)")
    #     plt.legend(loc='upper right')
    #     plt.grid(visible=True, linestyle='--', alpha=0.6)

    # 最终仅输出接近角结果（保留两位小数）
    if angle_x is not None:
        # print(f"接近角：{angle_x:.2f}")
        pass  # 空操作，保持语法合规
    else:
        # print("接近角计算失败")
        pass  # 空操作，保持语法合规

    # 显示所有可视化图
    # plt.show()

    return angle_x


# ===================== 运行程序 =====================
if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)