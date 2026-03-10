import trimesh
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局设置 =====================
# 设置中文字体（异常静默处理，避免字体报错）
try:
    plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
except Exception as e:
    pass

# ===================== 核心工具函数（保留挠度计算全部所需） =====================
def get_contour_from_stl(mesh, plane_z, plane):
    """计算STL与指定平面的交线轮廓"""
    try:
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
            return []

        intersections = trimesh.intersections.mesh_plane(mesh=mesh, plane_normal=plane_normal, plane_origin=plane_origin)
        points_crossing_plane = []
        for intersection in intersections:
            points_crossing_plane.append(np.array([intersection[0][0], intersection[0][1], intersection[0][2]]))
            points_crossing_plane.append(np.array([intersection[1][0], intersection[1][1], intersection[1][2]]))
        return points_crossing_plane
    except Exception as e:
        return []

def find_target_point1(points):
    """找前挡上端点"""
    try:
        if points is None or len(points) == 0:
            return None
        if isinstance(points, np.ndarray):
            points = points.tolist()
        sorted_points = sorted(points, key=lambda p: p[0])
        if len(sorted_points) < 3:
            return sorted_points[-1] if sorted_points else None
        prev_point = sorted_points[0]
        for current_point in sorted_points[2:]:
            if current_point[2] < prev_point[2]:
                return prev_point
            prev_point = current_point
        return sorted_points[-1]
    except Exception as e:
        return None

def find_target_point2(points):
    """找后挡上端点"""
    try:
        if points is None or len(points) == 0:
            return None
        if isinstance(points, np.ndarray):
            points = points.tolist()
        sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
        if len(sorted_points) < 3:
            return sorted_points[-1] if sorted_points else None
        prev_point = sorted_points[0]
        for current_point in sorted_points[2:]:
            if current_point[2] < prev_point[2]:
                return prev_point
            prev_point = current_point
        return sorted_points[-1]
    except Exception as e:
        return None

def max_z_difference(point1, point2, point3, target_index):
    """计算目标点z值与其他两点的最大绝对差值（挠度核心计算）"""
    try:
        # 过滤空点，避免报错
        valid_points = [p for p in [point1, point2, point3] if p is not None]
        if len(valid_points) < 2 or target_index >= len(valid_points):
            return 0.0
        # 取目标点并计算差值
        target_point = valid_points[target_index]
        target_z = target_point[2]
        diffs = [abs(target_z - p[2]) for i, p in enumerate(valid_points) if i != target_index]
        return max(diffs) if diffs else 0.0
    except Exception as e:
        return 0.0

def get_top3_points_by_x(points):
    """按x取前3个不同点（拟合直线用）"""
    try:
        if points is None or len(points) == 0:
            return []
        sorted_points = sorted(points, key=lambda point: point[0])
        seen = set()
        result = []
        for point in sorted_points:
            # 浮点精度处理，避免重复点
            point_tuple = tuple(np.round(point, 6))
            if point_tuple not in seen:
                seen.add(point_tuple)
                result.append(point)
                if len(result) == 3:
                    break
        return np.array(result)
    except Exception as e:
        return []

def linear_fit_xy(points_3d):
    """XZ平面线性拟合（前挡区域直线）"""
    try:
        points_3d = np.array(points_3d)
        if len(points_3d) < 2:
            return (0, 0), np.array([])
        x = points_3d[:, 0]
        z = points_3d[:, 2]
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, z, rcond=None)[0]
        y_fit = slope * x + intercept
        return (slope, intercept), y_fit
    except Exception as e:
        return (0, 0), np.array([])

def calculate_z_distances_to_line(points_3d, line_params):
    """计算点到拟合直线的垂直距离"""
    try:
        points_3d = np.array(points_3d)
        slope, intercept = line_params
        x = points_3d[:, 0]
        z = points_3d[:, 2]
        distances = np.abs(z - (slope * x + intercept)) / np.sqrt(1 + slope ** 2)
        return distances
    except Exception as e:
        return np.array([])

def find_closest_point(points_3d, distances):
    """找距离直线最近的点"""
    try:
        if len(distances) == 0:
            return -1, None, 0.0
        min_index = np.argmin(distances)
        closest_point = points_3d[min_index]
        min_distance = distances[min_index]
        return min_index, closest_point, min_distance
    except Exception as e:
        return -1, None, 0.0

# ===================== 主程序：计算顶棚挠度 =====================
def calculate(stl_path):
    # 初始化目标结果（默认0.00，异常时直接输出）
    dp_nd = None
    # 初始化可视化变量
    contour_points = np.array([])
    target1 = target2 = z_max = closest_point = None

    try:
        # 1. 加载小米STL模型并提取基础极值
        mesh = trimesh.load_mesh(stl_path)
        vertices = mesh.vertices
        z_min = np.min(vertices[:, 2])
        filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
        y_min = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
        y_max = filtered_vertices[np.argmax(filtered_vertices[:, 1])]
        y_mid = (y_max[1] + y_min[1]) / 2
        x_min = np.min(vertices[:, 0])
        x_max = np.max(vertices[:, 0])

        # 2. 获取XZ平面交线（y=mid平面）
        plane_z = y_mid
        contour_points = get_contour_from_stl(mesh, plane_z, 'xz')
        contour_points = np.array(contour_points) if contour_points else np.array([])
        if len(contour_points) == 0:
            raise Exception("未获取到平面交线")

        # 3. 筛选前/后挡区域，查找特征端点
        contour_points1 = contour_points[(contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] < (x_min + x_max)/2)]
        target1 = find_target_point1(contour_points1)
        contour_points2 = contour_points[(contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] > (x_min + x_max)/2)]
        target2 = find_target_point2(contour_points2)

        # 4. 查找顶棚最高点（x范围：target1.x+100 到 target2.x-600，适配小米模型）
        if target1 is not None and target2 is not None:
            contour_points3 = contour_points[(contour_points[:, 0] > target1[0] + 100) & (contour_points[:, 0] < (target2[0] - 600))]
            if len(contour_points3) > 0:
                z_max = contour_points3[np.argmax(contour_points3[:, 2])]

        # 5. 拟合前挡直线+找最近点（原逻辑保留，不影响挠度计算，仅可视化）
        if target1 is not None and len(contour_points1) > 0:
            top1 = get_top3_points_by_x(contour_points1[(contour_points1[:, 0] > (target1[0] + 10))])
            (slope, intercept), y_fit = linear_fit_xy(top1)
            points_3d = contour_points1[(contour_points1[:, 0] < target1[0])]
            if len(points_3d) > 0:
                distances = calculate_z_distances_to_line(points_3d, (slope, intercept))
                min_index, closest_point, min_distance = find_closest_point(points_3d, distances)

        # 6. 计算顶棚挠度（核心：z_max为目标点，索引2）
        if target1 is not None and target2 is not None and z_max is not None:
            dp_nd = max_z_difference(target1, target2, z_max, 2)

    except Exception as e:
        pass  # 异常静默处理，直接输出初始值0.00

    # # 可视化（可按需删除以下所有可视化代码）
    # try:
    #     if len(contour_points) > 0:
    #         point_size = 2
    #         plt.figure(figsize=(8, 6))
    #         # 绘制整体交线
    #         plt.plot(contour_points[:, 0], contour_points[:, 2], 'o', color='g', markersize=point_size, alpha=0.6)
    #         # 标注所有特征点
    #         if target1 is not None:
    #             plt.plot(target1[0], target1[2], 'o', color='r', markersize=6, label='前挡上端点')
    #         if target2 is not None:
    #             plt.plot(target2[0], target2[2], 'o', color='b', markersize=6, label='后挡上端点')
    #         if z_max is not None:
    #             plt.plot(z_max[0], z_max[2], 'o', color='c', markersize=6, label='截面最高点(顶棚)')
    #         if closest_point is not None:
    #             plt.plot(closest_point[0], closest_point[2], 'o', color='k', markersize=6, label='距离最小的点')
    #         plt.title('顶棚挠度特征点（小米模型）', fontsize=12)
    #         plt.xlabel('X-axis', fontsize=10)
    #         plt.ylabel('Z-axis', fontsize=10)
    #         plt.axis('equal')
    #         plt.grid(True, alpha=0.3)
    #         plt.legend(fontsize=10)
    #         plt.show()
    # except Exception as e:
    #     pass
    return dp_nd

if __name__ == "__main__":
    dp_nd = calculate(r"F:\一汽风噪\点云文件\小米.stl")
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)
