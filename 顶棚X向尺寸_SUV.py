import trimesh
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局设置 =====================
# 设置中文字体（移除异常静默处理）
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# ===================== 核心工具函数（仅保留x向尺寸计算所需） =====================
def get_contour_from_stl(mesh, plane_z, plane):
    """计算STL与指定平面的交线轮廓"""
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

def find_target_point1(points):
    """找前挡上端点"""
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

def find_target_point2(points):
    """找后挡上端点"""
    if points is None or len(points) == 0:
        return None
    if isinstance(points, np.ndarray):
        points = points.tolist()
    sorted_points = sorted(points, key=lambda p: p[2])
    if not sorted_points:
        return None
    prev_point = sorted_points[0]
    for current_point in sorted_points:
        if current_point[0] > prev_point[0]:
            return prev_point
        prev_point = current_point
    return sorted_points[-1]

def find_target_point3(points):
    """找SUV后盖点"""
    if points is None or len(points) == 0:
        return None
    if isinstance(points, np.ndarray):
        points = points.tolist()
    sorted_points = sorted(points, key=lambda p: p[2], reverse=True)
    if not sorted_points:
        return None
    prev_point = sorted_points[0]
    for current_point in sorted_points:
        if current_point[0] < prev_point[0]:
            return prev_point
        prev_point = current_point
    return sorted_points[-1]

# ===================== 主程序：计算顶棚x向尺寸 =====================
def calculate(stl_path):
    # 初始化目标结果
    dp_x = None
    # 初始化可视化所需变量
    mesh = None
    contour_points = np.array([])
    target1 = target2 = target3 = z_max = None

    # 1. 加载模型并提取基础极值
    mesh = trimesh.load_mesh(stl_path)
    vertices = mesh.vertices
    z_min = np.min(vertices[:, 2])
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    y_min = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    y_max = filtered_vertices[np.argmax(filtered_vertices[:, 1])]
    y_mid = (y_max[1] + y_min[1]) / 2
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])

    # 2. 获取XZ平面交线并筛选
    plane_z = y_mid
    contour_points = get_contour_from_stl(mesh, plane_z, 'xz')
    contour_points = np.array(contour_points) if contour_points else np.array([])
    # 筛选前挡部分，找前挡上端点
    contour_points1 = contour_points[(contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] < (x_min + x_max)/2)] if len(contour_points) > 0 else np.array([])
    target1 = find_target_point1(contour_points1)
    # 筛选后挡部分，找后挡上端点
    contour_points2 = contour_points[(contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] > (x_min + x_max)/2)] if len(contour_points) > 0 else np.array([])
    target2 = find_target_point2(contour_points2)
    # 找SUV后盖点
    points3 = contour_points[(contour_points[:, 2] > target2[2]) & (contour_points[:, 0] > (target2[0] - 50))] if (target2 is not None and len(contour_points) > 0) else np.array([])
    target3 = find_target_point3(points3)

    # 3. 计算顶棚x向尺寸
    if target1 is not None and target3 is not None:
        dp_x = abs(target3[0] - target1[0])

    # 可视化（可按需删除以下所有可视化代码）
    # if mesh is not None and len(contour_points) > 0:
    #     point_size = 2
    #     plt.figure()
    #     # 绘制整体交线
    #     plt.plot(contour_points[:, 0], contour_points[:, 2], 'o', color='g', markersize=point_size)
    #     # 标注特征点
    #     if target1 is not None:
    #         plt.plot(target1[0], target1[2], 'o', color='r', markersize=5, label='前挡上端点')
    #     if target2 is not None:
    #         plt.plot(target2[0], target2[2], 'o', color='b', markersize=5, label='后挡上端点')
    #     if target3 is not None:
    #         plt.plot(target3[0], target3[2], 'o', color='m', markersize=5, label='SUV后盖点')
    #     plt.title('顶棚x向尺寸特征点')
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Z-axis')
    #     plt.axis('equal')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    return dp_x

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)