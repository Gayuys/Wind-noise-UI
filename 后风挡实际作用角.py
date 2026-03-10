import trimesh
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局设置 =====================
# 设置中文字体（移除异常捕获）
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

    intersections = trimesh.intersections.mesh_plane(mesh=mesh, plane_normal=plane_normal,
                                                     plane_origin=plane_origin)
    points_crossing_plane = []
    for intersection in intersections:
        points_crossing_plane.append(np.array([intersection[0][0], intersection[0][1], intersection[0][2]]))
        points_crossing_plane.append(np.array([intersection[1][0], intersection[1][1], intersection[1][2]]))
    return points_crossing_plane, intersections


def find_target_point1(points):
    """找车尾气流分离点"""
    if points is None or len(points) == 0:
        return None
    if isinstance(points, np.ndarray):
        points = points.tolist()
    sorted_points = sorted(points, key=lambda p: p[0])
    if len(sorted_points) < 3:
        return sorted_points[-1] if sorted_points else None
    prev_point = sorted_points[0]
    for current_point in sorted_points[2:]:
        if current_point[0] < prev_point[0]:
            return prev_point
        prev_point = current_point
    return sorted_points[-1]


def find_target_point2(points):
    """找后挡上端点（适配小米模型的x降序排序）"""
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


def line_plane_intersection(intersections, plane):
    """
    计算点云交线和平面的交点
    :param intersections: 获取的点云交线，格式为 [[x1, y1, z1], [x2, y2, z2], ...]
    :param plane: 平面的参数，格式为 (A, B, C, D)
    :return: 直线和平面的交点
    """
    plane_normal = np.array(plane[:3])
    plane_d = plane[3]
    if abs(plane_normal[2]) > 1e-6:  # 如果 c != 0
        plane_origin = np.array([0, 0, -plane_d / plane_normal[2]])
    elif abs(plane_normal[1]) > 1e-6:  # 如果 b != 0
        plane_origin = np.array([0, -plane_d / plane_normal[1], 0])
    else:  # 假设 a != 0
        plane_origin = np.array([-plane_d / plane_normal[0], 0, 0])

    # 定义一个函数来计算线段与平面的交点
    def line_plane_intersection(line_start, line_end, plane_normal, plane_d, plane_origin):
        """
        计算线段与平面的交点
        参数：
            line_start: 线段起点 (3,)
            line_end: 线段终点 (3,)
            plane_normal: 平面法向量 (3,)
            plane_d: 平面方程的常数项 d
            plane_origin: 平面上的一个点 (3,)
        返回：
            交点 (3,) 或 None（如果没有交点或交点不在线段上）
        """
        # 线段方向向量
        line_dir = line_end - line_start
        # 计算线段方向与平面法向量的点积
        denom = np.dot(line_dir, plane_normal)

        # 如果 denom 接近 0，说明线段与平面平行，无交点
        if abs(denom) < 1e-6:
            return None

        # 计算参数 t：(plane_normal * (plane_origin - line_start)) / (plane_normal * line_dir)
        t = np.dot(plane_normal, plane_origin - line_start) / denom

        # 检查交点是否在线段范围内 (t 在 [0, 1] 之间)
        if 0 <= t <= 1:
            # 计算交点
            intersection_point = line_start + t * line_dir
            # 验证交点是否满足平面方程（可选，增加鲁棒性）
            if abs(np.dot(plane_normal, intersection_point) + plane_d) < 1e-6:
                return intersection_point
        return None

    # 存储所有交点
    intersection_points = []

    # 遍历每条交线段，计算与第二个平面的交点
    for line_segment in intersections:
        line_start = line_segment[0]  # 线段起点
        line_end = line_segment[1]  # 线段终点

        # 计算交点
        intersection = line_plane_intersection(
            line_start, line_end, plane_normal, plane_d, plane_origin
        )

        # 如果有有效交点，添加到列表
        if intersection is not None:
            intersection_points.append(intersection)

    # 转换为 NumPy 数组（可选）
    intersection_point = np.array(intersection_points)

    return intersection_point


# ===================== 主程序：计算顶棚x向尺寸 =====================
def calculate(stl_path):
    theta_x = None
    # 初始化目标结果（默认0.00，异常时直接输出）
    dp_x = 0.00
    # 初始化可视化变量
    contour_points = np.array([])
    target1 = target2 = None

    # 1. 加载小米STL模型并提取基础极值
    mesh = trimesh.load_mesh(stl_path)
    vertices = mesh.vertices
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])
    high = z_max - z_min
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    y_min = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    y_max = filtered_vertices[np.argmax(filtered_vertices[:, 1])]
    y_mid = (y_max[1] + y_min[1]) / 2
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])

    # 2. 获取XZ平面交线（y=mid平面）
    plane_z = y_mid
    contour_points, intersections = get_contour_from_stl(mesh, plane_z, 'xz')
    contour_points = np.array(contour_points) if contour_points else np.array([])
    if len(contour_points) == 0:
        raise Exception("未获取到平面交线")

    # 3. 筛选后部气流分离点，查找特征端点
    # 顶棚区域：x>中点 + z>y_min.z+200
    contour_points2 = contour_points[
        (contour_points[:, 2] > (y_min[2] + 200)) & (contour_points[:, 0] > (x_min + x_max) / 2)]
    target2 = find_target_point2(contour_points2)
    # 车尾区域
    contour_points1 = contour_points[
        (contour_points[:, 2] < (target2[2] - 100)) & (contour_points[:, 0] > (x_min + x_max) / 2) & (
                    contour_points[:, 2] > (y_min[2] - high * 0.1))]
    target1 = find_target_point1(contour_points1)

    # print(target2)
    # print(target1)

    x_data = [target1[0], target2[0]]
    y_data = [target1[2], target2[2]]

    slope, intercept = np.polyfit(x_data, y_data, 1)

    # 计算夹角（弧度）
    theta_radians = np.arctan(slope)

    # 将弧度转换为度数
    theta_degrees = np.degrees(theta_radians)
    theta_x = -1 * theta_degrees
    # print("后风挡实际作用角为：", theta_x)

    # 生成 x 值
    x0 = np.linspace(min(x_data), max(x_data), 100)

    # 计算 y 值
    y0 = slope * x0 + intercept

    # 绘制散点图
    # plt.axis('equal')
    # plt.scatter(contour_points[:, 0], contour_points[:, 2], s=5, c='blue', label="整车截面")
    # plt.scatter(x0, y0, s=10, c='red', label="顶棚")
    # plt.xlabel("X")
    # plt.ylabel("Z")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return theta_x

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)