import trimesh
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局设置 =====================
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 统一STL文件路径（修改为你的实际路径）
MESH_FILE_PATH =  r"F:\Users\liph\Desktop\自动提取新增\自动0130版本\CAR9.stL"

# ===================== 核心工具函数 =====================
def get_contour_from_stl(mesh, plane_z, plane):
    """计算STL与平面的交线轮廓"""
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
        return [], []

    intersections = trimesh.intersections.mesh_plane(mesh=mesh, plane_normal=plane_normal, plane_origin=plane_origin)
    points_crossing_plane = []
    for intersection in intersections:
        points_crossing_plane.append(np.array(intersection[0]))
        points_crossing_plane.append(np.array(intersection[1]))
    return points_crossing_plane, intersections

def line_plane_intersection(intersections, plane):
    """计算线面交点"""
    plane_normal = np.array(plane[:3])
    plane_d = plane[3]
    if abs(plane_normal[2]) > 1e-6:
        plane_origin = np.array([0, 0, -plane_d / plane_normal[2]])
    elif abs(plane_normal[1]) > 1e-6:
        plane_origin = np.array([0, -plane_d / plane_normal[1], 0])
    else:
        plane_origin = np.array([-plane_d / plane_normal[0], 0, 0])

    def segment_plane_intersection(line_start, line_end):
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
        if len(line_segment) < 2:
            continue
        inter = segment_plane_intersection(line_segment[0], line_segment[1])
        if inter is not None:
            intersection_points.append(inter)
    return np.array(intersection_points) if intersection_points else np.array([])

def get_top3_points_by_x1(points):
    """按x坐标从小到大取前2个不同点"""
    if not points.size:
        return []
    points_list = points.tolist()
    sorted_points = sorted(points_list, key=lambda p: p[0])
    seen = set()
    result = []
    for point in sorted_points:
        point_tuple = tuple(round(p, 6) for p in point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)
            if len(result) == 2:
                break
    return result

def get_top3_points_by_x2(points):
    """按z坐标从小到大取前2个不同点"""
    if not points.size:
        return []
    points_list = points.tolist()
    sorted_points = sorted(points_list, key=lambda p: p[2])
    seen = set()
    result = []
    for point in sorted_points:
        point_tuple = tuple(round(p, 6) for p in point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            result.append(point)
            if len(result) == 2:
                break
    return result

def get_front_point(points):
    """提取排序后的第2、3个点（前轮腔前尺寸专用）"""
    if not points.size:
        return None, None
    points_list = points.tolist()
    sorted_points = sorted(points_list, key=lambda p: p[0])
    unique_points = []
    seen = set()
    for point in sorted_points:
        point_tuple = tuple(round(p, 6) for p in point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)
    point2 = unique_points[1] if len(unique_points) > 1 else None
    point3 = unique_points[2] if len(unique_points) > 2 else None
    return point2, point3

# ===================== 主程序：计算前轮腔前X向尺寸 =====================
def calculate(stl_path):
    # 初始化结果
    qian_size = None
    point2, point3 = None, None
    contour_points3 = np.array([])

    # 1. 加载模型
    mesh = trimesh.load_mesh(stl_path)
    vertices = mesh.vertices
    z_min = np.min(vertices[:, 2])
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])

    # 2. 处理XY平面交线，筛选左前车轮区域
    plane_z = z_min + 80
    contour_points, _ = get_contour_from_stl(mesh, plane_z, 'xy')
    contour_points = np.array(contour_points)
    if len(contour_points) > 0:
        contour_points1 = contour_points[(contour_points[:, 1] < (y_min + y_max) / 2) & (contour_points[:, 0] < (x_min + x_max) / 2)]
        y_min_1 = np.min(contour_points1[:, 1]) if len(contour_points1) > 0 else 0
        y_max_1 = np.max(contour_points1[:, 1]) if len(contour_points1) > 0 else 0
        y_mid_1 = (y_min_1 + y_max_1) / 2
    else:
        y_mid_1 = 0

    # 3. 处理XZ平面交线，筛选前半部分
    contour_points2, intersections = get_contour_from_stl(mesh, y_mid_1, 'xz')
    contour_points2 = np.array(contour_points2)
    if len(contour_points2) > 0:
        contour_points3 = contour_points2[(contour_points2[:, 0] < (x_min + x_max) / 2)]

    # 4. 计算轮心X值
    plane1 = (0, 0, 1, -(z_min + 80))
    intersection_point1 = line_plane_intersection(intersections, plane1)
    result1 = get_top3_points_by_x1(intersection_point1)
    result1 = np.array(result1) if result1 else np.array([])
    if len(result1) > 0:
        x_1 = result1[np.argmin(result1[:, 0])]
        x_2 = result1[np.argmax(result1[:, 0])]
        x_mid_12 = (x_1[0] + x_2[0]) / 2
    else:
        x_mid_12 = 0

    # 5. 计算轮心Z值
    plane2 = (1, 0, 0, -x_mid_12)
    intersection_point2 = line_plane_intersection(intersections, plane2)
    result2 = get_top3_points_by_x2(intersection_point2)
    result2 = np.array(result2) if result2 else np.array([])
    if len(result2) > 0:
        z_1 = result2[np.argmin(result2[:, 2])]
        z_2 = result2[np.argmax(result2[:, 2])]
        z_mid_12 = (z_1[2] + z_2[2]) / 2
    else:
        z_mid_12 = 0

    # 6. 提取特征点，计算前轮腔前X向尺寸
    plane3 = (0, 0, 1, -z_mid_12)
    intersection_point3 = line_plane_intersection(intersections, plane3)
    if len(intersection_point3) > 0:
        intersection_point3 = intersection_point3[(intersection_point3[:, 0] < (x_min + x_max) / 2)]
    point2, point3 = get_front_point(intersection_point3)
    if point2 is not None and point3 is not None:
        qian_size = abs(point3[0] - point2[0])

    # # 可视化（可按需删除以下可视化代码块）
    # point_size = 2
    # plt.figure()
    # if len(contour_points3) > 0:
    #     plt.plot(contour_points3[:, 0], contour_points3[:, 2], 'o', color='g', markersize=point_size)
    # if point2 is not None:
    #     plt.plot(point2[0], point2[2], 'o', color='c', markersize=5, label='点2')
    # if point3 is not None:
    #     plt.plot(point3[0], point3[2], 'o', color='m', markersize=5, label='点3')
    # plt.title('前轮腔前X向尺寸特征点')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # 输出前轮腔前X向尺寸
    #print(f"前轮腔前X向尺寸:{qian_size:.2f}")
    return qian_size

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)