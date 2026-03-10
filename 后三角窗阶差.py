import trimesh
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def get_contour_from_stl(mesh, plane_z, plane):
    """
    使用 trimesh 计算 STL 文件与平面的交线轮廓
    返回: 交线段列表（每段为两个点的数组）
    """
    # 定义平面参数
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

    # 计算网格与平面的交线
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


def find_target_point(points):
    """
    从y值最小点开始向上查找，当x值首次增大时返回上一个点，仅考虑y值<=最小y值的点
    """
    # 处理空输入
    if points is None or len(points) == 0:
        return None

    # 将NumPy数组转换为列表
    if isinstance(points, np.ndarray):
        points = points.tolist()

    # 按x值升序排序
    sorted_points = sorted(points, key=lambda p: p[0], reverse=False)

    # 初始化x最小值点
    prev_point = sorted_points[0]

    # 从第二个点开始遍历
    for current_point in sorted_points[1:]:
        if current_point[1] < prev_point[1]:
            return prev_point
        prev_point = current_point

    # 如果所有点的x值都保持非递减，则返回最后一个点
    return sorted_points[-1]


# 主程序执行部分
def calculate(stl_path):
    posterior_window_difference = None
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标
    vertices = mesh.vertices

    # 找到z轴最小值
    z_min = np.min(vertices[:, 2])
    # 找到y轴最大、小值
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])

    # 找到y轴的中值
    y_mid = (y_max + y_min) / 2

    # 筛选顶点并计算point1
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 设置平面 z 坐标
    plane_z = point1[2] + 80
    # 获取与 XY 平面的交点
    contour_points, _ = get_contour_from_stl(mesh, plane_z, 'xy')
    # 转换为 NumPy 数组以便处理
    contour_points = np.array(contour_points)

    # 筛选x轴大于（点1 x轴值+500mm）的点
    contour_points1 = contour_points[contour_points[:, 0] > (point1[0] + 500)]
    x_max = contour_points1[np.argmax(contour_points1[:, 0])]
    contour_points1 = contour_points1[((contour_points1[:, 0] > (x_max[0] - 700)) & (contour_points1[:, 1] < y_mid))]

    # 找目标点
    points = contour_points1
    target = find_target_point(points)

    # 找钣金点
    contour_points2 = contour_points1[contour_points1[:, 0] >= target[0]]
    y_min_point = contour_points2[np.argmin(contour_points2[:, 1])]

    # 计算后三角窗阶差
    posterior_window_difference = abs(y_min_point[1] - target[1])
    # print("后三角窗阶差:", posterior_window_difference)

    # # 绘制图形
    # if contour_points1 is not None and target is not None and y_min_point is not None:
    #     # 设置点的大小
    #     point_size = 2
    #     plt.figure()
    #     plt.plot(contour_points1[:, 0], contour_points1[:, 1], 'o', color='g', markersize=point_size)
    #     plt.plot(target[0], target[1], 'o', color='c', markersize=5, label='目标点')
    #     plt.plot(y_min_point[0], y_min_point[1], 'o', color='m', markersize=5, label='钣金点')
    #     plt.title('Intersection Points with XZ Plane')
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Z-axis')
    #     plt.axis('equal')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    # else:
    #     print("缺少绘图所需的数据，无法绘制图形")

    return posterior_window_difference


if __name__ == "__main__":
    posterior_window_difference = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(posterior_window_difference)