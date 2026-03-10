import trimesh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from sklearn.linear_model import LinearRegression

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def project_3d_to_2d(points, plane):
    """
    将三维点投影到二维平面上
    :param points: 三维点集，格式为 [[x1, y1, z1], [x2, y2, z2], ...]
    :param plane: 投影平面，可以是 'xy', 'xz', 'yz'
    :return: 投影后的二维点集
    """
    if plane == 'xy':
        return np.array([point[:2] for point in points])
    elif plane == 'xz':
        return np.array([point[[0, 2]] for point in points])
    elif plane == 'yz':
        return np.array([point[1:] for point in points])
    else:
        raise ValueError("无效的平面参数，请选择 'xy', 'xz' 或 'yz'")


def distance_point_to_line(m, b, x0, y0):
    # 直线方程参数 A, B, C
    A = m
    B = -1
    C = b

    # 计算点到直线的距离
    numerator = abs(A * x0 + B * y0 + C)
    denominator = math.sqrt(A ** 2 + B ** 2)
    distance = numerator / denominator
    return distance


def calculate_line(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    # 计算斜率
    if x2 - x1 == 0:
        raise ValueError("两点X坐标相同，无法计算斜率")
    m = (y2 - y1) / (x2 - x1)

    # 计算截距
    b = y1 - m * x1

    return m, b


def get_contour_from_stl(mesh, plane_z, plane):
    """
    使用 trimesh 计算 STL 文件与平面 z=plane_z 的交线轮廓
    stl_file: STL 文件路径
    plane_z: 平面 z 坐标
    返回: 交线段列表（每段为两个点的数组）
    """
    # 定义平面
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


# 设置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate(stl_path):
    distance = None
    # 加载网格文件
    mesh = trimesh.load_mesh(stl_path)

    points = mesh.vertices
    if points.size == 0:
        raise ValueError("模型中没有有效的顶点数据")

    # 获取交线轮廓
    plane_z = min(points[:, 2]) + 500
    segments, intersections = get_contour_from_stl(mesh, plane_z, 'xy')
    if segments is None or intersections is None:
        raise ValueError("无法获取有效的交线轮廓数据")
    segments = np.unique(segments, axis=0)
    segments = np.array(segments)

    # 投影到XZ平面
    xy_projection = project_3d_to_2d(segments, 'xy')
    if xy_projection is None:
        raise ValueError("投影操作失败")
    #print(xy_projection.shape)

    middle_point_y = (max(xy_projection[:, 1]) + min(xy_projection[:, 1])) / 2
    middle_point_x = (max(xy_projection[:, 0]) + min(xy_projection[:, 0])) / 2

    # 计算前轮轮心位置
    plane_z2 = min(points[:, 2]) + 100
    segments2, intersections2 = get_contour_from_stl(mesh, plane_z2, 'xy')
    if segments2 is None or intersections2 is None:
        raise ValueError("无法获取第二个平面的交线数据")
    segments2 = np.array(segments2)
    point_zuoqian = segments2[
        (segments2[:, 0] <= middle_point_x) &
        (segments2[:, 1] <= middle_point_y)
        ]
    if len(point_zuoqian) == 0:
        raise ValueError("未找到符合条件的左前点")
    point_qianlunzhongxing = (max(point_zuoqian[:, 0]) + min(point_zuoqian[:, 0])) / 2

    # 筛选点云中车头左半部分的点
    Point_chetou = xy_projection[
        (xy_projection[:, 0] <= point_qianlunzhongxing) &
        (xy_projection[:, 0] >= point_qianlunzhongxing - 600) &
        (xy_projection[:, 1] <= middle_point_y)
        ]
    Point_chetou = np.array(Point_chetou)
    if len(Point_chetou) == 0:
        raise ValueError("未筛选到有效的车头点")
    qianlunzhongxing1 = Point_chetou[(Point_chetou[:, 0] == max(Point_chetou[:, 0]))]
    qianlunzhongxing = qianlunzhongxing1[(qianlunzhongxing1[:, 1] == min(qianlunzhongxing1[:, 1]))]
    if len(qianlunzhongxing) == 0:
        raise ValueError("未找到有效的前轮中心点")
    #print(Point_chetou.shape)

    # 将筛选出来的点沿X轴等分为60段，计算每一段Y轴的最小值
    x_min_filtered = np.min(Point_chetou[:, 0])
    x_max_filtered = np.max(Point_chetou[:, 0])
    x_segment_width = (x_max_filtered - x_min_filtered) / 60

    Point_qianbao = []
    Point_qianlun = []
    a = 0  # 点云归类指示器，a=0时的点归与前保，a=1时的点归为前轮
    # 将轮毂中心点Y坐标+100作为初始最小值，保证不会将前保前端错误分类
    current_min_y = qianlunzhongxing[0][1] + 100

    for i in range(60):
        # 获取当前段的 X 坐标范围
        x_start = x_min_filtered + i * x_segment_width
        x_end = x_min_filtered + (i + 1) * x_segment_width

        # 筛选当前段内的点
        segment_points = Point_chetou[
            (Point_chetou[:, 0] >= x_start) &
            (Point_chetou[:, 0] < x_end)
            ]

        if len(segment_points) > 0:
            # 找到该段内 Y 轴方向最小的点
            min_y_point = segment_points[np.argmin(segment_points[:, 1])]
        else:
            min_y_point = [(x_start + x_end) / 2, 0]
        if a == 0:
            if min_y_point[1] < current_min_y:
                Point_qianbao.append(min_y_point)
            else:
                a = 1
        else:
            Point_qianlun.append(min_y_point)

    Point_qianbao = np.array(Point_qianbao)
    Point_qianlun = np.array(Point_qianlun)

    if len(Point_qianbao) == 0 or len(Point_qianlun) == 0:
        raise ValueError("前保或前轮拟合点为空")

    # print(Point_qianbao.shape)
    # print(Point_qianlun.shape)

    # 拟合前保末端直线
    if len(Point_qianbao) == 0:
        raise ValueError("前保拟合点为空，无法进行拟合")
    point_qianbaomoduan = Point_qianbao[Point_qianbao[:, 0] > max(Point_qianbao[:, 0]) - 50]
    if len(point_qianbaomoduan) < 2:
        raise ValueError("前保末端点不足，无法拟合直线")

    x1, y1 = point_qianbaomoduan[:, 0], point_qianbaomoduan[:, 1]
    m_A, b_A = np.polyfit(x1, y1, 1)

    if len(Point_qianlun) == 0:
        raise ValueError("前轮拟合点为空，无法获取轮胎边缘点")
    point = Point_qianlun[Point_qianlun[:, 1] == min(Point_qianlun[:, 1])]  # 获取轮胎外边缘的点
    if len(point) == 0:
        raise ValueError("未找到有效的轮胎外边缘点")

    x = np.linspace(min(point_qianbaomoduan[:, 0]), point[0][0], 100)
    y = m_A * x + b_A

    distance = distance_point_to_line(m_A, b_A, point[0][0], point[0][1])  # 计算前保侧切线到前轮的距离
    if distance is None:
        raise ValueError("距离计算失败")

    #print('前保侧切线到前轮的距离为：', distance)

    # # 绘制散点图
    # plt.axis('equal')
    # plt.scatter(segments[:, 0], segments[:, 1], s=5, c='red', label="车身切面")
    # plt.scatter(Point_chetou[:, 0], Point_chetou[:, 1], s=5, c='green', label="半车头")
    # plt.scatter(Point_qianbao[:, 0], Point_qianbao[:, 1], s=5, c='blue', label="前保拟合点")
    # plt.scatter(Point_qianlun[:, 0], Point_qianlun[:, 1], s=5, c='yellow', label="前轮拟合点")
    # plt.scatter(x, y, s=5, c='black', label="前保侧切线")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return distance


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)