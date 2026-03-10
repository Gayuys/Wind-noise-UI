import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm


def extract_and_project(stl_file):
    # 加载 STL 文件
    mesh = trimesh.load_mesh(stl_file)

    # 获取所有顶点
    vertices = mesh.vertices
    if vertices.size == 0:
        raise ValueError("模型中没有顶点数据")

    # 找到 x 轴的最小值和最大值
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])

    # 计算中间位置
    x_mid = (x_min + x_max) / 2

    # 定义投影点的范围 (x_mid 左右各 20mm)
    x_lower_bound = x_mid - 20
    x_upper_bound = x_mid + 20

    # 筛选满足条件的点
    selected_points = vertices[(vertices[:, 0] >= x_lower_bound) & (vertices[:, 0] <= x_upper_bound)]

    if selected_points.size == 0:
        raise ValueError("没有找到符合X范围的点")

    # 投影到 yz 平面
    projected_points = selected_points[:, 1:]
    return vertices, projected_points


def find_special_points(vertices, projected_points):
    # 筛选 z 轴大于 300 的点
    valid_vertices = vertices[vertices[:, 2] > 300]
    y_min_point = None
    if len(valid_vertices) > 0:
        y_min_index = np.argmin(valid_vertices[:, 1])
        y_min_point = valid_vertices[y_min_index, 1:]  # 投影到 yz 面上
    else:
        print("警告: 没有找到z轴大于300的点")

    # 筛选 -100mm < y < 100mm 的点
    y_range_points = projected_points[(projected_points[:, 0] > -100) & (projected_points[:, 0] < 100)]
    z_max_point = None
    if len(y_range_points) > 0:
        # 找到该范围内 z 轴最大值点 (点2)
        z_max_index = np.argmax(y_range_points[:, 1])
        z_max_point = y_range_points[z_max_index]
    else:
        print("警告: 没有找到y在-100到100之间的点")

    return y_min_point, z_max_point


def extract_points_in_range(projected_points, point1_z, point2_z):
    # 定义 Z 范围
    z_min_1 = point1_z + 20  # 点1的 Z 坐标向下移动 20mm
    z_min_2 = point2_z - 150  # 点2的 Z 坐标向下移动 150mm

    # 确定 Z 坐标范围为两者之间
    z_range_min = min(z_min_1, z_min_2)  # 下边界
    z_range_max = max(z_min_1, z_min_2)  # 上边界

    # 计算 Y 范围的一半
    y_min = np.min(projected_points[:, 0])  # Y 最小值
    y_max = np.max(projected_points[:, 0])  # Y 最大值
    y_half = (y_min + y_max) / 2  # Y 范围的中点

    # 筛选符合范围的点
    filtered_points = projected_points[
        (projected_points[:, 1] >= z_range_min) &  # Z 坐标 >= 下边界
        (projected_points[:, 1] <= z_range_max) &  # Z 坐标 <= 上边界
        (projected_points[:, 0] < y_half)  # Y 坐标小于 Y 中点
        ]

    if filtered_points.size == 0:
        raise ValueError("没有找到符合Z和Y范围的点")

    return filtered_points


def linear_fit_and_angles(filtered_points):
    # 提取 Y 和 Z 坐标
    y_coords = filtered_points[:, 0]
    z_coords = filtered_points[:, 1]

    # 进行线性拟合，返回斜率和截距
    coeffs = np.polyfit(y_coords, z_coords, 1)
    slope = coeffs[0]  # 斜率
    intercept = coeffs[1]  # 截距

    # 计算拟合线与水平和竖直方向的角度
    angle_with_horizontal = np.degrees(np.arctan(slope))
    angle_with_vertical = 90 - angle_with_horizontal

    return slope, intercept, angle_with_horizontal, angle_with_vertical


def plot_all_points_with_highlight_and_fit(projected_points, filtered_points, slope, intercept, point1, point2):
    # 设置中文字体
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 替换为您的系统中文字体路径
    font_prop = fm.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制完整投影点
    plt.figure(figsize=(10, 8))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=1, color='blue', label='投影点')

    # 绘制筛选出的点
    plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=5, color='green', label='侧窗点')

    # 绘制拟合线
    y_fit = np.linspace(np.min(filtered_points[:, 0]), np.max(filtered_points[:, 0]), 100)
    z_fit = slope * y_fit + intercept
    plt.plot(y_fit, z_fit, color='red', label='拟合直线')

    # 标注点1 (黑色)
    plt.scatter(point1[0], point1[1], s=50, color='black', label='点1', edgecolors='black', linewidths=0.5)

    # 标注点2 (紫色)
    plt.scatter(point2[0], point2[1], s=50, color='purple', label='点2', edgecolors='black', linewidths=0.5)

    # 图例和标注
    # plt.xlabel('Y', fontproperties=font_prop)
    # plt.ylabel('Z', fontproperties=font_prop)
    # plt.title('投影点与拟合结果', fontproperties=font_prop)
    # plt.legend(prop=font_prop)
    # plt.axis('equal')
    # plt.grid(True)
    # plt.show()


# if __name__ == "__main__":
def calculate(stl_file):
    cechuangqingjiao = None

    # 提取并投影点
    vertices, projected_points = extract_and_project(stl_file)
    if vertices is None or projected_points is None:
        raise ValueError("无法获取顶点数据，程序终止")

    # 找到特殊点 (点1 和 点2)
    point1, point2 = find_special_points(vertices, projected_points)
    if point1 is None or point2 is None:
        raise ValueError("无法确定点1或点2，程序终止")

    # 提取范围内的点
    filtered_points = extract_points_in_range(projected_points, point1[1], point2[1])
    if filtered_points is None:
        raise ValueError("无法筛选出有效点，程序终止")

    # 线性拟合并计算角度
    slope, intercept, angle_with_horizontal, angle_with_vertical = linear_fit_and_angles(filtered_points)
    if slope is None:
        raise ValueError("无法完成线性拟合，程序终止")

    cechuangqingjiao = angle_with_vertical

    # 可视化结果
    # plot_all_points_with_highlight_and_fit(projected_points, filtered_points, slope, intercept, point1, point2)

    return cechuangqingjiao


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\AITO M9 Outer Surface.stl")
    print(rear_windshield_angle)