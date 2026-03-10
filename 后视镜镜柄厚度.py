import numpy as np
import trimesh
import matplotlib.pyplot as plt

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def calculate(stl_path):
    z_distance_difference = None
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标
    vertices = mesh.vertices
    if vertices.size == 0:
        raise ValueError("模型中没有有效的顶点数据")

    # 找到z轴最小值
    z_min = np.min(vertices[:, 2])

    # 筛选满足条件的点：z轴大于z轴最小值300mm的点
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    if len(filtered_vertices) == 0:
        raise ValueError("没有找到Z轴大于最小值+300的点")

    # 在这些筛选后的点中找y轴最大值对应的点（点1）
    point1 = filtered_vertices[np.argmax(filtered_vertices[:, 1])]

    # 筛选点
    z_threshold = point1[2] - 100
    y_upper = point1[1] - 200
    y_lower = point1[1] - 210
    x_threshold = point1[0] + 100

    filtered_vertices_2 = filtered_vertices[
        (filtered_vertices[:, 2] > z_threshold) &
        (filtered_vertices[:, 1] < y_upper) &
        (filtered_vertices[:, 1] > y_lower) &
        (filtered_vertices[:, 0] < x_threshold)
        ]
    if len(filtered_vertices_2) == 0:
        raise ValueError("第二次筛选未找到符合条件的点")

    # 将筛选的点投影到xz平面
    projected_points = filtered_vertices_2[:, [0, 2]]

    # 计算投影点的z轴最大值和最小值
    z_max = np.max(projected_points[:, 1])  # Z轴最大值
    z_min = np.min(projected_points[:, 1])  # Z轴最小值

    # 计算Z轴的距离差
    z_distance_difference = z_max - z_min

    # 输出结果
    # print(f"Z轴最大值: {z_max}")
    # print(f"Z轴最小值: {z_min}")
    # print(f"后视镜镜柄厚度（水切式）: {z_distance_difference}")

    # 找到最大点和最小点
    max_point = projected_points[projected_points[:, 1] == z_max]
    min_point = projected_points[projected_points[:, 1] == z_min]

    # # 可视化投影点
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, color="black", label="Projected Points")
    # ax.scatter(max_point[:, 0], max_point[:, 1], color="red", s=10, label="Max Point")
    # ax.scatter(min_point[:, 0], min_point[:, 1], color="blue", s=10, label="Min Point")

    # # 设置图表标签和标题
    # ax.set_xlabel("X")
    # ax.set_ylabel("Z")
    # ax.set_title("Projected Points on XZ Plane")
    # ax.axis("equal")
    # ax.grid(True)
    # ax.legend()

    # # 显示图表
    # plt.show()

    return z_distance_difference


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)