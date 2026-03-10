import numpy as np
import trimesh
import matplotlib.pyplot as plt

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def calculate(stl_path):
    angle_deg = None
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

    # 在这些筛选后的点中找y轴最小值对应的点（点1）
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 筛选z轴大于（点1 z轴值-50mm）的点
    filtered_vertices2 = vertices[
        (vertices[:, 2] > (point1[2] - 20)) &
        (vertices[:, 1] < (point1[1] + 250))
        ]
    if len(filtered_vertices2) == 0:
        raise ValueError("第二次筛选未找到符合条件的点")

    # 投影到xy平面（保留x和y坐标）
    projected_points = filtered_vertices2[:, :2]

    # 找出投影到XY平面的各向极值
    x_max = np.max(projected_points[:, 0])
    y_min = np.min(projected_points[:, 1])

    # 找到最大值和最小值对应的点的坐标
    point_x_max = projected_points[projected_points[:, 0] == x_max][0]
    point_y_min = projected_points[projected_points[:, 1] == y_min][0]

    # 计算并单独输出【直线与水平方向的夹角】
    # 两点的坐标分别是point_x_max和point_y_min
    line_points = np.array([point_x_max, point_y_min])
    # 计算直线与水平方向的夹角
    dx = point_x_max[0] - point_y_min[0]
    dy = point_x_max[1] - point_y_min[1]

    if dx == 0 and dy == 0:
        raise ValueError("两点坐标相同，无法计算夹角")

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    # print("=" * 50)
    # print(f"后视镜末端平行段角度: {angle_deg:.2f} 度")
    # print("=" * 50)

    # # 可视化：保留投影点、极值点和连接线（原样式）
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, color="black", label="投影点")
    # ax.scatter(point_x_max[0], point_x_max[1], c='red', s=50, label='X轴最大值点')
    # ax.scatter(point_y_min[0], point_y_min[1], c='blue', s=50, label='Y轴最小值点')
    # plt.plot(line_points[:, 0], line_points[:, 1], 'r-', linewidth=2,
    #         label=f'连接线 (夹角: {angle_deg:.2f}°)')
    # ax.set_xlabel("X (mm)", fontsize=12)
    # ax.set_ylabel("Y (mm)", fontsize=12)
    # ax.set_title("投影点与连接线夹角可视化", fontsize=14)
    # ax.axis("equal")
    # ax.grid(True, alpha=0.7)
    # ax.legend(fontsize=10)
    # plt.tight_layout()
    # plt.show()

    return angle_deg


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)