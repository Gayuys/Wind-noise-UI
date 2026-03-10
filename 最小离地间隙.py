import numpy as np
import trimesh
import matplotlib.pyplot as plt

# 设置 Matplotlib 显示中文
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def calculate(stl_path):
    Z = None
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标
    vertices = None
    if mesh is not None:
        vertices = mesh.vertices

    # 找到z轴最小值的点
    z_min_1 = None
    if vertices is not None:
        z_min_1 = vertices[np.argmin(vertices[:, 2])]

    # 找到y轴最大、小值
    y_min = None
    y_max = None
    if vertices is not None:
        y_min = np.min(vertices[:, 1])
        y_max = np.max(vertices[:, 1])

    # 找到y轴的中值
    y_mid = None
    if y_min is not None and y_max is not None:
        y_mid = (y_max + y_min) / 2

    # 筛选满足条件的点
    filtered_vertices = None
    if vertices is not None and y_mid is not None:
        filtered_vertices = vertices[(vertices[:, 1] < (y_mid + 500)) & (vertices[:, 1] > (y_mid - 500))]

    # 在这些筛选后的点中找z轴最小值对应的点
    z_min_2 = None
    if filtered_vertices is not None:
        z_min_2 = filtered_vertices[np.argmin(filtered_vertices[:, 2])]

    # 计算最小离地间隙
    Z = None
    if z_min_1 is not None and z_min_2 is not None:
        Z = abs(z_min_1[2] - z_min_2[2])
        # print(f"最小离地间隙: {Z:.2f}")
    else:
        # print("无法计算最小离地间隙，缺少必要的点数据")
        pass  # 空操作，保持语法合规

    # # 可视化结果
    # if vertices is not None and z_min_1 is not None and z_min_2 is not None:
    #     point_size = 2  # 设置点的大小
    #     plt.figure()
    #     plt.plot(vertices[:, 0], vertices[:, 2], 'ro', markersize=point_size)  # 绘制红色点，设置点的大小
    #     plt.plot(z_min_1[0], z_min_1[2], 'go', markersize=10, label='z_min_1')
    #     plt.plot(z_min_2[0], z_min_2[2], 'bo', markersize=10, label='z_min_2')
    #     plt.title('Intersection Points with XZ Plane')
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Z-axis')
    #     plt.axis('equal')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    # else:
    #     print("缺少可视化所需的关键数据，无法绘制图形")

    return Z


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)