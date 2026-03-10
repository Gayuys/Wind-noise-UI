import trimesh
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def calculate(stl_path):
    angle_horizontal = None
    # 读取 STL 文件
    mesh = trimesh.load_mesh(stl_path)

    # 提取所有顶点并去重
    vertices = mesh.vertices  # 获取顶点
    unique_vertices = np.unique(vertices, axis=0)  # 去除重复点

    # 筛选 z 轴大于 300 的点
    valid_vertices = unique_vertices[unique_vertices[:, 2] > 300]

    # 找到 y 轴方向的最大值点
    if len(valid_vertices) > 0:
        y_max_point = valid_vertices[np.argmax(valid_vertices[:, 1])]  # y 最大值点
        projected_y_max_point = y_max_point[[0, 2]]  # 投影到 XZ 平面
    else:
        projected_y_max_point = None  # 如果没有满足条件的点

    # 设定 y 坐标范围，提取投影点
    y_min, y_max = -10, 10  # 范围为 [-10mm, 10mm]
    xz_points = unique_vertices[(unique_vertices[:, 1] >= y_min) & (unique_vertices[:, 1] <= y_max)]
    projected_points = xz_points[:, [0, 2]]  # 投影到 XZ 平面

    # 检查是否有投影点并处理
    if projected_points.shape[0] == 0:
        # print("没有找到符合条件的点")
        pass  # 空操作，保持语法合规
    else:
        # 找到投影平面竖直方向 (Z 轴) 的最大值点
        z_max_index = np.argmax(projected_points[:, 1])  # 找到 Z 轴最大值的索引
        z_max_point = projected_points[z_max_index]     # 获取该点的坐标

        # 根据条件筛选符合范围的点
        if projected_y_max_point is not None:
            point1_z = projected_y_max_point[1]  # 点1的 Z 坐标
            point2_z = z_max_point[1]            # 点2的 Z 坐标

            # 定义范围
            z_min_1 = point1_z + 20  # 点1的 Z 坐标向下移动 50mm
            z_min_2 = point2_z - 150 # 点2的 Z 坐标向下移动 100mm

            # 确定 Z 坐标范围为两者之间
            z_range_min = min(z_min_1, z_min_2)  # 下边界
            z_range_max = max(z_min_1, z_min_2)  # 上边界

            # 计算 X 轴范围的一半
            x_min = np.min(projected_points[:, 0])  # X 最小值
            x_max = np.max(projected_points[:, 0])  # X 最大值
            x_half = (x_min + x_max) / 2            # X 范围的一半

            # 筛选符合范围的点
            filtered_points = projected_points[
                (projected_points[:, 1] >= z_range_min) &  # Z 坐标 >= 下边界
                (projected_points[:, 1] <= z_range_max) &  # Z 坐标 <= 上边界
                (projected_points[:, 0] < x_half)         # X 坐标小于 X 范围的一半
            ]
        else:
            # print("没有有效的projected_y_max_point，无法进行范围筛选")
            filtered_points = np.array([])

        # 检查是否有符合条件的点
        if filtered_points.shape[0] == 0:
            # print("没有找到符合条件的点")
            pass  # 空操作，保持语法合规
        else:
            # 将 X 坐标从最小值到最大值分成 20 段
            x_min = np.min(filtered_points[:, 0])  # X 最小值
            x_max = np.max(filtered_points[:, 0])  # X 最大值
            x_bins = np.linspace(x_min, x_max, 21)  # 将 X 轴分成 20 段

            selected_points = []

            # 遍历每一段，选择其中 Z 坐标最小的点
            for i in range(len(x_bins) - 1):
                # 选出当前 X 区间的点
                x_range_min = x_bins[i]
                x_range_max = x_bins[i + 1]
                bin_points = filtered_points[
                    (filtered_points[:, 0] >= x_range_min) & (filtered_points[:, 0] < x_range_max)
                ]

                if bin_points.shape[0] > 0:
                    # 选择 Z 坐标最小的点
                    min_z_point = bin_points[np.argmin(bin_points[:, 1])]
                    selected_points.append(min_z_point)

            # 将筛选的点转为 NumPy 数组
            selected_points = np.array(selected_points)

            # # 可视化所有投影点和选中点
            # plt.figure(figsize=(8, 6))
            # plt.scatter(projected_points[:, 0], projected_points[:, 1], s=2, c='blue', label="投影点")
            # if selected_points.shape[0] > 0:
            #     plt.scatter(selected_points[:, 0], selected_points[:, 1], s=50, c='orange', label="选中范围点")
            # plt.title("投影点和选中范围点")
            # plt.xlabel("X")
            # plt.ylabel("Z")
            # plt.grid(True)
            # plt.legend()
            # plt.axis("equal")
            # plt.show()

            # 如果筛选出的点不为空，进行线性拟合并计算角度
            if selected_points.shape[0] > 0:
                # 进行线性拟合
                x_data = selected_points[:, 0]
                z_data = selected_points[:, 1]
                slope, intercept = np.polyfit(x_data, z_data, 1)  # y = slope * x + intercept

                # 计算拟合直线与水平方向和竖直方向的角度
                angle_horizontal = np.arctan(slope) * (180 / np.pi)  # 与X轴夹角
                angle_vertical = np.arctan(1 / slope) * (180 / np.pi)  # 与Z轴夹角

                # 输出角度
                # print(f"拟合直线与水平方向（X 轴）的夹角：{angle_horizontal:.1f}°")
                # print(f"拟合直线与竖直方向（Z 轴）的夹角：{angle_vertical:.1f}°")
                # print(f"前风挡角度：{angle_horizontal:.1f}°")

                # # 可视化拟合结果
                # plt.figure(figsize=(8, 6))
                # plt.scatter(x_data, z_data, color='orange', label="拟合点")
                # plt.plot(x_data, slope * x_data + intercept, color='red', label="拟合直线", linewidth=2)
                # plt.title("拟合点和拟合直线")
                # plt.xlabel("X")
                # plt.ylabel("Z")
                # plt.legend()
                # plt.grid(True)
                # plt.show()
            else:
                # print("没有找到符合条件的选中点")
                pass  # 空操作，保持语法合规

    return angle_horizontal

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)