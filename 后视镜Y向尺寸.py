import numpy as np
import trimesh
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def calculate(stl_path):
    rearview_mirror_y_size = None
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
        (vertices[:, 1] < (point1[1] + 230))
        ]
    if len(filtered_vertices2) == 0:
        raise ValueError("第二次筛选未找到符合条件的点")

    # 投影到xy平面（保留x和y坐标，z坐标设为0）
    projected_points = filtered_vertices2[:, :2]

    # 找出投影到XY平面的Y向的最大值和最小值
    y_min = np.min(projected_points[:, 1])
    y_max = np.max(projected_points[:, 1])

    # 找到Y向极值对应的点的坐标
    point_y_min = projected_points[projected_points[:, 1] == y_min][0]
    point_y_max = projected_points[projected_points[:, 1] == y_max][0]

    # 计算Y向差值的绝对值（后视镜Y向尺寸）
    rearview_mirror_y_size = abs(y_max - y_min)

    # 打印后视镜Y向尺寸结果
    #print(f"后视镜Y向尺寸: {rearview_mirror_y_size:.2f}")

    # # 可视化部分（保留原逻辑，仅显示Y向极值点）
    # fig, ax = plt.subplots()
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, color="black")
    # ax.scatter(point_y_min[0], point_y_min[1], c='red', s=10, label='Y轴最小值点')
    # ax.scatter(point_y_max[0], point_y_max[1], c='red', s=10, label='Y轴最大值点')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_title("Projected Points on XY Plane (Y向尺寸)")
    # ax.axis("equal")
    # ax.grid(True)
    # plt.legend()
    # plt.show()

    return rearview_mirror_y_size


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)