import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.font_manager as fm

# 设置中文字体为黑体，英文字体为Times New Roman
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def calculate(stl_path):
    tri_cover_min_angle = None
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)

    # 获取顶点坐标
    vertices = mesh.vertices

    # 找到z轴最小值
    z_min = np.min(vertices[:, 2])

    # 筛选满足条件的点：z轴大于z轴最小值300mm的点
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]

    # 在这些筛选后的点中找y轴最小值对应的点（点1）
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]

    # 筛选z轴大于（点1 z轴值-50mm）的点
    filtered_vertices2 = vertices[(vertices[:, 2] > (point1[2] - 20)) & (vertices[:, 1] < (point1[1] + 240))]

    # 投影到xy平面（保留x和y坐标，z坐标设为0）
    projected_points = filtered_vertices2[:, :2]

    # 找出投影到XY平面的X向的最大值和最小值，以及Y向的最大值和最小值
    x_min = np.min(projected_points[:, 0])
    x_max = np.max(projected_points[:, 0])
    y_min = np.min(projected_points[:, 1])
    y_max = np.max(projected_points[:, 1])

    # 找到最大值和最小值对应的点的坐标
    point_x_min = projected_points[projected_points[:, 0] == x_min][0]
    point_x_max = projected_points[projected_points[:, 0] == x_max][0]
    point_y_min = projected_points[projected_points[:, 1] == y_min][0]
    point_y_max = projected_points[projected_points[:, 1] == y_max][0]

    # 筛选需要拟合部分的点
    projected_points1 = projected_points[(projected_points[:, 1] > (y_max - 20)) & (projected_points[:, 0] > point_y_max[0])]
    x_min1 = np.min(projected_points1[:, 0])
    x_max1 = np.max(projected_points1[:, 0])

    # 定义多项式拟合函数（4次多项式）
    def poly_func(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    # 求多项式函数的导数
    def poly_derivative(x, a, b, c, d, e):
        return 4*a*x**3 + 3*b*x**2 + 2*c*x + d

    # 计算角度函数
    def calculate_angle(slope):
        return math.degrees(math.atan(slope))

    # 将x方向分成30段
    x_segments = np.linspace(x_min1, x_max1, 31)

    # 存储每段的最大y值点
    segment_max_points = []

    for i in range(30):
        # 筛选当前段的点
        segment_points = projected_points1[
            (projected_points1[:, 0] >= x_segments[i]) &
            (projected_points1[:, 0] < x_segments[i + 1])
            ]

        # 如果该段有点，找最大y值的点
        if len(segment_points) > 0:
            max_y_point = segment_points[np.argmax(segment_points[:, 1])]
            segment_max_points.append(max_y_point)

    # 转换为numpy数组
    segment_max_points = np.array(segment_max_points)

    # 曲线拟合
    popt, _ = curve_fit(poly_func, segment_max_points[:, 0], segment_max_points[:, 1])

    # 生成拟合曲线的点
    x_fit = np.linspace(x_min1, x_max1, 200)
    y_fit = poly_func(x_fit, *popt)

    # 将拟合曲线分成3段
    x_segments = np.array_split(x_fit, 3)

    # 存储每段的斜率信息
    slope_results = []

    # # 创建图形和轴
    # fig, ax = plt.subplots(figsize=(12, 8))

    # # 绘制原始投影点
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], color='lightgray', s=1, label='Projected Points')
    # ax.scatter(projected_points1[:, 0], projected_points1[:, 1], color='green', s=1, label='Selected Points for Fitting')

    # # 绘制极值点
    # ax.scatter(point_x_min[0], point_x_min[1], c='red', s=50, marker='*', label='X Min')
    # ax.scatter(point_x_max[0], point_x_max[1], c='red', s=50, marker='*', label='X Max')
    # ax.scatter(point_y_min[0], point_y_min[1], c='red', s=50, marker='*', label='Y Min')
    # ax.scatter(point_y_max[0], point_y_max[1], c='red', s=50, marker='*', label='Y Max')

    # # 绘制分段点和拟合曲线
    # ax.scatter(segment_max_points[:, 0], segment_max_points[:, 1], color='red', s=10, label='Segment Max Points')
    # ax.plot(x_fit, y_fit, color='blue', linewidth=2, label='Fitting Curve')

    # # 颜色列表，用于区分不同段
    colors = ['orange', 'purple', 'brown']

    for i, x_segment in enumerate(x_segments):
        # 获取对应的y值
        y_segment = poly_func(x_segment, *popt)

        # 计算该段的斜率
        slopes = np.abs(poly_derivative(x_segment, *popt))

        # 计算斜率统计信息
        max_slope = np.max(slopes)
        min_slope = np.min(slopes)
        mean_slope = np.mean(slopes)

        # 转换为角度
        max_angle = calculate_angle(max_slope)
        min_angle = calculate_angle(min_slope)
        mean_angle = calculate_angle(mean_slope)

        slope_results.append({
            'max_angle': max_angle,
            'min_angle': min_angle,
            'mean_angle': mean_angle,
            'x_range': (x_segment[0], x_segment[-1]),
            'color': colors[i],  # 新增：记录分段颜色，用于后续标注
            'index': i+1         # 新增：记录分段序号
        })

        # # 绘制分段曲线，使用不同颜色
        # ax.plot(x_segment, y_segment, color=colors[i], linewidth=3,
        #         label=f'第{i + 1}段')

        # 找到每段的中间点
        mid_idx = len(x_segment) // 2
        mid_x = x_segment[mid_idx]
        mid_y = y_segment[mid_idx]

        # 设置标注的垂直间隔
        vertical_offset = 20  # 调整这个值来控制标注之间的距离

        # 计算每个标注的y位置
        y_positions = [
            mid_y - vertical_offset,  # 平均角度位置
            mid_y - vertical_offset * 2,  # 最大角度位置
            mid_y - vertical_offset * 3  # 最小角度位置
        ]

        # 竖直排列标注
        annotations = [
            {'text': f'Mean: {mean_angle:.2f}°', 'y': y_positions[0], 'color': 'black'},
            {'text': f'Max: {max_angle:.2f}°', 'y': y_positions[1], 'color': 'red'},
            {'text': f'Min: {min_angle:.2f}°', 'y': y_positions[2], 'color': 'blue'}
        ]

        # # 添加标注
        # for annotation in annotations:
        #     ax.text(mid_x, annotation['y'],
        #             annotation['text'],
        #             fontsize=9,
        #             color=annotation['color'],
        #             bbox=dict(facecolor='white', alpha=0.7),
        #             horizontalalignment='center',
        #             verticalalignment='center')

    # ===================== 新增核心逻辑：计算三角饰盖Z向最小夹角 =====================
    # 遍历slope_results，找到平均角度最小的分段
    min_mean_angle_item = min(slope_results, key=lambda x: x['mean_angle'])
    # 定义三角饰盖Z向最小夹角为该最小平均角度
    tri_cover_min_angle = min_mean_angle_item['mean_angle']
    tri_cover_min_segment = min_mean_angle_item['index']
    tri_cover_x_range = min_mean_angle_item['x_range']
    tri_cover_color = min_mean_angle_item['color']

    # # ===================== 可视化标注：三角饰盖Z向最小夹角 =====================
    # # 找到最小夹角分段的中间点，用于标注
    # min_segment_x = np.linspace(tri_cover_x_range[0], tri_cover_x_range[1], 100)
    # min_segment_y = poly_func(min_segment_x, *popt)
    # min_mid_idx = len(min_segment_x) // 2
    # min_mid_x = min_segment_x[min_mid_idx]
    # min_mid_y = min_segment_y[min_mid_idx]
    # # 添加醒目标注
    # ax.text(min_mid_x, min_mid_y + 30,  # 向上偏移30，避免遮挡
    #         f'三角饰盖Z向最小夹角\n{tri_cover_min_angle:.2f}°(第{tri_cover_min_segment}段)',
    #         fontsize=11,
    #         color=tri_cover_color,
    #         fontweight='bold',
    #         bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black'),
    #         horizontalalignment='center',
    #         verticalalignment='center')
    # # 加粗最小夹角分段的曲线，突出显示
    # ax.plot(min_segment_x, min_segment_y, color=tri_cover_color, linewidth=5, alpha=0.8,
    #         label=f'最小夹角段(第{tri_cover_min_segment}段)')

    # ===================== 控制台输出：保留原输出 + 新增三角饰盖Z向最小夹角 =====================
    # print("="*50)
    # print("曲线拟合参数:")
    # print(f"a = {popt[0]:.6f}, b = {popt[1]:.6f}, c = {popt[2]:.6f}, d = {popt[3]:.6f}, e = {popt[4]:.6f}")
    # print("\n各段角度分析结果:")
    # for i, result in enumerate(slope_results, 1):
    #     print(f"第{i}段 ({result['x_range'][0]:.2f} to {result['x_range'][1]:.2f}):")
    #     print(f"  最大角度: {result['max_angle']:.2f}°")
    #     print(f"  最小角度: {result['min_angle']:.2f}°")
    #     print(f"  平均角度: {result['mean_angle']:.2f}°")
    #     print()

    # 打印三角饰盖Z向最小夹角
    # print("="*50)
    # print(f"三角饰盖Z向最小夹角：{tri_cover_min_angle:.2f}°")
    # print("="*50)

    # # 图表基础设置
    # ax.set_xlabel("X坐标 (mm)", fontsize=12)
    # ax.set_ylabel("Y坐标 (mm)", fontsize=12)
    # ax.set_title("STL模型投影分析与曲线拟合（含三角饰盖Z向最小夹角）", fontsize=15, fontweight='bold')
    # ax.legend(fontsize=10, loc='best')
    # ax.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    # 原注释的打印代码可按需取消注释
    # # 打印最大值与最小值的坐标
    # print(f"X轴最小值坐标: {point_x_min}")
    # print(f"X轴最大值坐标: {point_x_max}")
    # print(f"Y轴最小值坐标: {point_y_min}")
    # print(f"Y轴最大值坐标: {point_y_max}")
    #
    # fig, ax = plt.subplots()
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, color="black")
    # ax.scatter(projected_points1[:, 0], projected_points1[:, 1], s=1, color="green")
    # ax.scatter(point_x_min[0], point_x_min[1], c='red', s=10, label='X轴最小值点')
    # ax.scatter(point_x_max[0], point_x_max[1], c='red', s=10, label='X轴最大值点')
    # ax.scatter(point_y_min[0], point_y_min[1], c='red', s=10, label='Y轴最小值点')
    # ax.scatter(point_y_max[0], point_y_max[1], c='red', s=10, label='Y轴最大值点')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_title("Projected Points on XZ Plane")
    # ax.axis("equal")
    # ax.grid(True)
    # plt.show()
    return tri_cover_min_angle

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"E:\一汽风噪\点云文件\AITO M9 Outer Surface.stl")
    print(rear_windshield_angle)