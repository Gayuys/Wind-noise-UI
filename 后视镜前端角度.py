import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import warnings

# ===================== 全局设置：解决中文字体警告 + 屏蔽无关警告 =====================
# 屏蔽字体查找警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# 设置Matplotlib中文显示（兼容Windows/macOS/Linux，优先系统中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'


def curve_angle(points, axis):
    """
    定义曲线表征，将弧线分为3段，分别计算每一段弧线上的最大最小值
    points: 需要拟合的点
    axis: 'x' or 'y'分别对应以横坐标或纵坐标将弧线分为3段
    返回: 每一段弧线上的最大、最小、平均角度
    """

    def poly_func(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    # 求多项式函数的导数
    def poly_derivative(x, a, b, c, d, e):
        return 4 * a * x ** 3 + 3 * b * x ** 2 + 2 * c * x + d

    # 计算角度函数
    def calculate_angle(slope):
        return math.degrees(math.atan(slope))

    x = points[:, 1]
    y = points[:, 0]
    slope_results = []

    # 空值判断：点数量不足无法拟合4次多项式（至少5个点）
    if len(points) < 5:
        # print("【警告】拟合点数量不足，无法进行4次多项式拟合")
        return slope_results

    # 曲线拟合
    popt, _ = curve_fit(poly_func, x, y)
    min_point = min(x)
    max_point = max(x)

    # 生成拟合曲线的点
    x_fit = np.linspace(min_point, max_point, 200)
    y_fit = poly_func(x_fit, *popt)

    # 将拟合曲线分成3段
    x_segments = np.array_split(x_fit, 3)

    # # 绘制画布（仅创建一次，修复冗余画布问题）
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.scatter(points[:, 0], points[:, 1], color='red', s=10, label='后视镜前端轮廓点')
    # ax.plot(y_fit, x_fit, color='blue', linewidth=2, label='4次多项式拟合曲线')

    # # 颜色列表，用于区分不同段
    # colors = ['orange', 'purple', 'brown']

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
        min_angle = 90 - calculate_angle(max_slope)
        max_angle = 90 - calculate_angle(min_slope)
        mean_angle = 90 - calculate_angle(mean_slope)

        slope_results.append({
            'max_angle': max_angle,
            'min_angle': min_angle,
            'mean_angle': mean_angle,
            'x_range': (y_segment[0], y_segment[-1])
        })

        # # 绘制分段曲线，使用不同颜色
        # ax.plot(y_segment, x_segment, color=colors[i], linewidth=3, label=f'第{i + 1}段')

        # 找到每段的中间点
        mid_idx = len(y_segment) // 2
        mid_x = y_segment[mid_idx]
        mid_y = x_segment[mid_idx]

        # 设置标注的垂直间隔
        vertical_offset = 20
        # 计算每个标注的y位置
        y_positions = [mid_y - vertical_offset, mid_y - vertical_offset * 2, mid_y - vertical_offset * 3]
        # 竖直排列标注
        annotations = [
            {'text': f'Mean: {mean_angle:.2f}°', 'y': y_positions[0], 'color': 'black'},
            {'text': f'Max: {max_angle:.2f}°', 'y': y_positions[1], 'color': 'red'},
            {'text': f'Min: {min_angle:.2f}°', 'y': y_positions[2], 'color': 'blue'}
        ]
        # 添加标注
        # for annotation in annotations:
        #     ax.text(mid_x, annotation['y'], annotation['text'], fontsize=9,
        #             color=annotation['color'], bbox=dict(facecolor='white', alpha=0.7),
        #             ha='center', va='center')

    # #输出拟合参数和分段角度
    # print("=" * 70)
    # print("【后视镜前端】曲线拟合参数(4次多项式：y = a*x^4 + b*x^3 + c*x^2 + d*x + e)")
    # print(f"a = {popt[0]:.6f}, b = {popt[1]:.6f}, c = {popt[2]:.6f}, d = {popt[3]:.6f}, e = {popt[4]:.6f}")
    # print("\n【后视镜前端】各段角度分析结果:")
    # for i, result in enumerate(slope_results, 1):
    #     print(f"第{i}段 (X范围：{result['x_range'][0]:.2f} ~ {result['x_range'][1]:.2f}):")
    #     print(
    #         f"  最大角度: {result['max_angle']:.2f}° | 最小角度: {result['min_angle']:.2f}° | 平均角度: {result['mean_angle']:.2f}°")
    #     print()

    # #图表样式设置
    # ax.set_xlabel("X坐标 (mm)", fontsize=12)
    # ax.set_ylabel("Y坐标 (mm)", fontsize=12)
    # ax.set_title("后视镜前端轮廓 - 4次多项式拟合与分段角度分析", fontsize=15, fontweight='bold')
    # ax.legend(fontsize=10, loc='best')
    # ax.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    return slope_results


# ===================== 主程序：后视镜前端角度核心计算逻辑 =============
def calculate(stl_path):
    front_mirror_angle = None
    # 读取STL文件
    mesh = trimesh.load_mesh(stl_path)
    vertices = mesh.vertices
    # print("STL文件加载成功，顶点数量：", len(vertices))
    # print("-" * 70)

    # 1. 顶点筛选（原逻辑保留）
    z_min = np.min(vertices[:, 2])
    filtered_vertices = vertices[vertices[:, 2] > (z_min + 300)]
    point1 = filtered_vertices[np.argmin(filtered_vertices[:, 1])]
    filtered_vertices2 = vertices[(vertices[:, 2] > (point1[2] - 20)) & (vertices[:, 1] < (point1[1] + 240))]
    projected_points = filtered_vertices2[:, :2]

    # 2. 计算投影面极值点和差值
    x_min, x_max = np.min(projected_points[:, 0]), np.max(projected_points[:, 0])
    y_min, y_max = np.min(projected_points[:, 1]), np.max(projected_points[:, 1])
    point_x_min = projected_points[projected_points[:, 0] == x_min][0]
    point_x_max = projected_points[projected_points[:, 0] == x_max][0]
    point_y_min = projected_points[projected_points[:, 1] == y_min][0]
    point_y_max = projected_points[projected_points[:, 1] == y_max][0]
    abs_x_diff, abs_y_diff = abs(x_max - x_min), abs(y_max - y_min)

    # 输出投影面基本信息
    # print("【投影面基本信息】")
    # print(f"X轴最值差值绝对值: {abs_x_diff:.2f}mm | Y轴最值差值绝对值: {abs_y_diff:.2f}mm")
    # print(
    #     f"X轴最小值点: ({point_x_min[0]:.2f}, {point_x_min[1]:.2f}) | X轴最大值点: ({point_x_max[0]:.2f}, {point_x_max[1]:.2f})")
    # print(
    #     f"Y轴最小值点: ({point_y_min[0]:.2f}, {point_y_min[1]:.2f}) | Y轴最大值点: ({point_y_max[0]:.2f}, {point_y_max[1]:.2f})")
    # print("-" * 70)

    # 3. 绘制投影面点与极值点
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, color="black", label='投影点')
    # ax.scatter([point_x_min[0], point_x_max[0], point_y_min[0], point_y_max[0]],
    #            [point_x_min[1], point_x_max[1], point_y_min[1], point_y_max[1]],
    #            c='red', s=30, marker='*', label='极值点')
    # ax.set_xlabel("X坐标 (mm)")
    # ax.set_ylabel("Y坐标 (mm)")
    # ax.set_title("STL模型XY平面投影点与极值点", fontsize=14)
    # ax.axis("equal")
    # ax.grid(True, alpha=0.7)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # 4. 筛选后视镜前端点区域
    points_front_mirror = projected_points[
        (projected_points[:, 1] >= point_y_min[1]) & (projected_points[:, 1] <= point_x_min[1])
        ]

    # ===================== 后视镜前端角度计算 =====================
    point_front = []
    if len(points_front_mirror) > 0:
        length1 = max(points_front_mirror[:, 1]) - min(points_front_mirror[:, 1])
        length1 = int(length1)
        for i in range((length1 // 10) + 1):
            points = points_front_mirror[
                (points_front_mirror[:, 1] >= point_y_min[1] + i * 10) &
                (points_front_mirror[:, 1] <= point_y_min[1] + (i + 1) * 10)
                ]
            # 空值判断：当前分段有点才提取
            if len(points) > 0:
                point = points[points[:, 0] == min(points[:, 0])][0]
                point_front.append(point)
        point_front = np.array(point_front)

        # 调用拟合函数，获取分段角度
        slope_results = curve_angle(point_front, 'y')
        # 计算前端角度：三段中最大角度的全局最大值
        if slope_results:
            all_max_angles = [res['max_angle'] for res in slope_results]
            front_mirror_angle = max(all_max_angles)
            max_angle_seg = all_max_angles.index(front_mirror_angle) + 1
            # print("=" * 70)
            # print(f"后视镜前端角度：{front_mirror_angle:.2f}°")
            # print(f"该角度所属分段：第{max_angle_seg}段")
            # print("=" * 70)
        else:
            # print("【错误】无法计算后视镜前端角度：拟合失败")
            pass  # 空操作，保持语法合规
    else:
        # print("【错误】未筛选到后视镜前端点")
        pass  # 空操作，保持语法合规

    # # 绘制后视镜前端点图
    # if 'point_front' in locals() and len(point_front) > 0:
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.scatter(point_front[:, 0], point_front[:, 1], s=5, color="black", label="后视镜前端点")
    #     ax.set_xlabel("X坐标 (mm)")
    #     ax.set_ylabel("Y坐标 (mm)")
    #     ax.set_title("后视镜前端特征点", fontsize=14)
    #     ax.axis("equal")
    #     ax.grid(True, alpha=0.7)
    #     ax.legend()
    #     plt.tight_layout()
    #     plt.show()

    return front_mirror_angle


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"E:\一汽风噪\点云文件\AITO M9 Outer Surface.stl")
    print(rear_windshield_angle)