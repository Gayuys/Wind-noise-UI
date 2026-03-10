import numpy as np
import trimesh
import matplotlib.pyplot as plt
import math
import warnings

# ===================== 全局设置：解决中文字体警告 + 屏蔽无关警告 =====================
# 屏蔽字体查找警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# 设置Matplotlib中文显示（兼容Windows/macOS/Linux，优先系统中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'


# ===================== 主程序：后视镜后端角度核心计算逻辑 =====================
def calculate(stl_path):
    theta_x_deg = None
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

    # # 3. 绘制投影面点与极值点
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

    # 4. 筛选后视镜后端点区域
    points_back_mirror = projected_points[
        (projected_points[:, 1] >= point_x_max[1]) & (projected_points[:, 1] <= point_x_min[1])
        ]

    # ===================== 后视镜后端角度计算 =====================
    point_back = []
    if len(points_back_mirror) > 0:
        length2 = max(points_back_mirror[:, 1]) - min(points_back_mirror[:, 1])
        length2 = int(length2)
        for a in range((length2 // 10) + 1):
            points = points_back_mirror[
                (points_back_mirror[:, 1] >= point_x_max[1] + a * 10) &
                (points_back_mirror[:, 1] <= point_x_max[1] + (a + 1) * 10)
                ]
            # 空值判断：当前分段有点才提取
            if len(points) > 0:
                point = points[points[:, 0] == max(points[:, 0])][0]
                point_back.append(point)
            else:
                pass  # 补充空分段的pass语句，避免语法隐患

        point_back = np.array(point_back)

        # 计算后端点的直线夹角
        if len(point_back) >= 2:
            p1 = point_back[point_back[:, 1] == max(point_back[:, 1])][0]
            p2 = point_back[point_back[:, 1] == min(point_back[:, 1])][0]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            # 除零判断：避免垂直X轴/Y轴时报错
            if dx == 0:
                theta_x_deg = 90.0  # 垂直X轴，与X轴夹角90度
                theta_y_deg = 0.0  # 与Y轴夹角0度
            else:
                m = dy / dx
                theta_x_rad = math.atan(m)
                theta_x_deg = abs(math.degrees(theta_x_rad))
                # 计算与y轴的夹角 (theta_y) 保留原逻辑，增加m=0的除零判断
                theta_y_rad = math.atan(1 / m) if m != 0 else math.pi / 2
                theta_y_deg = abs(math.degrees(theta_y_rad))

            # 保留你需要的两个原始打印语句（已注释）
            # print('与 x 轴的夹角为', theta_x_deg, '度')
            # print('与 y 轴的夹角为', theta_y_deg, '度')
            # # 输出后端角度结果
            # print("后视镜后端角度：", f"{theta_x_deg:.2f}°")
            # print("=" * 70)
        else:
            # print("【错误】无法计算后视镜后端角度：后端点数量不足")
            pass  # 空操作，保持语法合规
    else:
        # print("【错误】未筛选到后视镜后端点")
        pass  # 空操作，保持语法合规

    # # 绘制后视镜后端点图
    # if 'point_back' in locals() and len(point_back) > 0:
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.scatter(point_back[:, 0], point_back[:, 1], s=5, color="red", label="后视镜后端点")
    #     ax.set_xlabel("X坐标 (mm)")
    #     ax.set_ylabel("Y坐标 (mm)")
    #     ax.set_title("后视镜后端特征点", fontsize=14)
    #     ax.axis("equal")
    #     ax.grid(True, alpha=0.7)
    #     ax.legend()
    #     plt.tight_layout()
    #     plt.show()
    return theta_x_deg

if __name__ == "__main__":
    rear_windshield_angle = calculate(r"F:\一汽风噪\点云文件\XpengG9 Outer Surface.stl")
    print(rear_windshield_angle)