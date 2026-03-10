import cv2
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 迎风面积
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


def calculate(stl_path):
    true_area = None
    # 加载网格文件
    mesh = trimesh.load_mesh(stl_path)

    points = mesh.vertices
    # 取两位小数
    points_1 = np.round(points, 2)
    maxlength = max(points[:, 0])
    minlength = min(points[:, 0])

    # 投影到YZ平面
    yz_projection = project_3d_to_2d(points_1, 'yz')
    if len(yz_projection) == 0:
        raise ValueError("投影后的点集为空")

    # 计算边界和矩形参数
    max_y = max(yz_projection[:, 0])
    min_y = min(yz_projection[:, 0])
    max_z = max(yz_projection[:, 1])
    min_z = min(yz_projection[:, 1])

    x = max_y + 100  # 矩形起点x（对应YZ投影的y轴）
    y = min_z  # 矩形起点y（对应YZ投影的z轴）
    width = max_y - min_y  # 矩形宽度
    height = max_z - min_z  # 矩形高度
    rect = patches.Rectangle((x, y), width, height, linewidth=0, edgecolor='none', facecolor='blue')
    rect_area = width * height  # 矩形面积

    #绘制散点图并保存
    plt.axis('off')
    plt.scatter(yz_projection[:, 0], yz_projection[:, 1], s=5, c='blue')
    plt.gca().add_patch(rect)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 修复路径转义问题（使用正斜杠）
    plt.savefig('./output2.jpg', dpi=300)
    #plt.show()

    # 读取图像（修复路径转义问题）
    img = cv2.imread('./output2.jpg')
    if img is None:
        raise FileNotFoundError("无法读取保存的图像文件，请检查路径是否正确")

    # 转换为灰度图和二值图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("未找到任何轮廓")

    # 计算每个轮廓的像素面积
    Pixels_area = []
    for contour in contours:
        area = cv2.contourArea(contour)
        Pixels_area.append(area)

    Pixels_area = np.array(Pixels_area)
    if len(Pixels_area) < 3:
        raise IndexError("轮廓数量不足，无法计算实际面积")

    # 计算实际面积（单位转换为平方米）
    true_area = rect_area * Pixels_area[2] / Pixels_area[1]
    #print("迎风面积为：", true_area * 0.000001, "m^2")

    # # 绘制轮廓并保存
    # i = 0
    # for contour in contours:
    #     cv2.drawContours(img, [contour], -1, (0, 0, 255), 3)
    #     cv2.imwrite(f"contour_{i}.png", img)
    #     i += 1
    # cv2.destroyAllWindows()

    return true_area * 0.000001


if __name__ == "__main__":
    rear_windshield_angle = calculate(r"E:\一汽风噪\点云文件\AITO M9 Outer Surface.stl")
    print(rear_windshield_angle)