import sys
import os
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QWidget, QMessageBox,QFrame, QStyleOption, QStyledItemDelegate, QMainWindow, QVBoxLayout
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QPixmap, QImage,QPainter
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Tuple
from openpyxl import Workbook
from openpyxl import load_workbook
import re
import shutil
import pandas as pd
import xcepxin_train
import typing
from PySide6.QtCore import QSize,QTimer

# 设置 Matplotlib 中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Arial']  # 优先使用支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示
current_dir = os.path.dirname(os.path.abspath(__file__)) #获取当前程序所在文件夹


def load_stl_and_plot_separate_views(stl_path):
    try:
        mesh = trimesh.load_mesh(stl_path)
        vertices = mesh.vertices
        print(f"STL文件加载成功！顶点数：{len(vertices)}，面数：{len(mesh.faces)}")
    except FileNotFoundError:
        print(f"错误：未找到STL文件，请检查路径：{stl_path}")
        return None
    except Exception as e:
        print(f"加载STL文件失败：{str(e)}")
        return None

    separate_views = [
        {"x_coord": vertices[:, 0], "y_coord": vertices[:, 2],
         "plot_title": "正视图（X-Z平面投影）", "x_label": "X轴", "y_label": "Z轴", "window_title": "正视图"},
        {"x_coord": vertices[:, 0], "y_coord": vertices[:, 1],
         "plot_title": "俯视图（X-Y平面投影）", "x_label": "X轴", "y_label": "Y轴", "window_title": "俯视图"},
        {"x_coord": vertices[:, 1], "y_coord": vertices[:, 2],
         "plot_title": "侧视图（Y-Z平面投影）", "x_label": "Y轴", "y_label": "Z轴", "window_title": "侧视图"}
    ]

    pixmaps = []
    point_size = 2
    for view in separate_views:
        fig = plt.figure(figsize=(4, 3), dpi=100)  # 调整大小以适应 QLabel
        plt.scatter(view["x_coord"], view["y_coord"], color='g', s=point_size, alpha=0.7, label="模型顶点")
        plt.title(view["plot_title"], fontsize=10, fontweight='bold', pad=10)
        plt.xlabel(view["x_label"], fontsize=8)
        plt.ylabel(view["y_label"], fontsize=8)
        plt.axis('equal')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=8)
        plt.tight_layout()

        # 将 Matplotlib 图形转换为 QPixmap
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)
        pixmaps.append(pixmap)
        plt.close(fig)  # 关闭图形以释放内存
        buf.close()

    return pixmaps  # 返回三个视图的 QPixmap 列表


def degrees_to_radians(angles: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """将角度（度）转换为弧度"""
    return tuple(np.radians(angle) for angle in angles)


def create_rotation_matrices(rx: float, ry: float, rz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建绕X、Y、Z轴的旋转矩阵"""
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    return R_x, R_y, R_z


def rotate_stl_vertices(vertices: np.ndarray, rx: float, ry: float, rz: float,
                        rotation_order: str = "xyz") -> np.ndarray:
    """对STL模型的顶点进行绕轴旋转"""
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    rx_rad, ry_rad, rz_rad = degrees_to_radians((rx, ry, rz))
    R_x, R_y, R_z = create_rotation_matrices(rx_rad, ry_rad, rz_rad)

    rotation_matrix = np.eye(3)
    for axis in rotation_order.lower():
        if axis == "x":
            rotation_matrix = rotation_matrix @ R_x
        elif axis == "y":
            rotation_matrix = rotation_matrix @ R_y
        elif axis == "z":
            rotation_matrix = rotation_matrix @ R_z
        else:
            raise ValueError(f"无效的旋转轴：{axis}，仅支持'x'、'y'、'z'")

    vertices_rotated = vertices_centered @ rotation_matrix.T
    vertices_final = vertices_rotated + center
    return vertices_final


def create_rotated_stl(mesh: trimesh.Trimesh, rotated_vertices: np.ndarray) -> trimesh.Trimesh:
    """基于旋转后的顶点创建新的STL网格对象"""
    rotated_mesh = trimesh.Trimesh(
        vertices=rotated_vertices,
        faces=mesh.faces,
        metadata=mesh.metadata
    )
    return rotated_mesh


def plot_rotated_views(rotated_mesh: trimesh.Trimesh, rx: float, ry: float, rz: float):
    """绘制旋转后模型的三视图，并返回 QPixmap 列表"""
    rot_verts = rotated_mesh.vertices
    views = [
        {"title": f"旋转后正视图（X-Z）\n(绕X:{rx}° Y:{ry}° Z:{rz}°)", "x": rot_verts[:, 0], "y": rot_verts[:, 2],
         "x_label": "X轴", "y_label": "Z轴"},
        {"title": f"旋转后俯视图（X-Y）\n(绕X:{rx}° Y:{ry}° Z:{rz}°)", "x": rot_verts[:, 0], "y": rot_verts[:, 1],
         "x_label": "X轴", "y_label": "Y轴"},
        {"title": f"旋转后侧视图（Y-Z）\n(绕X:{rx}° Y:{ry}° Z:{rz}°)", "x": rot_verts[:, 1], "y": rot_verts[:, 2],
         "x_label": "Y轴", "y_label": "Z轴"}
    ]

    pixmaps = []
    for view in views:
        fig = plt.figure(figsize=(4, 3), dpi=100)
        plt.scatter(view["x"], view["y"], c='crimson', s=1, alpha=0.6, label="旋转后模型")
        plt.title(view["title"], fontsize=10, fontweight='bold')
        plt.xlabel(view["x_label"], fontsize=8)
        plt.ylabel(view["y_label"], fontsize=8)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)
        pixmaps.append(pixmap)
        plt.close(fig)
        buf.close()

    return pixmaps

# ---------------- 主窗口类 ---------------- #
class BackgroundFrame(QFrame):
    def __init__(self, parent=None, bg_image_path=None):
        super().__init__(parent)
        self.bg_pixmap = QPixmap()
        if bg_image_path and os.path.exists(bg_image_path):
            self.bg_pixmap = QPixmap(bg_image_path)
        else:
            print(f"警告：背景图路径无效或文件不存在：{bg_image_path}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self.bg_pixmap.isNull():
            rect = self.rect()
            # 🔴 改成“按比例扩展铺满控件（允许裁剪）”
            scaled_pixmap = self.bg_pixmap.scaled(
                rect.size(),
                Qt.KeepAspectRatioByExpanding,  # 扩展到覆盖整个控件
                Qt.SmoothTransformation
            )
            # 居中裁剪显示
            pixmap_rect = scaled_pixmap.rect()
            pixmap_rect.moveCenter(rect.center())
            painter.drawPixmap(rect, scaled_pixmap, pixmap_rect)  # 用控件区域裁剪图片

        super().paintEvent(event)

class MyWindow:
    def __init__(self):
        # 1.加载登录界面
        login_window_name = "login.ui"  # 登录界面ui文件
        login_window_file = os.path.join(current_dir, login_window_name)
        self.current_window = self.load_ui(login_window_file)
        if not self.current_window:
            return
        # 2. 替换背景QFrame（必须修改这里的objectName！）
        TARGET_FRAME_NAME = "frame_2"  # 🔴 改成你Qt Designer中背景QFrame的objectName（比如frame、frame_1）
        original_frame = self.current_window.findChild(QFrame, TARGET_FRAME_NAME)
        if not original_frame:
            print(f"错误：找不到名为'{TARGET_FRAME_NAME}'的QFrame，请检查objectName！")
            return

        # 3. 手动指定背景图路径（避免解析样式表的问题，直接写绝对路径）
        bg_folder_name = "绘图\登录背景"  # 背景图所在文件夹（单独文件夹，不要包含文件名）
        bg_image_name = "登录背景.png"  # 背景图文件名
        bg_image_path = os.path.join(current_dir, bg_folder_name, bg_image_name)  # 正确拼接路径

        # 检查路径是否有效
        if not os.path.exists(bg_image_path):
            print(f"错误：背景图文件不存在！路径：{bg_image_path}")
            return

        # 4. 创建自定义Frame并替换（修改这部分）
        parent_widget = original_frame.parentWidget()
        layout = original_frame.layout()

        self.custom_frame = BackgroundFrame(parent=parent_widget, bg_image_path=bg_image_path)
        self.custom_frame.setObjectName(original_frame.objectName())
        self.custom_frame.setStyleSheet(original_frame.styleSheet())

        # 🔴 移除setGeometry，改用布局约束（让Frame随父控件自适应）
        if parent_widget.layout():
            parent_widget.layout().replaceWidget(original_frame, self.custom_frame)
        else:
            # 若父控件无布局，设置Frame为父控件的中心部件
            parent_widget.setCentralWidget(self.custom_frame)

        # 转移布局（保留子控件）
        if layout:
            self.custom_frame.setLayout(layout)

            # 显示自定义Frame，隐藏原Frame
            original_frame.hide()
            self.custom_frame.show()

            # 显示窗口
            self.current_window.show()

        # 绑定登录按钮（你 UI 中的 pushButton）
        if hasattr(self.current_window, "pushButton"):
            self.current_window.pushButton.clicked.connect(self.handle_login_button)
        else:
            print("⚠️ 警告：login.ui 中未找到 pushButton 组件")

        self.current_window.show()
        
        #定义用于数据交换所需的全局变量
        self.model_file = "./缓存/best_model.pth"  # 要移动的模型路径
        self.histroy_data = "./缓存/histroy_data.xlsx"  # 要移动的历史数据路径
        # self.model_file = "./bestmodel.pth"  # 要移动的模型路径
        # self.model_file = "./bestmodel.pth"  # 要移动的模型路径

    def switch_to_main_ui(self):
        """切换到主界面 UIzhujiemian.ui"""
        # 关闭当前窗口
        if self.current_window:
            self.current_window.close()

        # 加载新的主界面 UI
        zhujiemian_window_name = "UIzhujiemian.ui" #主界面ui文件
        zhujiemian_window_file = os.path.join(current_dir, zhujiemian_window_name)
        self.current_window = self.load_ui(zhujiemian_window_file)
        if not self.current_window:
            return

        # ←←← 新增：主界面加载完毕后，自动加载14张示意图
        QTimer.singleShot(100, self.load_styling_schematic_images)

        self.current_window.show()

    def check_login_valid(self) -> bool:
        """验证登录账号和密码"""
        user = self.current_window.lineEdit_1.text().strip() if hasattr(self.current_window, "lineEdit_1") else ""
        password = self.current_window.lineEdit_2.text().strip() if hasattr(self.current_window, "lineEdit_2") else ""

        if user == "Faw" and password == "19530715":
            return True
        else:
            QMessageBox.warning(self.current_window, "登录失败", "账号或密码错误，请重新输入！")
            return False

    def handle_login_button(self):
        """点击登录按钮后执行登录验证并跳转主界面"""
        if self.check_login_valid():
            self.switch_to_main_ui()

        # ---------------- 参数设置模块功能按钮 ---------------- #
        #------模型训练功能---------
        # 选择 目标定义数据集
        if hasattr(self.current_window, "pushButton_32"):
            self.current_window.pushButton_32.clicked.connect(self.select_file_yucemoxing_input)
        # 输出 目标定义结果
        if hasattr(self.current_window, "pushButton_36"):
            self.current_window.pushButton_36.clicked.connect(self.select_file_yucemoxing_output)
        if hasattr(self.current_window, "CPB_1"):
            self.current_window.CPB_1.clicked.connect(self.model_train)
        #保存模型
        if hasattr(self.current_window, "CPB_2"):
            self.current_window.CPB_2.clicked.connect(self.save_model)
        # 加载模型
        if hasattr(self.current_window, "CPB_3"):
            self.current_window.CPB_3.clicked.connect(self.select_Data_folder_canshushezhi)
            
        #------模型加载功能---------
        # 选择 目标定义数据集
        if hasattr(self.current_window, "pushButton_1"):
            self.current_window.pushButton_1.clicked.connect(self.select_Data_file)
        # 输出 目标定义结果
        if hasattr(self.current_window, "pushButton_3"):
            self.current_window.pushButton_3.clicked.connect(self.plot_photo)

        # ---------------- 目标定义模块功能按钮 ---------------- #
        #------基于响度目标定义功能---------
        # 选择 目标定义数据集
        if hasattr(self.current_window, "pushButton"):
            self.current_window.pushButton.clicked.connect(self.select_Data_folder_xingdudingyi)
        # 输出 目标定义结果
        if hasattr(self.current_window, "pushButton_2"):
            self.current_window.pushButton_2.clicked.connect(self.plot_xingdudingyi_data)
            
        #------基于噪声曲线目标定义功能---------
        # 选择 目标定义数据集
        if hasattr(self.current_window, "pushButton_1"):
            self.current_window.pushButton_1.clicked.connect(self.select_Data_file)
        # 输出 目标定义结果
        if hasattr(self.current_window, "pushButton_3"):
            self.current_window.pushButton_3.clicked.connect(self.plot_photo)

        # ---------------- 造型评估模块功能按钮 ---------------- #
        # 选择 STL 文件
        if hasattr(self.current_window, "pushButton_13"):
            self.current_window.pushButton_13.clicked.connect(self.select_file)
        # 显示原始三视图
        if hasattr(self.current_window, "pushButton_14"):
            self.current_window.pushButton_14.clicked.connect(self.run_stl_plot)
        # 执行旋转并显示旋转后三视图
        if hasattr(self.current_window, "pushButton_15"):
            self.current_window.pushButton_15.clicked.connect(self.run_stl_rotation)
        # 选择保存路径
        if hasattr(self.current_window, "pushButton_16"):
            self.current_window.pushButton_16.clicked.connect(self.save_rotated_stl)
        # 造型提取
        if hasattr(self.current_window, "pushButton_17"):
            self.current_window.pushButton_17.clicked.connect(self.select_file_2)
        # 点击 pushButton_8 输入数据（车高计算、SUV/轿车数据填充）
        if hasattr(self.current_window, "pushButton_18"):
            self.current_window.pushButton_18.clicked.connect(self.run_height_and_fill_data)
        if hasattr(self.current_window, "pushButton_19"):
            self.current_window.pushButton_19.clicked.connect(self.fill_default_values)

        #------初步判断功能---------
        #导入造型参数值
        if hasattr(self.current_window, "pushButton_23"):
            self.current_window.pushButton_23.clicked.connect(self.select_chubupanduan_zaoxingdaoru_file)
        #导入造型数据库
        if hasattr(self.current_window, "pushButton_24"):
            self.current_window.pushButton_24.clicked.connect(self.select_chubupanduan_zaoxingtuijian_file)
            
        #导入最大最小值
        if hasattr(self.current_window, "pushButton_25"):
            self.current_window.pushButton_25.clicked.connect(self.plot_zaoxingcanshupingjia_half)
        #导入90%区间值
        if hasattr(self.current_window, "pushButton_26"):
            self.current_window.pushButton_26.clicked.connect(self.plot_zaoxingcanshupingjia_half)
        #导入90%概率值
        if hasattr(self.current_window, "pushButton_27"):
            self.current_window.pushButton_27.clicked.connect(self.plot_zaoxingcanshupingjia_half)
                  #执行造型参数评价
        if hasattr(self.current_window, "pushButton_28"):
            self.current_window.pushButton_28.clicked.connect(self.plot_zaoxingcanshupingjia)                
           
        #------灵敏度分析功能---------
        #点击导入模型及数据集
        if hasattr(self.current_window, "pushButton_33"):
            self.current_window.pushButton_33.clicked.connect(self.select_folder_lingmingdu)
        #点击导入数据
        if hasattr(self.current_window, "pushButton_51"):
            self.current_window.pushButton_51.clicked.connect(self.select_lingmingduData_file)
        #点击进行灵敏度分析
        if hasattr(self.current_window, "pushButton_52"):
            self.current_window.pushButton_52.clicked.connect(self.plot_photo_lingmingdu)

        # ---------------- 预测模型模块功能按钮 ---------------- #
        #------模型预测---------
        #导入模型
        if hasattr(self.current_window, "pushButton_35"):
            self.current_window.pushButton_35.clicked.connect(self.select_folder_yucemoxing_model)
        #导入预测值
        if hasattr(self.current_window, "pushButton_54"):
            self.current_window.pushButton_54.clicked.connect(self.select_file_yucemoxing_predict)
        #执行模型预测
        if hasattr(self.current_window, "pushButton_55"):
            self.current_window.pushButton_55.clicked.connect(self.plot_photo_moxingyuce)


        #---------------- 造型优化模块功能按钮 ---------------- #
        
        #----基于具体频段-----
        if hasattr(self.current_window, "pushButton_7"):
            self.current_window.pushButton_7.clicked.connect(self.select_folder_and_fill_files)
        if hasattr(self.current_window, "pushButton_8"):
            self.current_window.pushButton_8.clicked.connect(self.select_file_zxpg_4)
        if hasattr(self.current_window, "pushButton_9"):
            self.current_window.pushButton_9.clicked.connect(self.plot_photo_moxingyouhua_pindian)  # 运行优化
        # if hasattr(self.current_window, "pushButton_10"):
        #     self.current_window.pushButton_10.clicked.connect(self.select_save_dir_zxpg)  # 保存优化结果
            
        #----基于整体响度-----
        if hasattr(self.current_window, "pushButton_11"):
            self.current_window.pushButton_11.clicked.connect(self.select_folder_and_fill_files_xiangdu)
        if hasattr(self.current_window, "pushButton_12"):
            self.current_window.pushButton_12.clicked.connect(self.select_file_zxpg_4_xiangdu)
        if hasattr(self.current_window, "pushButton_21"):
            self.current_window.pushButton_21.clicked.connect(self.plot_photo_moxingyouhua)


        # 显示主界面
        self.current_window.show()

    # ---------------- 登陆界面模块功能 ---------------- #
    def load_ui(self, path):
        ui_file = QFile(path)
        if not ui_file.open(QFile.ReadOnly):
            print(f"❌ 无法打开UI文件: {ui_file.errorString()}")
            return None
        loader = QUiLoader()
        window = loader.load(ui_file)
        ui_file.close()
        if not window:
            print(f"❌ UI加载失败: {loader.errorString()}")
            return None
        return window

    # ---------------- 参数设置模块功能 ---------------- #
    
    #----模型训练------
    #读取造型及技术方案   
    def select_file_yucemoxing_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
        self.current_window,
        "选择文件",
        "",
        "造型及技术方案文件 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "C_1"):
            self.current_window.C_1.setText(file_path)
            
     #导入造型数据库
    def select_file_yucemoxing_output(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window,
            "选择文件",
            "",
            "车内噪声文件 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "C_2"):
            self.current_window.C_2.setText(file_path)  
    #执行模型训练    
    #绘制箱型图
    def plot_fitness_history(self, best_fitness_history, avg_fitness_history,max_generations,widget_name):
        """绘制最佳适应度和平均适应度的折线图，并根据QWidget的尺寸调整图像大小"""
        
        # 获取 QWidget 的尺寸
        plot_widget = self.current_window.findChild(QWidget, widget_name)
        if not plot_widget:
            print(f"警告: 找不到名为'{widget_name}'的QWidget")
            return
        
        # 获取 QWidget 的宽度和高度
        widget_width = plot_widget.width()
        widget_height = plot_widget.height()
        
        # 创建matplotlib图形并调整图像大小以适应 QWidget_1 的尺寸
        fig, ax = plt.subplots(figsize=(widget_width / 100, widget_height / 100))  # 转换为英寸（1英寸=100像素）
        ax.plot(range(1, max_generations + 1), best_fitness_history, 'r-', linewidth=2,
                label='每代最优适应度')
        ax.plot(range(1, max_generations + 1), avg_fitness_history, 'b--', linewidth=2,
                label='每代平均适应度')
        
        # 设置标题和标签
        ax.set_title('遗传算法优化过程中的适应度曲线', fontsize=14)
        ax.set_xlabel('迭代次数', fontsize=14)
        ax.set_ylabel('适应度值', fontsize=14)
        ax.legend()
        
        # 将图表嵌入到 QWidget_2 中
        canvas = FigureCanvas(fig)
        canvas.setParent(plot_widget)
        canvas.draw()

        # 自动适应 QWidget_1 的大小
        canvas.setGeometry(plot_widget.rect())  # 根据QWidget_1的大小来设置图像尺寸
        canvas.setSizePolicy(plot_widget.sizePolicy())
        layout = plot_widget.layout()
        if layout is None:
            layout = QVBoxLayout(plot_widget)  #设置布局管理器

        # 显示图表
        plot_widget.layout().addWidget(canvas)    
        
    def plot_loss_history(self, losses, val_losses,widget_name):
        """绘制损失函数图，并根据QWidget的尺寸调整图像大小"""
    
        # 获取 QWidget 的尺寸
        plot_widget = self.current_window.findChild(QWidget, widget_name)
        if not plot_widget:
            print(f"警告: 找不到名为'{widget_name}'的QWidget")
            return
        
        # 获取 QWidget 的宽度和高度
        widget_width = plot_widget.width()
        widget_height = plot_widget.height()
        
        # 创建matplotlib图形并调整图像大小以适应 QWidget_1 的尺寸
        fig, ax = plt.subplots(figsize=(widget_width / 100, widget_height / 100))  # 转换为英寸（1英寸=100像素）
        ax.plot(range(1, len(losses) + 1), losses, label='训练损失', linewidth=2)
        ax.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失', linewidth=2)
        
        # 设置标题和标签
        ax.set_title('训练+验证损失曲线', fontsize=14)
        ax.set_xlabel('轮次', fontsize=14)
        ax.set_ylabel('损失', fontsize=14)
        ax.legend()
        
        # 将图表嵌入到 QWidget_1 中
        canvas = FigureCanvas(fig)
        canvas.setParent(plot_widget)
        canvas.draw()

        # 自动适应 QWidget_1 的大小
        canvas.setGeometry(plot_widget.rect())  # 根据QWidget_1的大小来设置图像尺寸
        canvas.setSizePolicy(plot_widget.sizePolicy())
        layout = plot_widget.layout()
        if layout is None:
            layout = QVBoxLayout(plot_widget)  #设置布局管理器

        # 显示图表
        plot_widget.layout().addWidget(canvas)
        
    def plot_boxplot(self, errors,widget_name):
        """绘制 losses 和 val_losses 之间差值的箱型图"""
        #定义频率刻度
        frequencies = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                   2000, 2500, 3150, 4000, 5000, 6300, 8000]
        
        # 获取 QWidget_1 的尺寸
        plot_widget = self.current_window.findChild(QWidget, widget_name)
        if not plot_widget:
            print(f"警告: 找不到名为'{widget_name}'的QWidget")
            return
        
        # 获取 QWidget_1 的宽度和高度
        widget_width = plot_widget.width()
        widget_height = plot_widget.height()
        
        # 创建 matplotlib 图形并调整图像大小以适应 QWidget_1 的尺寸
        fig, ax = plt.subplots(figsize=(widget_width / 100, widget_height / 100))  # 转换为英寸（1英寸=100像素）

        # 绘制箱型图
        bp = ax.boxplot(
            [errors[:, i] for i in range(errors.shape[1])],
            patch_artist=True,
            widths=0.6,
            tick_labels=[f'{f}Hz' for f in frequencies]
        )

        # 美化颜色
        for patch in bp['boxes']:
            patch.set_facecolor('#A0D8EF')       # 浅蓝色填充
        for whisker in bp['whiskers']:
            whisker.set_color('#333333')
        for cap in bp['caps']:
            cap.set_color('#333333')
        for median in bp['medians']:
            median.set_color('#FF4500')          # 中位数用橙红色
            median.set_linewidth(2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='#FF0000', alpha=0.5, markersize=5)

        # 8. 零误差参考线
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='零误差线')

        # 9. 坐标轴设置
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('误差（预测值 - 真实值） (dB)', fontsize=12)
        ax.set_title('各频率点预测误差分布', fontsize=14, pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor') #标签轴旋转
        plt.tight_layout()
        # 显示图表
        # 将图表嵌入到 QWidget_1 中
        canvas = FigureCanvas(fig)
        canvas.setParent(plot_widget)
        canvas.draw()

        # 自动适应 QWidget_1 的大小
        canvas.setGeometry(plot_widget.rect())  # 根据QWidget_1的大小来设置图像尺寸
        canvas.setSizePolicy(plot_widget.sizePolicy())

        layout = plot_widget.layout()
        if layout is None:
            layout = QVBoxLayout(plot_widget)  #设置布局管理器
        
        # 显示图表
        plot_widget.layout().addWidget(canvas)
    
    def model_train(self):
        #处理参数设置
        try:
            ga_max_generations = int(self.current_window.C_4.text().strip())
        except ValueError:
            QMessageBox.warning(self.current_window, "输入错误", "遗传算法迭代次数必须为数字！")
        try:
            ga_pop_size = int(self.current_window.C_3.text().strip())
        except ValueError:
            QMessageBox.warning(self.current_window, "输入错误", "遗传算法方案数量必须为数字！")
            
        best_fitness_history, avg_fitness_history, losses, val_losses, y_true_denorm, y_pred_denorm = xcepxin_train.model_Train_main(
             input_file_path=self.current_window.C_1.text().strip(),
             output_file_path=self.current_window.C_2.text().strip(),
             ga_max_generations=ga_max_generations,
             ga_pop_size=ga_pop_size
        )
        errors = np.subtract(y_pred_denorm, y_true_denorm )
        errors_to_list = errors.tolist() if hasattr(errors, 'tolist') else errors
        #绘制损失函数图
        self.plot_loss_history(losses, val_losses, "Cwidget_1") 
        # 绘制适应度历史图
        self.plot_fitness_history(best_fitness_history, avg_fitness_history, ga_max_generations, "Cwidget_2")     
        #绘制箱型图
        self.plot_boxplot(errors, "Cwidget_3")       
        #保存过程数据
        wb = Workbook()
        # 2. 将过程数据写入sheet
        ws1 = wb.active
        ws1.title = "适应度"  # 第一个sheet命名为“适应度”
        ws1.append(best_fitness_history.tolist() if hasattr(best_fitness_history, 'tolist') else best_fitness_history)
        ws1.append(avg_fitness_history.tolist() if hasattr(avg_fitness_history, 'tolist') else avg_fitness_history)
        ws2 = wb.create_sheet(title="损失")
        ws2.append(losses.tolist() if hasattr(losses, 'tolist') else losses)
        ws2.append(val_losses.tolist() if hasattr(val_losses, 'tolist') else val_losses)
        ws3 = wb.create_sheet(title="误差")
        for row in errors_to_list:
            ws3.append(row)
        wb.save("./缓存/histroy_data.xlsx")
        print("数据已写入多个sheet，文件保存成功！")
        
    #----模型保存------              
    #保存训练好的模型
    def save_model(self):
        """在旋转完成后保存 STL 文件"""
        if not hasattr(self, "model_train"):
            print("❌ 尚未进行模型训练，无法保存！")
            return

        # 弹出文件选择对话框
        save_path, _ = QFileDialog.getSaveFileName(self.current_window, "保存训练好的模型", "", "文件夹 (*)")
        try:
            # 4. 创建新文件夹（exist_ok=False 避免重名）
            os.makedirs(save_path, exist_ok=False)
        except FileExistsError:
            QMessageBox.critical(None, "错误", f"文件夹「{save_path}」已存在！")
            return
        except Exception as e:
            QMessageBox.critical(None, "错误", f"创建文件夹失败：{str(e)}")
            return

        # 5. 检查要移动的模型是否存在
        if not os.path.exists(self.model_file):
            QMessageBox.critical(None, "错误", f"指定文件「{self.model_file}」不存在！")
            return
        if not os.path.exists(self.histroy_data):
            QMessageBox.critical(None, "错误", f"指定文件「{self.histroy_data}」不存在！")
            return
        input_file_path=self.current_window.C_1.text().strip()
        output_file_path=self.current_window.C_2.text().strip()

        # 6. 拼接文件移动后的新路径
        model_name = os.path.basename(self.model_file) 
        new_model_path = os.path.join(save_path, model_name) #保存模型
        data_name = os.path.basename(self.histroy_data)
        new_data_path = os.path.join(save_path, data_name) #保存历史数据
        input_name = os.path.basename(input_file_path)
        new_input_path = os.path.join(save_path, input_name) #保存输入数据
        output_name = os.path.basename(output_file_path)
        new_output_path = os.path.join(save_path, output_name) #保存输入数据

        try:
            # 7. 移动文件到新文件夹
            shutil.move(self.model_file, new_model_path)
            shutil.move(self.histroy_data, new_data_path)
            shutil.move(input_file_path, new_input_path)
            shutil.move(output_file_path, new_output_path)
        except Exception as e:
            QMessageBox.critical(None, "错误", f"移动文件失败：{str(e)}")
            return

        # 8. 弹窗提示文件保存的路径
        QMessageBox.information(
            None, "成功", f"文件已移动至：\n{save_path}"
        )
    #-----模型导入功能---------
    def select_Data_folder_canshushezhi(self):
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        folder_path = QFileDialog.getExistingDirectory(None, "选择包含模型和数据的文件夹")
        if not folder_path:
            return

        pth_path = ""
        input_xlsx_path = ""
        output_xlsx_path = ""

        for file_name in os.listdir(folder_path):
            lower_name = file_name.lower()
            full_path = os.path.join(folder_path, file_name)

            if lower_name.endswith(".pth") and not pth_path:
                pth_path = full_path
            elif file_name == "输入数据.xlsx":
                input_xlsx_path = full_path
            elif file_name == "输出数据.xlsx":
                output_xlsx_path = full_path
            elif file_name == "histroy_data.xlsx":
                histroy_data = full_path
        #参数设置界面文件路径展示
        if hasattr(self.current_window, "C_5"):
            self.current_window.C_5.setText(pth_path)
        if hasattr(self.current_window, "C_6"):
            self.current_window.C_6.setText(input_xlsx_path)
        if hasattr(self.current_window, "C_7"):
            self.current_window.C_7.setText(output_xlsx_path)
        #目标定义界面文件路径展示
        if hasattr(self.current_window, "M_2"):
            self.current_window.M_2.setText(pth_path)
        #灵敏度分析界面文件路径展示
        if hasattr(self.current_window, "ZL_1"):
            self.current_window.ZL_1.setText(pth_path)
        #模型预测界面文件路径展示
        if hasattr(self.current_window, "Y_1"):
            self.current_window.Y_1.setText(pth_path)
        #造型优化界面文件路径展示
        if hasattr(self.current_window, "ZJP_1"):
            self.current_window.ZJP_1.setText(pth_path) #基于具体频段优化
        if hasattr(self.current_window, "ZJX_1"):
            self.current_window.ZJX_1.setText(pth_path) #基于具体频段优化

        msg = f"📁 已选择文件夹：{folder_path}\n"
        msg += f"\n模型文件 (.pth)：{pth_path if pth_path else '未找到'}"
        msg += f"\n输入数据.xlsx：{input_xlsx_path if input_xlsx_path else '未找到'}"
        msg += f"\n输出数据.xlsx：{output_xlsx_path if output_xlsx_path else '未找到'}"
        QMessageBox.information(None, "文件检测结果", msg)
        #解析训练历史数据
        # 读取指定sheet的数据
        df_fitness = pd.read_excel(histroy_data, sheet_name="适应度", header=None)
        df_loss = pd.read_excel(histroy_data, sheet_name="损失", header=None)
        df_error = pd.read_excel(histroy_data, sheet_name="误差", header=None)
        # 读取指定行（pandas的行索引从0开始，与你的原代码逻辑一致）
        best_fitness_history = df_fitness.iloc[0].tolist()  # 第0行=最优适应度
        avg_fitness_history = df_fitness.iloc[1].tolist()  # 第1行=平均适应度
        losses = df_loss.iloc[0].tolist()                  # 第0行=训练损失
        val_losses = df_loss.iloc[1].tolist()              # 第1行=验证损失
        errors = df_error.values.tolist()                  # 读取所有误差数据
        errors = np.array(errors)
        
        #绘制损失函数图
        self.plot_loss_history(losses, val_losses, "Cwidget_4") 
        # 绘制适应度历史图
        ga_max_generations = len(best_fitness_history)
        self.plot_fitness_history(best_fitness_history, avg_fitness_history, ga_max_generations, "Cwidget_5") 
        #绘制箱型图
        self.plot_boxplot(errors, "Cwidget_6") 
           
          
            

    # ---------------- 目标定义模块功能 ---------------- #
    #-----基于响度目标定义功能---------
    def select_Data_folder_xingdudingyi(self):
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        folder_path = QFileDialog.getExistingDirectory(None, "选择包含模型和数据的文件夹")
        if not folder_path:
            return

        pth_path = ""
        input_xlsx_path = ""
        output_xlsx_path = ""

        for file_name in os.listdir(folder_path):
            lower_name = file_name.lower()
            full_path = os.path.join(folder_path, file_name)

            if lower_name.endswith(".pth") and not pth_path:
                pth_path = full_path
            elif file_name == "输入数据.xlsx":
                input_xlsx_path = full_path
            elif file_name == "输出数据.xlsx":
                output_xlsx_path = full_path

        if hasattr(self.current_window, "lineEdit_5"):
            self.current_window.lineEdit_5.setText(pth_path)
        # if hasattr(self.current_window, "lineEdit_137"):
        #     self.current_window.lineEdit_137.setText(input_xlsx_path)
        # if hasattr(self.current_window, "lineEdit_115"):
        #     self.current_window.lineEdit_115.setText(output_xlsx_path)

        msg = f"📁 已选择文件夹：{folder_path}\n"
        msg += f"\n模型文件 (.pth)：{pth_path if pth_path else '未找到'}"
        msg += f"\n输入数据.xlsx：{input_xlsx_path if input_xlsx_path else '未找到'}"
        msg += f"\n输出数据.xlsx：{output_xlsx_path if output_xlsx_path else '未找到'}"
        QMessageBox.information(None, "文件检测结果", msg)
        #结果导入
    def plot_xingdudingyi_data(self):
        """计算评价及写入"""

        try:

            data1 = ["52.10","69.81","37.41","73.68","0.00","9.48", "2.71","3.22","0.85",
                    "23.04","33.18","33.18","25.80","25.80","78.56", "78.56", "58.17","58.17"]
            
            data2 = [ "111.68","187.32","2282.34","2876.36","32.98","53.80","38.48","65.24","54.87",
                     "59.30","2.60","7.74","22.63","42.11","82.34","90.00","1.63","2.02"]
            data3 = ["204.01","252.34", "209.01","250.36","148.94","170.74","63.29","87.24","68.11",
                     "75.08","170.72","264.00","17.00","22.50", "18.00","25.00","149.41","157.04"]
            data4 = ["75.51","126.58","34.06","70.15","5.79","32.00","0.00", "3.71","0.00",
                     "11.58","4.50","12.86","2.42","29.03","0.00","45.71","7.14", "12.46"]
            data5 = ["76.41","141.75","26.57","63.56","9.81","23.07","0.07","2.89","6.38",
                     "8.75", "1.76","8.24","5.13","20.30","0.00","39.25","7.14","12.46"]

            # 选择输出数据
            output_data = data1 

            # 写入 lineEdit_549 ~ lineEdit_598
            for i, value in enumerate(data1):
                line_name = f"MAS_{i+1}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)
                    
            for i, value in enumerate(data2):
                line_name = f"MAX_{i+1}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)
                    
            for i, value in enumerate(data3):
                line_name = f"MH_{i+1}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)

            for i, value in enumerate(data4):
                line_name = f"MY0_{i+1}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)

            for i, value in enumerate(data5):
                line_name = f"MQ_{i+1}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)

        except Exception as e:
            QMessageBox.critical(self.current_window, "错误", f"运行出错：\n{e}")
    
      
    #-----基于噪声曲线目标定义功能---------
    def select_Data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window,
            "选择文件",
            "",
            "数据集 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "lineEdit_6"):
            self.current_window.lineEdit_6.setText(file_path)
            
    def plot_photo(self):
        """绘制目标定义结果图"""
        
        #从文件夹中提取图像
        def load_images_to_array(folder_path, image_names):
            """
            从指定文件夹读取图像并存储到数组中
            
            Args:
                folder_path (str): 图像文件夹路径
                image_names (list): 要读取的图像文件名列表（最多4个）
                
            Returns:
                list: 包含QPixmap对象的数组，如果图像不存在则对应位置为None
            """
            # 初始化结果数组
            pixmaps = []
            
            # 确保image_names是列表且最多包含4个文件名
            if not isinstance(image_names, list):
                raise TypeError("image_names必须是一个列表")
            
            # 限制为最多4张图像
            image_names = image_names[:4]
            
            for img_name in image_names:
                # 构建完整的文件路径
                img_path = os.path.join(folder_path, img_name)
                
                # 检查文件是否存在
                if os.path.exists(img_path):
                    # 创建QPixmap对象
                    pixmap = QPixmap(img_path)
                    
                    # 检查图像是否成功加载
                    if not pixmap.isNull():
                        pixmaps.append(pixmap)
                        print(f"✅ 成功加载图像: {img_name}")
                    else:
                        pixmaps.append(None)
                        print(f"❌ 无法加载图像: {img_name}（格式不支持或文件损坏）")
                else:
                    pixmaps.append(None)
                    print(f"❌ 图像文件不存在: {img_name}")
            
            return pixmaps
        folder_name = "绘图\目标定义"
        folder_path = os.path.join(current_dir, folder_name)
        image_names = ["数据展示.png", "A.png", "B.png", "L.png"]
        # 加载图像
        pixmaps = load_images_to_array(folder_path, image_names)
        
        if pixmaps and len(pixmaps) == 4:

            if hasattr(self.current_window, "label_272"):
                self.current_window.label_272.setPixmap(pixmaps[1].scaled(
                    self.current_window.label_272.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_272 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
            if hasattr(self.current_window, "label_273"):
                self.current_window.label_273.setPixmap(pixmaps[2].scaled(
                    self.current_window.label_273.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_273 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
            if hasattr(self.current_window, "label_271"):
                self.current_window.label_271.setPixmap(pixmaps[3].scaled(
                self.current_window.label_271.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_2 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
        else:
            print("❌ 无法生成目标定义图，请检查数据集文件！")
        

        

    # ---------------- 造型评估模块功能 ---------------- #
    
    # ----- STL文件预处理 -----
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window,
            "选择文件",
            "",
            "STL文件 (*.stl);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "lineEdit_22"):
            self.current_window.lineEdit_22.setText(file_path)

    def run_stl_plot(self):
        """从 lineEdit 获取 STL 文件路径并将三视图显示在 label_86、label_87、label_88 中"""
        if hasattr(self.current_window, "lineEdit_22"):
            stl_path = self.current_window.lineEdit_22.text().strip()
            if stl_path:
                pixmaps = load_stl_and_plot_separate_views(stl_path)
                if pixmaps and len(pixmaps) == 3:
                    if hasattr(self.current_window, "label_86"):
                        self.current_window.label_86.setPixmap(pixmaps[0].scaled(
                            self.current_window.label_86.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                    else:
                        print("❌ label_86 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
                    if hasattr(self.current_window, "label_87"):
                        self.current_window.label_87.setPixmap(pixmaps[1].scaled(
                            self.current_window.label_87.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                    else:
                        print("❌ label_87 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
                    if hasattr(self.current_window, "label_88"):
                        self.current_window.label_88.setPixmap(pixmaps[2].scaled(
                            self.current_window.label_88.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                    else:
                        print("❌ label_88 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
                else:
                    print("❌ 无法生成三视图，请检查 STL 文件！")
            else:
                print("❌ lineEdit 为空，请先选择 STL 文件！")

    def run_stl_rotation(self):
        """执行 STL 旋转并将旋转后三视图显示在 label_95、label_96、label_97 中"""
        if not hasattr(self.current_window, "lineEdit_22"):
            print("❌ lineEdit 不存在，请检查 UI 文件")
            return

        stl_path = self.current_window.lineEdit_22.text().strip()
        if not stl_path:
            print("❌ lineEdit 为空，请先选择 STL 文件！")
            return

        # 获取旋转角度
        try:
            rx = float(self.current_window.lineEdit_25.text().strip()) if hasattr(self.current_window,
                                                                                  "lineEdit_25") else 0
            ry = float(self.current_window.lineEdit_26.text().strip()) if hasattr(self.current_window,
                                                                                  "lineEdit_26") else 0
            rz = float(self.current_window.lineEdit_27.text().strip()) if hasattr(self.current_window,
                                                                                  "lineEdit_27") else 0
        except ValueError:
            print("❌ 旋转角度输入无效，请输入有效数字！")
            return

        # 加载 STL 文件
        try:
            self.original_mesh = trimesh.load_mesh(stl_path, force='mesh')
            print(f"原始模型信息：顶点数={len(self.original_mesh.vertices)}，面数={len(self.original_mesh.faces)}")
        except FileNotFoundError:
            print(f"❌ 未找到 STL 文件：{stl_path}")
            return
        except Exception as e:
            print(f"加载 STL 文件失败：{str(e)}")
            return

        # 执行旋转
        print(f"正在执行旋转（顺序：xyz）...")
        self.rotated_vertices = rotate_stl_vertices(
            vertices=self.original_mesh.vertices,
            rx=rx, ry=ry, rz=rz,
            rotation_order="xyz"
        )
        self.rotated_mesh = create_rotated_stl(self.original_mesh, self.rotated_vertices)

        # 生成旋转后三视图并显示
        pixmaps = plot_rotated_views(self.rotated_mesh, rx, ry, rz)
        if pixmaps and len(pixmaps) == 3:
            if hasattr(self.current_window, "label_95"):
                self.current_window.label_95.setPixmap(pixmaps[0].scaled(
                    self.current_window.label_95.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            if hasattr(self.current_window, "label_96"):
                self.current_window.label_96.setPixmap(pixmaps[1].scaled(
                    self.current_window.label_96.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            if hasattr(self.current_window, "label_97"):
                self.current_window.label_97.setPixmap(pixmaps[2].scaled(
                    self.current_window.label_97.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        else:
            print("❌ 无法生成旋转后三视图，请检查 STL 文件或旋转参数！")

    def save_rotated_stl(self):
        """在旋转完成后保存 STL 文件"""
        if not hasattr(self, "rotated_mesh"):
            print("❌ 尚未旋转 STL，无法保存！")
            return

        # 弹出文件选择对话框
        save_path, _ = QFileDialog.getSaveFileName(self.current_window, "保存旋转后的 STL 文件", "", "STL Files (*.stl)")
        if save_path:
            try:
                self.rotated_mesh.export(save_path)
                print(f"旋转后的 STL 已保存至：{save_path}")
            except Exception as e:
                print(f"保存旋转后 STL 失败：{str(e)}")

    def select_file_2(self):
        """选择 STL 文件路径，写入 lineEdit_28"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window, "选择STL文件", "", "STL文件 (*.stl);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "lineEdit_28"):
            self.current_window.lineEdit_28.setText(file_path)
            print(f"✅ 已选择STL文件：{file_path}")
        else:
            print("❌ 未选择文件或 lineEdit_28 不存在")

    def run_height_and_fill_data(self):
        """计算车高并写入 SUV/轿车数据到 lineEdit_500~548"""
        stl_path = self.current_window.lineEdit_28.text().strip()

        if not stl_path:
            QMessageBox.warning(self.current_window, "提示", "请先选择STL文件！")
            return

        try:
            mesh = trimesh.load_mesh(stl_path)
            vertices = mesh.vertices

            # 计算车高
            z_min = np.min(vertices[:, 2])
            z_max = np.max(vertices[:, 2])
            H = z_max - z_min
            print(f"计算得到车高 H = {H:.2f} mm")

            # SUV 数据
            data1 = [
                "76.41 - 141.75", "26.57 - 63.56", "9.81 - 23.07", "0.07 - 2.89", "6.38 - 8.75",
                "1.76 - 8.24", "5.13 - 20.30", "0.00 - 39.25", "7.14 - 12.46", "75.51 - 126.58",
                "34.06 - 70.15", "5.79 - 32.00", "0.00 - 3.71", "0.00 - 11.58", "4.50 - 12.86",
                "2.42 - 29.03", "0.00 - 45.71", "7.14 - 12.46", "204.01 - 252.34", "209.01 - 250.36",
                "148.94 - 170.74", "63.29 - 87.24", "68.11 - 75.08", "170.72 - 264.00", "17.00 - 22.50",
                "18.00 - 25.00", "149.41 - 157.04", "111.68 - 187.32", "2282.34 - 2876.36", "32.98 - 53.80",
                "38.48 - 65.24", "54.87 - 59.30", "2.60 - 7.74", "22.63 - 42.11", "82.34 - 90.00",
                "1.63 - 2.02", "19 - 22", "21 - 25", "52.10 - 69.81", "37.41 - 73.68",
                "0.00 - 9.48", "2.71 - 3.22","0.85 - 23.04", "33.18 - 60.57", "25.80 - 34.30",
                "78.56 - 81.68", "58.17 - 65.76", "180 - 270", "1.63 - 2.02"
            ]

            # 轿车数据
            data2 = [
                "71.80 - 178.75", "22.17 - 46.09", "2.13 - 41.44", "0.20 - 3.64", "5.97 - 14.98",
                "1.98 - 11.43", "0.19 - 37.75", "0.00 - 29.51", "6.55 - 15.00", "71.42 - 159.29",
                "24.35 - 67.09", "3.53 - 107.43", "0.11 - 3.87", "4.99 - 12.63", "3.64 - 17.77",
                "1.46 - 38.77", "0.00 - 28.04", "6.55 - 15.00", "172.43 - 232.44", "183.01 - 240.44",
                "127.63 - 171.84", "60.66 - 96.24", "69.64 - 77.52", "125.20 - 243.69", "13.00 - 19.00",
                "14.00 - 20.00", "148.97 - 181.66", "8.88 - 148.54", "564.81 - 3244.37", "17.98 - 66.08",
                "12.85 - 79.97", "55.61 - 64.15", "3.19 - 10.07", "12.77 - 59.69", "52.12 - 90.00",
                "1.72 - 2.24", "13 - 18", "14 - 19", "50.46 - 68.26", "40.27 - 69.54",
                "0.00 - 16.54", "2.42 - 3.43","16.14 - 28.41", "28.12 - 89.71", "23.79 - 44.28",
                "75.11 - 84.25", "49.52 - 68.02", "125 - 180", "1.72 - 2.24"
            ]

            # 选择输出数据
            output_data = data1 if H > 1600 else data2
            car_type = "SUV" if H > 1600 else "轿车"
            print(f"检测结果：{car_type}（H = {H:.2f} mm）")

            # 写入 lineEdit_500 ~ lineEdit_548
            for i, value in enumerate(output_data):
                line_name = f"lineEdit_{i + 500}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)

            QMessageBox.information(
                self.current_window,
                "完成",
                f"检测结果：{car_type}\n车高 H = {H:.2f} mm\n数据已写入 lineEdit_500~lineEdit_548"
            )

        except Exception as e:
            QMessageBox.critical(self.current_window, "错误", f"运行出错：\n{e}")

    def fill_default_values(self):
        """点击按钮后向 lineEdit_500~548 写入默认数据（红色）"""

        # SUV 数据
        data1 = [
            "76.41", "26.57", "9.81", "0.07", "6.38",
            "1.76", "5.13", "0.00", "7.14", "75.51",
            "34.06", "5.79", "0.00", "0.00", "4.50",
            "2.42", "0.00", "7.14", "204.01", "209.01",
            "148.94", "63.29", "68.11", "170.72", "17.00",
            "18.00", "149.41", "111.68", "2282.34", "32.98",
            "38.48", "54.87", "2.60", "22.63", "82.34",
            "1.63", "19", "21", "52.10", "37.41",
            "0.00", "2.71", "0.85", "33.18", "25.80",
            "78.56", "58.17", "180", "1.63"
        ]

        # # 轿车数据（如需使用，把 data1 改成 data2 即可）
        # data2 = [
        #     "71.80", "22.17", "2.13", "0.20", "5.97",
        #     "1.98", "0.19", "0.00", "6.55", "71.42",
        #     "24.35", "3.53", "0.11", "4.99", "3.64",
        #     "1.46", "0.00", "6.55", "172.43", "183.01",
        #     "127.63", "60.66", "69.64", "125.20", "13.00",
        #     "14.00", "148.97", "8.88", "564.81", "17.98",
        #     "12.85", "55.61", "3.19", "12.77", "52.12",
        #     "1.72", "13", "14", "50.46", "40.27",
        #     "0.00", "2.42", "16.14", "28.12", "23.79",
        #     "75.11", "49.52", "125", "1.72"
        # ]

        # 选择要填充的数据（默认 SUV）
        values = data1

        # 遍历 lineEdit_500 ~ lineEdit_548
        start_id = 500
        for i, val in enumerate(values):
            obj_name = f"lineEdit_{start_id + i}"

            if hasattr(self.current_window, obj_name):
                le = getattr(self.current_window, obj_name)
                le.setText(val)
                le.setStyleSheet("color: red;")  # 设置红色字体
            else:
                print(f"⚠ 未找到控件：{obj_name}（请检查 UIzhujiemianv3.ui）")

    # --------造型示意图------------
    def load_styling_schematic_images(self):
        """加载14张造型示意图（使用安全的相对路径，兼容直接运行和打包成exe）"""
        # 正确定义根目录
        if getattr(sys, 'frozen', False):
            current_dir = os.path.dirname(sys.executable)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))

        folder_name = "绘图/造型示意图"
        folder_path = os.path.join(current_dir, folder_name)

        # 打包后路径兼容
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else current_dir
            folder_path = os.path.join(base_path, folder_name)

        image_names = [
            "A柱上端X向尺寸.png", "A柱上端Y向尺寸.png", "前风挡上端R角.png",
            "A柱下端X向尺寸.png", "A柱下端Y向尺寸.png", "前风挡下端R角.png",
            "后视镜X向尺寸.png", "后视镜Y向尺寸.png", "后视镜末端.png",
            "前轮腔前（后）X向尺寸.png", "后三角窗阶差.png", "顶棚挠度.png",
            "接近角.png", "离去角.png"
        ]

        label_names = [
            "label_14", "label_21", "label_22", "label_27", "label_28",
            "label_40", "label_42", "label_51", "label_53", "label_148",
            "label_61", "label_56", "label_58", "label_59"
        ]

        if len(image_names) != len(label_names):
            print(f"【错误】图片数量({len(image_names)}) ≠ label数量({len(label_names)})")
            return

        success_count = 0

        # 打印表头，方便查看
        print(f"\n{'Label名称':<12} | {'尺寸(WxH)':<12} | {'状态':<8} | {'图片文件'}")
        print("-" * 80)

        for idx, (img_name, label_name) in enumerate(zip(image_names, label_names), 1):
            img_path = os.path.normpath(os.path.join(folder_path, img_name))

            # 1. 检查Label是否存在
            label = self.current_window.findChild(QLabel, label_name)
            if not label:
                print(f"{label_name:<12} | {'None':<12} | ❌ 缺失 | {img_name}")
                continue

            # 2. 【核心修改】在此处获取并打印尺寸
            w, h = label.width(), label.height()

            # 标记尺寸状态
            size_str = f"{w}x{h}"
            # 如果宽或高小于50，通常意味着布局未完成或在隐藏Tab页中，会导致图片缩成点
            status = "⚠️ 极小" if (w < 50 or h < 50) else "✅ 正常"

            print(f"{label_name:<12} | {size_str:<12} | {status} | {img_name}")

            # 3. 检查文件
            if not os.path.exists(img_path):
                print(f"  -> ❌ 图片不存在: {img_path}")
                continue

            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                print(f"  -> ❌ 图片损坏")
                continue

            # 4. 加载图片
            # 【建议】如果发现尺寸极小(status是警告)，强行给一个默认尺寸，防止图片不可见
            target_size = label.size()
            if w < 50 or h < 50:
                # 给一个临时默认值，确保图片能看清（例如 400x300）
                target_size = QSize(699, 536)

            scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
            label.setAlignment(Qt.AlignCenter)
            label.setVisible(True)
            success_count += 1

        if getattr(sys, 'frozen', False):
            print(f"【打包】临时目录路径：{sys._MEIPASS if hasattr(sys, '_MEIPASS') else '未知'}")
        
    #------初步判断功能---------
     #导入造型参数值
    def select_chubupanduan_zaoxingdaoru_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window,
            "选择文件",
            "",
            "造型参数文件 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "lineEdit_8"):
            self.current_window.lineEdit_8.setText(file_path)
            
     #导入造型数据库
    def select_chubupanduan_zaoxingtuijian_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window,
            "选择文件",
            "",
            "造型参数文件 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "lineEdit_24"):
            self.current_window.lineEdit_24.setText(file_path)
    
    #显示评价范围
    def plot_zaoxingcanshupingjia_half(self):
        """计算评价及写入"""

        try:

            data1 = [
                    " ", " ", " ", " ", " ", 
                    "52.10", "37.41", "0.00", "2.71", "0.85", 
                    "69.81", "73.68", "9.48", "3.22", "23.04", 
                    " ", " ", " ", " ", " ", 
                    " ", " ", " ", " ", 
                    "33.18", "25.80", "78.56", "58.17", 
                    "33.18", "25.80", "78.56", "58.17", 
                    " ", " ", " ", " ",
                    " ", " ", " ", " ", " ", 
                    "111.68", "2282.34", "32.98", "38.48", "54.87", 
                    "187.32", "2876.36", "53.80", "65.24", "59.30", 
                    " ", " ", " ", " ", " ", 
                    " ", " ", " ", " ", 
                    "2.60", "22.63", "82.34", "1.63", 
                    "7.74", "42.11", "90.00", "2.02", 
                    " ", " ", " ", " ", 
                    " ", " ", " ", " ", " ", 
                    "204.01", "209.01", "148.94", "63.29", "68.11", 
                    "252.34", "250.36", "170.74", "87.24", "75.08", 
                    " ", " ", " ", " ", " ", 
                    " ", " ", " ", " ", 
                    "170.72", "17.00", "18.00", "149.41", 
                    "264.00", "22.50", "25.00", "157.04", 
                    " ", " ", " ", " ", 
                    " ", " ", " ", " ", " ", 
                    "75.51", "34.06", "5.79", "0.00", "0.00", 
                    "126.58", "70.15", "32.00", "3.71", "11.58", 
                    " ", " ", " ", " ", " ", 
                    " ", " ", " ", " ", 
                    "4.50", "2.42", "0.00", "7.14", 
                    "12.86", "29.03", "45.71", "12.46", 
                    " ", " ", " ", " ", 
                    " ", " ", " ", " ", " ", 
                    "76.41", "26.57", "9.81", "0.07", "6.38", 
                    "141.75", "63.56", "23.07", "2.89", "8.75", 
                    " ", " ", " ", " ", " ", 
                    " ", " ", " ", " ", 
                    "1.76", "5.13", "0.00", "7.14", 
                    "8.24", "20.30", "39.25", "12.46", 
                    " ", " ", " ", " ", 

                ]

            # 选择输出数据
            output_data = data1 

            # 写入 lineEdit_549 ~ lineEdit_598
            for i, value in enumerate(data1):
                line_name = f"lineEdit_{i + 550}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)

        except Exception as e:
            QMessageBox.critical(self.current_window, "错误", f"运行出错：\n{e}")
              
    #显示分析结果
    def plot_zaoxingcanshupingjia(self):
        """计算评价及写入"""

        try:

            data1 = [
                    "52.10", "37.41", "0.00", "2.71", "0.85", 
                    "52.10", "37.41", "0.00", "2.71", "0.85", 
                    "69.81", "73.68", "9.48", "3.22", "23.04", 
                    "正常", "正常", "正常", "正常", "正常", 
                    "33.18", "25.80", "78.56", "58.17", 
                    "33.18", "25.80", "78.56", "58.17", 
                    "33.18", "25.80", "78.56", "58.17", 
                    "正常", "正常", "正常", "正常",
                    "111.68", "2282.34", "32.98", "38.48", "54.87", 
                    "111.68", "2282.34", "32.98", "38.48", "54.87", 
                    "187.32", "2876.36", "53.80", "65.24", "59.30", 
                    "正常", "正常", "正常", "正常", "正常", 
                    "2.60", "22.63", "82.34", "1.63", 
                    "2.60", "22.63", "82.34", "1.63", 
                    "7.74", "42.11", "90.00", "2.02", 
                    "正常", "正常", "正常", "正常", 
                    "204.01", "209.01", "148.94", "63.29", "68.11", 
                    "204.01", "209.01", "148.94", "63.29", "68.11", 
                    "252.34", "250.36", "170.74", "87.24", "75.08", 
                    "正常", "正常", "正常", "正常", "正常", 
                    "170.72", "17.00", "18.00", "149.41", 
                    "170.72", "17.00", "18.00", "149.41", 
                    "264.00", "22.50", "25.00", "157.04", 
                    "正常", "正常", "正常", "正常", 
                    "75.51", "34.06", "5.79", "0.00", "0.00", 
                    "75.51", "34.06", "5.79", "0.00", "0.00", 
                    "126.58", "70.15", "32.00", "3.71", "11.58", 
                    "正常", "正常", "正常", "正常", "正常", 
                    "4.50", "2.42", "0.00", "7.14", 
                    "4.50", "2.42", "0.00", "7.14", 
                    "12.86", "29.03", "45.71", "12.46", 
                    "正常", "正常", "正常", "正常", 
                    "76.41", "26.57", "9.81", "0.07", "6.38", 
                    "76.41", "26.57", "9.81", "0.07", "6.38", 
                    "141.75", "63.56", "23.07", "2.89", "8.75", 
                    "正常", "正常", "正常", "正常", "正常", 
                    "1.76", "5.13", "0.00", "7.14", 
                    "1.76", "5.13", "0.00", "7.14", 
                    "8.24", "20.30", "39.25", "12.46", 
                    "正常", "正常", "正常", "正常", 

                ]

            # 选择输出数据
            output_data = data1 

            # 写入 lineEdit_549 ~ lineEdit_598
            for i, value in enumerate(data1):
                line_name = f"lineEdit_{i + 550}"
                if hasattr(self.current_window, line_name):
                    getattr(self.current_window, line_name).setText(value)

        except Exception as e:
            QMessageBox.critical(self.current_window, "错误", f"运行出错：\n{e}")



    #--------灵敏度分析功能------------
    def select_folder_lingmingdu(self):
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        folder_path = QFileDialog.getExistingDirectory(None, "选择包含模型和数据的文件夹")
        if not folder_path:
            return

        pth_path = ""
        input_xlsx_path = ""
        output_xlsx_path = ""

        for file_name in os.listdir(folder_path):
            lower_name = file_name.lower()
            full_path = os.path.join(folder_path, file_name)

            if lower_name.endswith(".pth") and not pth_path:
                pth_path = full_path
            elif file_name == "输入数据.xlsx":
                input_xlsx_path = full_path
            elif file_name == "输出数据.xlsx":
                output_xlsx_path = full_path

        if hasattr(self.current_window, "lineEdit_136"):
            self.current_window.lineEdit_136.setText(pth_path)
        # if hasattr(self.current_window, "lineEdit_137"):
        #     self.current_window.lineEdit_137.setText(input_xlsx_path)
        # if hasattr(self.current_window, "lineEdit_115"):
        #     self.current_window.lineEdit_115.setText(output_xlsx_path)

        msg = f"📁 已选择文件夹：{folder_path}\n"
        msg += f"\n模型文件 (.pth)：{pth_path if pth_path else '未找到'}"
        msg += f"\n输入数据.xlsx：{input_xlsx_path if input_xlsx_path else '未找到'}"
        msg += f"\n输出数据.xlsx：{output_xlsx_path if output_xlsx_path else '未找到'}"
        QMessageBox.information(None, "文件检测结果", msg)
        
    def select_lingmingduData_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.current_window,
            "选择文件",
            "",
            "数据集 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path and hasattr(self.current_window, "lineEdit_116"):
            self.current_window.lineEdit_116.setText(file_path)
            
    def plot_photo_lingmingdu(self):
        """绘制目标定义结果图"""
        #从输入框中获取图像名称
        def parse_coordinate_string(text):
            """
            将格式为"(200,300)"的文本解析成包含两个数字的数组
            
            Args:
                text (str): 输入的坐标字符串，格式为"(数字1,数字2)"
                
            Returns:
                list: 包含两个整数的列表 [数字1, 数字2]
                
            Raises:
                ValueError: 当输入格式不正确或无法转换为数字时
            """
            try:
                # 移除括号并去除前后空白字符
                clean_text = text.strip('() ')
                
                # 以逗号为分隔符分割字符串
                parts = clean_text.split(',')
                
                # 确保只有两个部分
                if len(parts) != 2:
                    raise ValueError("输入格式不正确，应为'(数字1,数字2)'格式")
                
                # 去除每个部分的空白字符并转换为整数
                num1 = int(parts[0].strip())
                num2 = int(parts[1].strip())
                
                # 返回包含两个数字的列表
                return [num1, num2]
            except Exception as e:
                # 如果解析失败，抛出详细的错误信息
                raise ValueError(f"无法解析输入字符串: {e}")
        #从文件夹中提取图像
        def load_images_to_array(folder_path, image_names):
            """
            从指定文件夹读取图像并存储到数组中
            
            Args:
                folder_path (str): 图像文件夹路径
                image_names (list): 要读取的图像文件名列表（最多4个）
                
            Returns:
                list: 包含QPixmap对象的数组，如果图像不存在则对应位置为None
            """
            # 初始化结果数组
            pixmaps = []
            
            # 确保image_names是列表且最多包含4个文件名
            if not isinstance(image_names, list):
                raise TypeError("image_names必须是一个列表")
            
            # 限制为最多4张图像
            image_names = image_names[:18]
            
            for img_name in image_names:
                # 构建完整的文件路径
                img_path = os.path.join(folder_path, img_name)
                
                # 检查文件是否存在
                if os.path.exists(img_path):
                    # 创建QPixmap对象
                    pixmap = QPixmap(img_path)
                    
                    # 检查图像是否成功加载
                    if not pixmap.isNull():
                        pixmaps.append(pixmap)
                        print(f"✅ 成功加载图像: {img_name}")
                    else:
                        pixmaps.append(None)
                        print(f"❌ 无法加载图像: {img_name}（格式不支持或文件损坏）")
                else:
                    pixmaps.append(None)
                    print(f"❌ 图像文件不存在: {img_name}")
            
            return pixmaps
        folder_name = "绘图\灵敏度结果"
        folder_path = os.path.join(current_dir, folder_name)
        image_names = ["全频段.png", "200Hz.png", "250Hz.png", "315Hz.png", "400Hz.png", "500Hz.png", "630Hz.png", 
                       "800Hz.png", "1000Hz.png", "1250Hz.png", "1600Hz.png", "2000Hz.png", "2500Hz.png", "3150Hz.png",
                       "4000Hz.png", "5000Hz.png", "6300Hz.png", "8000Hz.png"]
        # 加载图像
        pixmaps = load_images_to_array(folder_path, image_names)
        photo_name = self.current_window.lineEdit_4.text().strip() if hasattr(self.current_window, "lineEdit_4") else "" #获取文本
        fre_range = parse_coordinate_string(photo_name) #转换为数字
              
        if pixmaps and len(pixmaps) == 18:
            if fre_range[0] == fre_range[1]:
                target_filename = f"{fre_range[0]}Hz.png"
                try:
                    position = image_names.index(target_filename)                  
                except ValueError:
                    print(f"{target_filename} 超出计算范围")
 
                if hasattr(self.current_window, "label_166"):
                    self.current_window.label_166.setPixmap(pixmaps[position].scaled(
                        self.current_window.label_166.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                if hasattr(self.current_window, "label_166"):
                    self.current_window.label_166.setPixmap(pixmaps[0].scaled(
                        self.current_window.label_166.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))               

        else:
            print("❌ 无法进行灵敏度计算，请检查数据集文件！")   

    # ---------------- 预测模型模块功能 ---------------- #

    #----模型预测------
    #加载模型文件
    def select_folder_yucemoxing_model(self):
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        folder_path = QFileDialog.getExistingDirectory(None, "选择包含模型和数据的文件夹2")
        if not folder_path:
            return

        pth_path = ""
        input_xlsx_path = ""
        output_xlsx_path = ""

        for file_name in os.listdir(folder_path):
            lower_name = file_name.lower()
            full_path = os.path.join(folder_path, file_name)

            if lower_name.endswith(".pth") and not pth_path:
                pth_path = full_path
            elif file_name == "输入数据.xlsx":
                input_xlsx_path = full_path
            elif file_name == "输出数据.xlsx":
                output_xlsx_path = full_path

        if hasattr(self.current_window, "lineEdit_138"):
            self.current_window.lineEdit_138.setText(pth_path)
        # if hasattr(self.current_window, "lineEdit_137"):
        #     self.current_window.lineEdit_137.setText(input_xlsx_path)
        # if hasattr(self.current_window, "lineEdit_115"):
        #     self.current_window.lineEdit_115.setText(output_xlsx_path)

        msg = f"📁 已选择文件夹：{folder_path}\n"
        msg += f"\n模型文件 (.pth)：{pth_path if pth_path else '未找到'}"
        msg += f"\n输入数据.xlsx：{input_xlsx_path if input_xlsx_path else '未找到'}"
        msg += f"\n输出数据.xlsx：{output_xlsx_path if output_xlsx_path else '未找到'}"
        QMessageBox.information(None, "文件检测结果", msg)
        
    #加载预测数据
    def select_file_yucemoxing_predict(self):
        file_path, _ = QFileDialog.getOpenFileName(
        self.current_window,
        "选择文件",
        "",
        "车内噪声文件 (*.xlsx);;所有文件 (*.*)"
    )
        if file_path and hasattr(self.current_window, "lineEdit_118"):
            self.current_window.lineEdit_118.setText(file_path)
            
    #绘制预测结果  
    def plot_photo_moxingyuce(self):
        """绘制模型预测结果图"""
        
        #从文件夹中提取图像
        def load_images_to_array(folder_path, image_names):
            """
            从指定文件夹读取图像并存储到数组中
            
            Args:
                folder_path (str): 图像文件夹路径
                image_names (list): 要读取的图像文件名列表（最多4个）
                
            Returns:
                list: 包含QPixmap对象的数组，如果图像不存在则对应位置为None
            """
            # 初始化结果数组
            pixmaps = []
            
            # 确保image_names是列表且最多包含4个文件名
            if not isinstance(image_names, list):
                raise TypeError("image_names必须是一个列表")
            
            # 限制为最多4张图像
            image_names = image_names[:4]
            
            for img_name in image_names:
                # 构建完整的文件路径
                img_path = os.path.join(folder_path, img_name)
                
                # 检查文件是否存在
                if os.path.exists(img_path):
                    # 创建QPixmap对象
                    pixmap = QPixmap(img_path)
                    
                    # 检查图像是否成功加载
                    if not pixmap.isNull():
                        pixmaps.append(pixmap)
                        print(f"✅ 成功加载图像: {img_name}")
                    else:
                        pixmaps.append(None)
                        print(f"❌ 无法加载图像: {img_name}（格式不支持或文件损坏）")
                else:
                    pixmaps.append(None)
                    print(f"❌ 图像文件不存在: {img_name}")
            
            return pixmaps
        folder_name = "绘图\预测模型"
        folder_path = os.path.join(current_dir, folder_name)
        image_names = ["预测结果.png","预测结果数据.png"]
        # 加载图像
        pixmaps = load_images_to_array(folder_path, image_names)
        
        if pixmaps and len(pixmaps) == 2:
            if hasattr(self.current_window, "label_170"):
                self.current_window.label_170.setPixmap(pixmaps[0].scaled(
                    self.current_window.label_170.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_170 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
            if hasattr(self.current_window, "label_63"):
                self.current_window.label_63.setPixmap(pixmaps[1].scaled(
                    self.current_window.label_63.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_170 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
        else:
            print("❌ 无法生成目标定义图，请检查数据集文件！") 



    # ---------------- 造型优化模块功能 ---------------- #
    #----基于具体频段-----
    def select_folder_and_fill_files(self):
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        folder_path = QFileDialog.getExistingDirectory(None, "选择包含模型和数据的文件夹")
        if not folder_path:
            return

        pth_path = ""

        for file_name in os.listdir(folder_path):
            lower_name = file_name.lower()
            full_path = os.path.join(folder_path, file_name)

            if lower_name.endswith(".pth") and not pth_path:
                pth_path = full_path
            elif file_name == "输入数据.xlsx":
                input_xlsx_path = full_path
            elif file_name == "输出数据.xlsx":
                output_xlsx_path = full_path

        if hasattr(self.current_window, "lineEdit_9"):
            self.current_window.lineEdit_9.setText(pth_path)

        msg = f"📁 已选择文件夹：{folder_path}\n"
        msg += f"\n模型文件 (.pth)：{pth_path if pth_path else '未找到'}"
        QMessageBox.information(None, "文件检测结果", msg)

    def select_file_zxpg_4(self):
        """选择 new_input_path 文件并自动读取原始值、最小值、最大值，填入 lineEdit"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "选择需要优化的造型数据",
            "",
            "Excel 文件 (*.xlsx)"
        )

        if not file_path:
            return

        # 写入 lineEdit_15
        self.current_window.lineEdit_15.setText(file_path)

        # ---------------------- 读取 Excel 并自动填入界面 ---------------------- #
        try:
            import pandas as pd

            df = pd.read_excel(file_path, sheet_name="sheet1")

            required_cols = ["原始值", "最小值", "最大值"]
            if not all(col in df.columns for col in required_cols):
                QMessageBox.warning(
                    None, "格式错误",
                    "Excel sheet1 必须包含 '原始值'、'最小值'、'最大值' 三列！"
                )
                return

            base_params = df['原始值'].values
            param_min = df['最小值'].values
            param_max = df['最大值'].values

            # 转换为原生 python float，避免 np.float64(...) 的字符串
            try:
                param_min_py = [float(x) for x in param_min]
                param_max_py = [float(x) for x in param_max]
                base_params_py = [float(x) for x in base_params]
            except Exception:
                # 如果逐元素转换失败，退回到逐项用 safe 提取
                param_min_py = [self._safe_to_float(str(x)) for x in param_min]
                param_max_py = [self._safe_to_float(str(x)) for x in param_max]
                base_params_py = [self._safe_to_float(str(x)) for x in base_params]

            # 自动识别可调整参数
            adjust_indices = [i for i in range(len(base_params_py)) if param_min_py[i] != param_max_py[i]]

            # ---------------------- 写入 UI（只写入可调整参数的信息） ---------------------- #
            # 索引写成 "0,1,2" 格式，便于后续 parse
            self.current_window.lineEdit_12.setText(", ".join(str(i) for i in adjust_indices))

            # --- 这里是修改的核心部分 ---
            # 根据 adjust_indices 过滤出对应的最小值和最大值
            adjusted_param_min = [param_min_py[i] for i in adjust_indices]
            adjusted_param_max = [param_max_py[i] for i in adjust_indices]

            # 只将可调整参数的最小/最大值写成 "1.0, 2.0, 3.0" 格式
            self.current_window.lineEdit_13.setText(", ".join(str(x) for x in adjusted_param_min))
            self.current_window.lineEdit_14.setText(", ".join(str(x) for x in adjusted_param_max))

            QMessageBox.information(
                None, "读取成功",
                "已成功读取 Excel：\n"
                f"识别到可调整参数个数：{len(adjust_indices)}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(None, "错误", f"读取 Excel 时出错：\n{e}")

    def plot_photo_moxingyouhua_pindian(self):
        """绘制模型预测结果图"""

        # 从文件夹中提取图像
        def load_images_to_array(folder_path, image_names):
            """
            从指定文件夹读取图像并存储到数组中

            Args:
                folder_path (str): 图像文件夹路径
                image_names (list): 要读取的图像文件名列表（最多4个）

            Returns:
                list: 包含QPixmap对象的数组，如果图像不存在则对应位置为None
            """
            # 初始化结果数组
            pixmaps = []

            # 确保image_names是列表且最多包含4个文件名
            if not isinstance(image_names, list):
                raise TypeError("image_names必须是一个列表")

            # 限制为最多4张图像
            image_names = image_names[:4]

            for img_name in image_names:
                # 构建完整的文件路径
                img_path = os.path.join(folder_path, img_name)

                # 检查文件是否存在
                if os.path.exists(img_path):
                    # 创建QPixmap对象
                    pixmap = QPixmap(img_path)

                    # 检查图像是否成功加载
                    if not pixmap.isNull():
                        pixmaps.append(pixmap)
                        print(f"✅ 成功加载图像: {img_name}")
                    else:
                        pixmaps.append(None)
                        print(f"❌ 无法加载图像: {img_name}（格式不支持或文件损坏）")
                else:
                    pixmaps.append(None)
                    print(f"❌ 图像文件不存在: {img_name}")

            return pixmaps

        folder_name = "绘图\优化结果"
        folder_path = os.path.join(current_dir, folder_name)
        image_names = ["频点对比折线图.png", "参数调整对比图.png"]
        # 加载图像
        pixmaps = load_images_to_array(folder_path, image_names)

        if pixmaps and len(pixmaps) == 2:
            if hasattr(self.current_window, "label_35"):
                self.current_window.label_35.setPixmap(pixmaps[0].scaled(
                    self.current_window.label_35.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_35 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
            if hasattr(self.current_window, "label_36"):
                self.current_window.label_36.setPixmap(pixmaps[1].scaled(
                    self.current_window.label_36.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_36 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
        else:
            print("❌ 无法生成目标定义图，请检查数据集文件！")
            
    #----基于整体响度-----
    def select_folder_and_fill_files_xiangdu(self):
        """选择文件夹，自动搜索 .pth、输入数据.xlsx、输出数据.xlsx 并写入相应输入框"""
        folder_path = QFileDialog.getExistingDirectory(None, "选择包含模型和数据的文件夹")
        if not folder_path:
            return

        pth_path = ""
        input_xlsx_path = ""
        output_xlsx_path = ""

        for file_name in os.listdir(folder_path):
            lower_name = file_name.lower()
            full_path = os.path.join(folder_path, file_name)

            if lower_name.endswith(".pth") and not pth_path:
                pth_path = full_path
            elif file_name == "输入数据.xlsx":
                input_xlsx_path = full_path
            elif file_name == "输出数据.xlsx":
                output_xlsx_path = full_path

        if hasattr(self.current_window, "lineEdit_16"):
            self.current_window.lineEdit_16.setText(pth_path)
        # if hasattr(self.current_window, "lineEdit_131"):
        #     self.current_window.lineEdit_131.setText(input_xlsx_path)
        # if hasattr(self.current_window, "lineEdit_132"):
        #     self.current_window.lineEdit_132.setText(output_xlsx_path)

        msg = f"📁 已选择文件夹：{folder_path}\n"
        msg += f"\n模型文件 (.pth)：{pth_path if pth_path else '未找到'}"
        msg += f"\n输入数据.xlsx：{input_xlsx_path if input_xlsx_path else '未找到'}"
        msg += f"\n输出数据.xlsx：{output_xlsx_path if output_xlsx_path else '未找到'}"
        QMessageBox.information(None, "文件检测结果", msg)

    def select_file_zxpg_4_xiangdu(self):
        """选择 new_input_path 文件并自动读取原始值、最小值、最大值，填入 lineEdit"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "选择需要优化的造型数据",
            "",
            "Excel 文件 (*.xlsx)"
        )

        if not file_path:
            return

        # 写入 lineEdit_133
        self.current_window.lineEdit_23.setText(file_path)

        # ---------------------- 读取 Excel 并自动填入界面 ---------------------- #
        try:
            import pandas as pd

            df = pd.read_excel(file_path, sheet_name="sheet1")

            required_cols = ["原始值", "最小值", "最大值"]
            if not all(col in df.columns for col in required_cols):
                QMessageBox.warning(
                    None, "格式错误",
                    "Excel sheet1 必须包含 '原始值'、'最小值'、'最大值' 三列！"
                )
                return

            base_params = df['原始值'].values
            param_min = df['最小值'].values
            param_max = df['最大值'].values

            # 转换为原生 python float，避免 np.float64(...) 的字符串
            try:
                param_min_py = [float(x) for x in param_min]
                param_max_py = [float(x) for x in param_max]
                base_params_py = [float(x) for x in base_params]
            except Exception:
                # 如果逐元素转换失败，退回到逐项用 safe 提取
                param_min_py = [self._safe_to_float(str(x)) for x in param_min]
                param_max_py = [self._safe_to_float(str(x)) for x in param_max]
                base_params_py = [self._safe_to_float(str(x)) for x in base_params]

            # 自动识别可调整参数
            adjust_indices = [i for i in range(len(base_params_py)) if param_min_py[i] != param_max_py[i]]

            # ---------------------- 写入 UI（只写入可调整参数的信息） ---------------------- #
            # 索引写成 "0,1,2" 格式，便于后续 parse
            self.current_window.lineEdit_19.setText(", ".join(str(i) for i in adjust_indices))

            # --- 这里是修改的核心部分 ---
            # 根据 adjust_indices 过滤出对应的最小值和最大值
            adjusted_param_min = [param_min_py[i] for i in adjust_indices]
            adjusted_param_max = [param_max_py[i] for i in adjust_indices]

            # 只将可调整参数的最小/最大值写成 "1.0, 2.0, 3.0" 格式
            self.current_window.lineEdit_20.setText(", ".join(str(x) for x in adjusted_param_min))
            self.current_window.lineEdit_21.setText(", ".join(str(x) for x in adjusted_param_max))

            QMessageBox.information(
                None, "读取成功",
                "已成功读取 Excel：\n"
                f"识别到可调整参数个数：{len(adjust_indices)}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(None, "错误", f"读取 Excel 时出错：\n{e}")
            
    def plot_photo_moxingyouhua(self):
        """绘制模型预测结果图"""
        
        #从文件夹中提取图像
        def load_images_to_array(folder_path, image_names):
            """
            从指定文件夹读取图像并存储到数组中
            
            Args:
                folder_path (str): 图像文件夹路径
                image_names (list): 要读取的图像文件名列表（最多4个）
                
            Returns:
                list: 包含QPixmap对象的数组，如果图像不存在则对应位置为None
            """
            # 初始化结果数组
            pixmaps = []
            
            # 确保image_names是列表且最多包含4个文件名
            if not isinstance(image_names, list):
                raise TypeError("image_names必须是一个列表")
            
            # 限制为最多4张图像
            image_names = image_names[:4]
            
            for img_name in image_names:
                # 构建完整的文件路径
                img_path = os.path.join(folder_path, img_name)
                
                # 检查文件是否存在
                if os.path.exists(img_path):
                    # 创建QPixmap对象
                    pixmap = QPixmap(img_path)
                    
                    # 检查图像是否成功加载
                    if not pixmap.isNull():
                        pixmaps.append(pixmap)
                        print(f"✅ 成功加载图像: {img_name}")
                    else:
                        pixmaps.append(None)
                        print(f"❌ 无法加载图像: {img_name}（格式不支持或文件损坏）")
                else:
                    pixmaps.append(None)
                    print(f"❌ 图像文件不存在: {img_name}")
            
            return pixmaps
        folder_name = "绘图\优化结果"
        folder_path = os.path.join(current_dir, folder_name)
        image_names = ["结果对比响度.png", "参数对比响度.png"]
        # 加载图像
        pixmaps = load_images_to_array(folder_path, image_names)
        
        if pixmaps and len(pixmaps) == 2:
            if hasattr(self.current_window, "label_49"):
                self.current_window.label_49.setPixmap(pixmaps[0].scaled(
                    self.current_window.label_49.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_49 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
            if hasattr(self.current_window, "label_50"):
                self.current_window.label_50.setPixmap(pixmaps[1].scaled(
                    self.current_window.label_50.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            else:
                print("❌ label_50 不存在，请检查 UIXINbuhanbanzidong.ui 文件")
        else:
            print("❌ 无法生成目标定义图，请检查数据集文件！") 




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec())