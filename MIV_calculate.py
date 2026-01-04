import numpy as np
import torch.nn as nn
import torch
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 归一化程序
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# -------------------------- 路径配置 --------------------------
# 定义项目根目录（相对于当前脚本的位置）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义各子目录的相对路径
SAVE_DIR = os.path.join(BASE_DIR, '缓存')     # 结果保存目录

# 创建目录（如果不存在）
for dir_path in [SAVE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
# 设置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
  

# 深度可分离卷积
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv1d, self).__init__()
        
        # 深度可分离卷积
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Xception 块
class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(XceptionBlock, self).__init__()
        
        # 主分支
        self.separable_conv1 = SeparableConv1d(in_channels, out_channels, stride=stride)
        self.separable_conv2 = SeparableConv1d(out_channels, out_channels)
        
        # 下采样分支
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        
        return x + residual

# Xception 模型
class AdvancedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 输入层
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Xception 块
        self.xception_blocks = nn.Sequential(
            XceptionBlock(32, 64),
            XceptionBlock(64, 128),
            XceptionBlock(128, 256)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # 调整输入维度 (batch_size, features) -> (batch_size, 1, features)
        x = x.unsqueeze(1)
        
        # 前向传播
        x = self.input_conv(x)
        x = self.xception_blocks(x)
        x = self.global_pool(x)
        
        # 展平
        x = x.squeeze(-1)
        x = self.fc(x)
        
        return x

def predict(model_path, new_input_data, input_cols, input_mean, input_std):
    # 加载最佳模型
    model = AdvancedModel(
        input_dim=len(input_cols),
        output_dim=17
    )
    model.load_state_dict(torch.load(model_path))
    
    # 确定模型设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 标准化输入数据
    new_input_data = (new_input_data - input_mean) / (input_std + 1e-8)
    
    # 转换为张量并移到与模型相同的设备
    input_tensor = torch.tensor(new_input_data, dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        predictions = model(input_tensor)
    
    return predictions.cpu().numpy()  # 将结果移回CPU并转为numpy数组

# 数据导入函数
def load_data_from_excels(input_file_path, output_file_path):
    # 读取输入Excel文件
    input_df = pd.read_excel(input_file_path, header=0)
    
    # 读取输出Excel文件
    output_df = pd.read_excel(output_file_path, header=0)
    
    # 打印列名以供参考
    print("输入文件列名:", list(input_df.columns))
    print("输出文件列名:", list(output_df.columns))
    
    # 保存原始数据的均值和标准差
    input_mean = input_df.values.mean(axis=0)
    input_std = input_df.values.std(axis=0)
    output_mean = output_df.values.mean(axis=0)
    output_std = output_df.values.std(axis=0)
    
    # 标准化处理
    X = (input_df.values - input_mean) / (input_std + 1e-8)
    y = (output_df.values - output_mean) / (output_std + 1e-8)
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return (X_tensor, y_tensor,
            input_df.columns.tolist(),
            output_df.columns.tolist(),
            input_mean, input_std,
            output_mean, output_std)

def MIV_analysis(model_path, X,  origin_x, input_cols, output_cols, input_mean, input_std, output_mean, output_std):
    """wq
    修复版 MIV 分析：传入 cols 等参数，支持多输出
    X为进行MIV分析的输入数据
    origin_x为原始数据集，用于计算变化造型参数
    """
      
    X = np.asarray(X)
    #y = np.asarray(y)
    m, n = X.shape  # m=1, n=41
    k = len(output_cols)  # 17
    print(f"调试: m={m}, n={n}, k={k}")
    
    outputs_up1 = []
    outputs_down1 = []
    predicted_data = []
    outputs = []

    #变化造型参数+技术方案参数
    for i in range(53):
        max_data = origin_x.iloc[:, i].max()
        min_data = origin_x.iloc[:, i].min()
        # 生成改变的值
        change = (max_data - min_data)*0.2
     
        #计算原始方案预测值
        output = predict(model_path, X, input_cols, input_mean, input_std)
        predicted_data = output * output_std + output_mean
        predicted_data = predicted_data.squeeze()  # (1,17)

        
        # 上调
        X_up = X.copy()    
        X_up[:, i] = X_up[:, i] + change       
        output_up = predict(model_path, X_up, input_cols, input_mean, input_std)
        predicted_data_up = output_up * output_std + output_mean
        predicted_data_up = predicted_data_up.squeeze()  # (1,17)
        
        # 下调
        X_down = X.copy()
        X_down[:, i] = X_down[:, i] - change
        output_down = predict(model_path, X_down, input_cols, input_mean, input_std)
        predicted_data_down = output_down * output_std + output_mean
        predicted_data_down = predicted_data_down.squeeze()  # (1,17)
        
        outputs.append(predicted_data)
        outputs_up1.append(predicted_data_up)
        outputs_down1.append(predicted_data_down)

    outputs_up1 = np.array(outputs_up1)  
    outputs_down1 = np.array(outputs_down1)
    outputs = np.array(outputs)  
    
    print(f"调试: outputs_up 形状 {outputs_up1.shape}, outputs_down 形状 {outputs_down1.shape}")

    
    # IV: 差异 (60,1,17)
    IV1 = abs(outputs_up1 - outputs)
    IV2 = abs(outputs - outputs_down1) 

    
    print(f"调试: IV_absnew 形状 {IV1.shape}")    
    
    
    # MIV: 平均过样本 (41,17)
    MIV = abs(outputs_up1 - outputs_down1)
    
    print(f"调试: MIV1 形状 {MIV.shape}")

    return MIV, IV1, IV2

def save_result(MIV, IV1, IV2, Characteristic_name,save_dir=SAVE_DIR):
    """
    保存 MIV, IV1, IV2 到指定目录
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    #定义频点
    freq_labels = ["200Hz", "250Hz", "315Hz", "400Hz", "500Hz", "630Hz", "800Hz", "1000Hz", "1250Hz",
                   "1600Hz", "2000Hz", "2500Hz", "3150Hz", "4000Hz", "5000Hz", "6300Hz", "8000Hz"]
    #读取技术方案名称
    file_path = Characteristic_name #获取技术方案名称
    data = pd.read_excel(file_path, header=0)  # 第一行作为列名
    labels = data.columns.tolist()
    #保存MIV数组
    iv_save_name = 'MIV数组.xlsx'
    miv_save_path = os.path.join(save_dir, iv_save_name)
    miv = MIV.T
    corr_df = pd.DataFrame(miv, index=freq_labels, columns=labels) # 创建DataFrame用于绘图
    corr_df_reversed = corr_df[::-1] #数据取反，符合频点从小到大排序
    corr_df_reversed.to_excel(miv_save_path, index=True)
    #保存IV1数组(增大)
    iv_save_name = 'IV1数组.xlsx'
    iv1_save_path = os.path.join(save_dir, iv_save_name)
    iv1 = IV1.T
    corr_df = pd.DataFrame(iv1, index=freq_labels, columns=labels) # 创建DataFrame用于绘图
    corr_df_reversed = corr_df[::-1] #数据取反，符合频点从小到大排序
    corr_df_reversed.to_excel(iv1_save_path, index=True)
    #保存IV2数组(减小)
    iv_save_name = 'IV2数组.xlsx'
    iv2_save_path = os.path.join(save_dir, iv_save_name)
    iv2 = IV2.T
    corr_df = pd.DataFrame(iv2, index=freq_labels, columns=labels) # 创建DataFrame用于绘图
    corr_df_reversed = corr_df[::-1] #数据取反，符合频点从小到大排序
    corr_df_reversed.to_excel(iv2_save_path, index=True)
    print(f"MIV数组已保存到: {miv_save_path}")
    print(f"IV1数组已保存到: {iv1_save_path}")
    print(f"IV2数组已保存到: {iv2_save_path}")

def calculate_result(input_file_path, output_file_path, new_input_path, model_path,Characteristic_name,save_dir=SAVE_DIR):

    X, y, input_cols, output_cols, input_mean, input_std, output_mean, output_std = load_data_from_excels(
            input_file_path,
            output_file_path
        )
    input_df = pd.read_excel(input_file_path, header=0) #读取输入数据
    # 读取样本
    new_input_data = pd.read_excel(new_input_path).values
    # ===== 计算 MIV =====（添加 cols 等参数）
    MIV, IV_new1, IV_new2 = MIV_analysis(model_path, new_input_data, input_df, input_cols, output_cols, input_mean, input_std, output_mean, output_std)
    save_result(MIV, IV_new1, IV_new2, Characteristic_name, save_dir)
    return MIV, IV_new1, IV_new2
# 加载训练后的模型
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_file_path = 'E:/一汽风噪/灵敏度分析/处理后有噪声扩充造型数据6-8-18-24-31-32-33.xlsx'   # 输入数据文件
    output_file_path = 'E:/一汽风噪/灵敏度分析/有噪声扩充风噪数据6-8-18-24-31-32-33.xlsx' # 输出数据文件 
    new_input_path = 'E:/一汽风噪/灵敏度分析/造型数据CAR32.xlsx'
    model_path = 'best_model1111.pth'
    Characteristic_name = 'E:/一汽风噪/数据扩充/新版数据集/造型特征+技术方案名称.xlsx' #获取技术方案名称
    save_dir = 'E:/一汽风噪/灵敏度分析/分析结果'
    Characteristic_name2 = 'E:/一汽风噪/UI界面/data/造型优化特征.xlsx'
    name = pd.read_excel(Characteristic_name2, header=0)
    param_names = name.columns.tolist()  # 获取所有列名
    #name = np.array(name).reshape(-1, 1)
    new_input_data = pd.read_excel(new_input_path)
    data = new_input_data.iloc[0, :].values
    new_data = data.T
    # 定义列名
    #columns = ["参数名称", "原始值", "最小值", "最大值"]
    df = pd.DataFrame(param_names, columns=['参数名称'])
    df['原始值'] = new_data
    df['最小值'] = new_data
    df['最大值'] = new_data
    save_path = "优化方案.xlsx"
    df.to_excel(save_path, index=False, engine="openpyxl")
    print(f"Excel文件已生成，保存路径：{save_path}")
    

    print(name.shape)
    #MIV, IV_new1, IV_new2 = calculate_result(input_file_path, output_file_path, new_input_path, model_path,Characteristic_name,save_dir)
 


# # ====================== 修正：基于全频段热力图列数据求和+从大到小排序取前十（修复列名不一致问题） ======================
# def sum_and_rank_params_from_heatmap(MIV, param_labels, freq_labels, save_path):
#     """
#     基于全频段热力图数据，计算每个参数对应的17个频点MIV数据之和，按从大到小排序取前十
#     :param MIV: 灵敏度矩阵（形状：参数数×17频点）
#     :param param_labels: 参数名称列表（对应热力图的列）
#     :param freq_labels: 频点名称列表（对应热力图的行）
#     :param save_path: 结果保存路径
#     """
#     # 1. 构建与热力图一致的DataFrame（参数×频点）
#     # 截断参数名称，确保与MIV行数一致
#     heatmap_df = pd.DataFrame(MIV,
#                               index=param_labels[:MIV.shape[0]],  # 防止参数名称数量与MIV行数不匹配
#                               columns=freq_labels)  # 列：频点（对应热力图的行）

#     # 2. 计算每个参数的17个频点MIV数据之和（对每个参数行求和）
#     param_total = heatmap_df.sum(axis=1)  # axis=1：对行求和（每个参数的17个频点）

#     # 3. 组合参数名称与对应总和（列名保持一致）
#     param_sum_df = pd.DataFrame({
#         "参数名称": param_total.index,
#         "17频点MIV总和（上调-下调噪声差值绝对值之和）": param_total.values  # 统一列名
#     })

#     # 4. 按总和【从大到小】排序，取前十名（列名与上面一致，修复KeyError）
#     param_sum_sorted = param_sum_df.sort_values(
#         by="17频点MIV总和（上调-下调噪声差值绝对值之和）",
#         ascending=False  # 改为False，实现从大到小排序
#     ).head(10)

#     # 5. 打印结果
#     print("=" * 80)
#     print("全频段热力图-每个参数17个频点MIV总和（从大到小排序，取前十）")
#     print("=" * 80)
#     print(param_sum_sorted)
#     print("=" * 80)

#     # 6. 保存到Excel（增加目录创建和异常捕获）
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     try:
#         with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
#             param_sum_sorted.to_excel(writer, index=False)
#         print(f"✅ 参数MIV总和排序结果已保存到：{save_path}")
#     except PermissionError:
#         print(f"❌ 权限错误：无法写入 {save_path}，请关闭该文件后重试")
#     except Exception as e:
#         print(f"❌ 保存失败：{str(e)}")

#     return param_sum_sorted


# # ====================== 修正：基于全频段热力图列数据求和+从大到小排序取前十（已修复列名问题） ======================
# def sum_and_rank_params_from_heatmap(MIV, param_labels, freq_labels, save_path):
#     """
#     基于全频段热力图数据，计算每个参数对应的17个频点MIV数据之和，按从大到小排序取前十
#     :param MIV: 灵敏度矩阵（形状：参数数×17频点）
#     :param param_labels: 参数名称列表（对应热力图的列）
#     :param freq_labels: 频点名称列表（对应热力图的行）
#     :param save_path: 结果保存路径
#     """
#     # 1. 构建与热力图一致的DataFrame（参数×频点）
#     # 截断参数名称，确保与MIV行数一致
#     heatmap_df = pd.DataFrame(MIV,
#                               index=param_labels[:MIV.shape[0]],  # 防止参数名称数量与MIV行数不匹配
#                               columns=freq_labels)  # 列：频点（对应热力图的行）

#     # 2. 计算每个参数的17个频点MIV数据之和（对每个参数行求和）
#     param_total = heatmap_df.sum(axis=1)  # axis=1：对行求和（每个参数的17个频点）

#     # 3. 组合参数名称与对应总和（列名保持一致）
#     param_sum_df = pd.DataFrame({
#         "参数名称": param_total.index,
#         "17频点MIV总和（上调-下调噪声差值绝对值之和）": param_total.values  # 统一列名
#     })

#     # 4. 按总和【从大到小】排序，取前十名（列名与上面一致，修复KeyError）
#     param_sum_sorted = param_sum_df.sort_values(
#         by="17频点MIV总和（上调-下调噪声差值绝对值之和）",
#         ascending=False  # 改为False，实现从大到小排序
#     ).head(10)

#     # 5. 打印结果
#     print("=" * 80)
#     print("全频段热力图-每个参数17个频点MIV总和（从大到小排序，取前十）")
#     print("=" * 80)
#     print(param_sum_sorted)
#     print("=" * 80)

#     # 6. 保存到Excel（增加目录创建和异常捕获）
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     try:
#         with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
#             param_sum_sorted.to_excel(writer, index=False)
#         print(f"✅ 参数MIV总和排序结果已保存到：{save_path}")
#     except PermissionError:
#         print(f"❌ 权限错误：无法写入 {save_path}，请关闭该文件后重试")
#     except Exception as e:
#         print(f"❌ 保存失败：{str(e)}")

#     return param_sum_sorted


# # ====================== 调整：提取最值保留两位小数并直接覆盖原优化文件 ======================
# def match_params_and_fill_min_max(top10_params_path,
#                                   source_data_path,
#                                   optimize_data_path,
#                                   param_name_path):
#     """
#     1. 读取前十参数列表
#     2. 匹配参数对应的序号
#     3. 从源数据中提取对应序号的最小值和最大值（保留两位小数）
#     4. 直接覆盖回填到原需要优化的造型数据Excel中
#     :param top10_params_path: 前十参数保存路径
#     :param source_data_path: 源数据文件（处理后有噪声扩充造型数据...xlsx）
#     :param optimize_data_path: 需要优化的造型数据文件路径（直接覆盖此文件）
#     :param param_name_path: 造型特征+技术方案名称文件路径
#     """
#     # 1. 读取各文件（增加异常捕获）
#     try:
#         # 读取前十参数
#         top10_df = pd.read_excel(top10_params_path)
#         top10_param_names = top10_df["参数名称"].tolist()
#         print(f"\n读取到前十参数：{top10_param_names}")

#         # 读取参数名称列表（获取参数序号）
#         param_name_df = pd.read_excel(param_name_path, header=0)
#         all_param_names = param_name_df.columns.tolist()
#         print(f"总参数数量：{len(all_param_names)}")

#         # 读取源数据（提取最值）
#         source_df = pd.read_excel(source_data_path, header=0)
#         source_cols = source_df.columns.tolist()
#         print(f"源数据列数量：{len(source_cols)}")

#         # 读取需要优化的造型数据（原文件）
#         optimize_df = pd.read_excel(optimize_data_path, header=0)
#         # 确保优化文件有“最小值”“最大值”列（无则创建）
#         if "最小值" not in optimize_df.columns:
#             optimize_df["最小值"] = np.nan
#         if "最大值" not in optimize_df.columns:
#             optimize_df["最大值"] = np.nan
#         print(f"需要优化的造型数据列名：{optimize_df.columns.tolist()}")

#     except FileNotFoundError as e:
#         print(f"❌ 文件未找到：{str(e)}")
#         return None
#     except Exception as e:
#         print(f"❌ 读取文件失败：{str(e)}")
#         return None

#     # 2. 匹配前十参数对应的序号并提取最值（保留两位小数）
#     for param_name in top10_param_names:
#         # 匹配参数在总参数列表中的序号（索引）
#         if param_name in all_param_names:
#             param_col_index = all_param_names.index(param_name)
#             # 确保序号不超过源数据列数
#             if param_col_index < len(source_cols):
#                 param_col_name = source_cols[param_col_index]  # 对应源数据的列名
#                 print(f"\n匹配到参数：{param_name}，序号：{param_col_index}，列名：{param_col_name}")

#                 # 提取源数据中该列的最小值和最大值，并保留两位小数
#                 param_min = round(source_df[param_col_name].min(), 2)  # 保留两位小数
#                 param_max = round(source_df[param_col_name].max(), 2)  # 保留两位小数
#                 print(f"  对应最小值：{param_min}，最大值：{param_max}")

#                 # 回填到优化数据中
#                 if "参数名称" in optimize_df.columns:
#                     # 按参数名称匹配回填（优先方案）
#                     optimize_df.loc[optimize_df["参数名称"] == param_name, "最小值"] = param_min
#                     optimize_df.loc[optimize_df["参数名称"] == param_name, "最大值"] = param_max
#                 else:
#                     # 按参数序号匹配（假设优化文件行顺序与参数序号一致）
#                     param_row_index = all_param_names.index(param_name)
#                     if param_row_index < len(optimize_df):
#                         optimize_df.loc[param_row_index, "最小值"] = param_min
#                         optimize_df.loc[param_row_index, "最大值"] = param_max
#                     else:
#                         print(f"警告：参数 {param_name} 序号 {param_row_index} 超出优化文件行数")
#             else:
#                 print(f"警告：参数 {param_name} 序号 {param_col_index} 超出源数据列数")
#         else:
#             print(f"警告：参数 {param_name} 未在总参数列表中找到，跳过")

#     # 3. 直接覆盖原文件保存（核心调整：无新文件，直接写入原路径）
#     try:
#         # 先关闭可能占用文件的句柄，再写入
#         with pd.ExcelWriter(optimize_data_path, engine='openpyxl', mode='w') as writer:
#             optimize_df.to_excel(writer, index=False)
#         print(f"\n✅ 已直接覆盖原文件：{optimize_data_path}")
#         print(f"✅ 最值（保留两位小数）回填完成，原文件数据已更新")
#     except PermissionError:
#         print(f"❌ 权限错误：无法覆盖 {optimize_data_path}，请先关闭该Excel文件")
#     except Exception as e:
#         print(f"❌ 覆盖文件失败：{str(e)}")

#     return optimize_df


# # ====================== 调用函数 ======================
# # 1. 先执行MIV总和排序
# sum_rank_save_path = os.path.join(save_dir, "全频段参数MIV总和_前十.xlsx")
# os.makedirs(os.path.dirname(sum_rank_save_path), exist_ok=True)
# param_rank_result = sum_and_rank_params_from_heatmap(
#     MIV=MIV,
#     param_labels=labels,
#     freq_labels=freq_labels,
#     save_path=sum_rank_save_path
# )

# # 2. 再执行最值回填（直接覆盖原文件，最值保留两位小数）
# # 定义各文件路径
# top10_params_path = sum_rank_save_path
# source_data_path = 'F:\python第一课\灵敏度分析\处理后有噪声扩充造型数据6-8-18-24-31-32-33.xlsx'
# optimize_data_path = 'F:\python第一课\灵敏度分析\需要优化的造型数据.xlsx'  # 原文件路径（直接覆盖）
# param_name_path = 'F:\python第一课\灵敏度分析\造型特征+技术方案名称.xlsx'

# # 执行回填（无额外保存路径，直接覆盖原文件）
# optimize_df_filled = match_params_and_fill_min_max(
#     top10_params_path=top10_params_path,
#     source_data_path=source_data_path,
#     optimize_data_path=optimize_data_path,
#     param_name_path=param_name_path
# )