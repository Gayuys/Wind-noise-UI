import numpy as np
import torch.nn as nn
import torch
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 归一化程序
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize

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

def predict(new_input_data, model, input_mean, input_std):
    
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
#修改归一化
# def load_data_from_excels(input_file_path, output_file_path):
#     input_df = pd.read_excel(input_file_path, header=0)
#     output_df = pd.read_excel(output_file_path, header=0)
#     print("输入文件列名:", list(input_df.columns))
#     print("输出文件列名:", list(output_df.columns))

#     # 标准化相关参数
#     input_mean = input_df.values.mean(axis=0)
#     input_std = input_df.values.std(axis=0)

#     # 按列进行归一化（除了倒数第七列）
#     X = (input_df.values - input_mean) / (input_std + 1e-8)

#     # 获取倒数第7列的索引
#     index_last_seven = -7
#     # 获取该列的数据
#     column_to_normalize = input_df.iloc[:, index_last_seven].values

#     # 将该列归一化到(-1, 1)的范围
#     min_val = column_to_normalize.min()
#     max_val = column_to_normalize.max()
#     column_normalized = 2 * (column_to_normalize - min_val) / (max_val - min_val) - 1

#     # 用归一化后的列替换原始列
#     X[:, index_last_seven] = column_normalized

#     output_mean = output_df.values.mean(axis=0)
#     output_std = output_df.values.std(axis=0)

#     # 处理输出数据的标准化
#     y = (output_df.values - output_mean) / (output_std + 1e-8)

#     # 转为张量
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.float32)
#     return (X_tensor, y_tensor, input_df.columns.tolist(), output_df.columns.tolist(),
#             input_mean, input_std, output_mean, output_std)
#原始归一化
def load_data_from_excels(input_file_path, output_file_path):
    input_df = pd.read_excel(input_file_path, header=0)
    output_df = pd.read_excel(output_file_path, header=0)
    print("输入文件列名:", list(input_df.columns))
    print("输出文件列名:", list(output_df.columns))

    # 标准化相关参数
    input_mean = input_df.values.mean(axis=0)
    input_std = input_df.values.std(axis=0)
    output_mean = output_df.values.mean(axis=0)
    output_std = output_df.values.std(axis=0)

    # 标准化
    X = (input_df.values - input_mean) / (input_std + 1e-8)
    y = (output_df.values - output_mean) / (output_std + 1e-8)

    # 转为张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return (X_tensor, y_tensor, input_df.columns.tolist(), output_df.columns.tolist(),
            input_mean, input_std, output_mean, output_std)
    
# 加载训练后的模型
def call_model(input_file_path, output_file_path, new_input_data, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    # 读取模型输入输出维度
    X, y, input_cols, output_cols, input_mean, input_std, output_mean, output_std = load_data_from_excels(  
            input_file_path,
            output_file_path
        )
    # 初始化模型
    model = AdvancedModel(
            input_dim=60,   # 动态获取输入维度
            output_dim=17   # 动态获取输出维度
        )
        
    model.load_state_dict(torch.load(model_path))

    #原始数据预测
    predictions = predict(new_input_data, model, input_mean, input_std)
    #反归一化
    predicted_data = predictions * output_std + output_mean
    predicted_data = predicted_data.squeeze()
    
    return predicted_data

if __name__ == "__main__":
    # # 获取当前Python文件所在的目录
    current_dir = 'E:/一汽风噪/模型预测/0108训练模型' #模型及数据集位置
    current_dir2 = 'E:/一汽风噪/模型预测' #验证集位置
    print(f"程序所在目录: {current_dir}")

    input_file_name = "处理后有噪声扩充造型数据6-8-18-24-31-32-33添加玻璃强化.xlsx" #获取原始数据集，不需要改变
    output_file_name = "有噪声扩充风噪数据6-8-18-24-31-32-33添加玻璃强化.xlsx" #获取原始数据集，不需要改变
    input_file_path = os.path.join(current_dir, input_file_name)
    output_file_path = os.path.join(current_dir, output_file_name)
    new_input_data_name = "造型数据CAR1修改玻璃.xlsx" #验证集的造型加技术方案
    new_input_data_path = os.path.join(current_dir2, new_input_data_name)
    new_input_data = pd.read_excel(new_input_data_path).values
    # 加载保存的模型权重
    model_name = 'best_model.pth'
    model_path = os.path.join(current_dir, model_name)
    save_path = current_dir2
    predicted_data = call_model(input_file_path, output_file_path, new_input_data, model_path)
    print(predicted_data)


