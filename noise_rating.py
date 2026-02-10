import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 解决OpenMP冲突（必须保留，防止模型加载报错）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 绘图中文配置（解决中文乱码、负号显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. 模型结构（修复squeeze语法错误） --------------------------
class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv1d, self).__init__()
        self.depthwise = torch.nn.Conv1d(in_channels, in_channels, kernel_size,
                                         stride, padding, groups=in_channels)
        self.pointwise = torch.nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class XceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(XceptionBlock, self).__init__()
        self.separable_conv1 = SeparableConv1d(in_channels, out_channels, stride=stride)
        self.separable_conv2 = SeparableConv1d(out_channels, out_channels)
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            torch.nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        return x + residual


class AdvancedModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )
        self.xception_blocks = torch.nn.Sequential(
            XceptionBlock(32, 64),
            XceptionBlock(64, 128),
            XceptionBlock(128, 256)
        )
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 特征数) -> (batch, 1, 特征数) 适配1D卷积
        x = self.input_conv(x)
        x = self.xception_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # 修复：将self.squeeze改为x.squeeze
        x = self.fc(x)
        return x


# -------------------------- 2. 核心工具函数（预测+MAPE+绘图） --------------------------
def load_normalization_params(input_file_path, output_file_path):
    """加载标准化参数（均值/标准差），返回参数+17个频点标签"""
    input_df = pd.read_excel(input_file_path, header=0)
    output_df = pd.read_excel(output_file_path, header=0)
    # 计算输入（造型参数）、输出（噪声值）的标准化参数
    input_mean = input_df.values.mean(axis=0)
    input_std = input_df.values.std(axis=0)
    output_mean = output_df.values.mean(axis=0)
    output_std = output_df.values.std(axis=0)
    # 固定17个频点标签（200-8000Hz，和模型输出匹配）
    freq_labels = [200, 250, 315, 400, 500, 630, 800,
                   1000, 1250, 1600, 2000, 2500, 3150,
                   4000, 5000, 6300, 8000]
    return input_mean, input_std, output_mean, output_std, freq_labels


def load_trained_model(model_path, input_dim, output_dim):
    """加载训练好的pth模型，自动适配CUDA/CPU，设置为评估模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedModel(input_dim=input_dim, output_dim=output_dim)
    # 加载模型权重，忽略设备差异，避免权重加载报错
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # 关键：评估模式，关闭Dropout/BatchNorm训练特性
    print(f"✅ 模型加载成功，使用设备：{device}")
    return model, device


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差MAPE，过滤0值防止除0错误，保留2位小数"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 过滤真实值为0的情况，避免除以0报错
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0
    # 计算MAPE并转百分比
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    return np.round(mape, 2)


def calculate_noise_score(mape):
    """根据MAPE计算噪声曲线得分，公式：(1 - MAPE/100)² × 100，保留2位小数"""
    # 防止MAPE超过100导致得分负数，做边界处理
    mape = min(mape, 100.0)
    score = (1 - mape / 100) ** 2 * 100
    return np.round(score, 2)


def plot_pred_vs_target(freq_labels, y_pred, y_true, mape, score, save_path):
    """绘制预测值与目标值对比折线图（横坐标均匀展示），左上角标注MAPE和得分，高清保存"""
    plt.figure(figsize=(16, 8))  # 加宽画布，适配17个均匀标签
    # 生成均匀的横坐标位置（0-16），替代原频率数值
    x_pos = range(len(freq_labels))

    # 绘制折线：红色目标值，蓝色预测值，加粗+标记点（用均匀x_pos绘图）
    plt.plot(x_pos, y_true, 'ro-', linewidth=2.5, markersize=7, label='目标噪声值')
    plt.plot(x_pos, y_pred, 'bo-', linewidth=2.5, markersize=7, label='模型预测值')

    # 左上角标注MAPE和噪声曲线得分，带绿色背景框，醒目易看
    plt.text(0.02, 0.98, f'MAPE = {mape}% | 噪声曲线得分 = {score}', transform=plt.gca().transAxes,
             fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
             verticalalignment='top')

    # 核心修改：设置均匀的横坐标标签（显示原频率值，位置均匀）
    plt.xticks(x_pos, freq_labels, rotation=45, fontsize=11)  # x_pos是均匀位置，标签是原频率值
    plt.yticks(fontsize=12)
    plt.xlabel('频率(Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('噪声值(dB)', fontsize=14, fontweight='bold')
    plt.title('200-8000Hz噪声值：模型预测值 vs 目标值', fontsize=16, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')  # 浅灰色网格，辅助对比
    plt.tight_layout()  # 自适应布局，防止标签截断

    # 高清保存图片（300dpi），关闭画布释放内存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测vs目标对比图已保存至：{save_path}")


# -------------------------- 3. 主预测函数（核心：适配横向无表头Excel读取+横向保存） --------------------------
def calculate_rating(MODEL_PATH, INPUT_FILE, OUTPUT_FILE, PARAM_FILE_PATH, TARGET_FILE_PATH):
    # ====================== 请确认你的文件路径（仅修改这部分，其他不动） ======================
    # MODEL_PATH 训练好的模型.pth路径
    # INPUT_FILE 训练用输入数据（求标准化参数）
    # OUTPUT_FILE 训练用输出数据（求标准化参数）
    # PARAM_FILE_PATH 造型特征参数Excel
    # TARGET_FILE_PATH 目标噪声Excel
    # RESULT_SAVE_PATH 预测结果保存
    # PLOT_SAVE_PATH 对比图保存路径
    # ======================================================================================

    # 步骤1：加载标准化参数和17个频点标签
    print("🔹 加载数据标准化参数和频点标签...")
    input_mean, input_std, output_mean, output_std, freq_labels = load_normalization_params(INPUT_FILE, OUTPUT_FILE)
    input_dim = len(input_mean)  # 模型输入维度（25个造型参数）
    output_dim = len(output_mean)  # 模型输出维度（17个噪声频点）
    print(f"✅ 模型输入维度（造型参数数）：{input_dim} | 模型输出维度（噪声频点数）：{output_dim}")

    # 步骤2：加载造型参数
    print("\n🔹 加载造型参数（横向无表头格式）...")
    # header=None：不读取表头；usecols=None：读取所有列；iloc[0]：读取第一行的所有数值
    param_df = pd.read_excel(PARAM_FILE_PATH, sheet_name='Sheet1', header=0, engine='openpyxl')
    # 读取第一行的所有数值，过滤空值，转为浮点型
    pred_params = param_df.iloc[0].dropna().values.astype(np.float32)
    # 校验参数数量
    if len(pred_params) != input_dim:
        raise ValueError(
            f"造型参数个数({len(pred_params)})与模型输入维度({input_dim})不匹配！请检查Excel第一行的数值数量")
    print(f"✅ 造型参数加载完成（共{len(pred_params)}个）")

    # 步骤3：加载目标噪声值
    print("\n🔹 加载目标噪声值（横向无表头格式）...")
    # header=None：不读取表头；usecols=None：读取所有列；iloc[0]：读取第一行的所有数值
    target_df = pd.read_excel(TARGET_FILE_PATH, sheet_name='Sheet1', header=0, engine='openpyxl')
    # 读取第一行的所有数值，过滤空值，转为浮点型
    target_noise = target_df.iloc[0].dropna().values.astype(np.float32)
    # 校验噪声值数量
    if len(target_noise) != output_dim:
        raise ValueError(
            f"目标噪声值个数({len(target_noise)})与模型输出维度({output_dim})不匹配！请检查Excel第一行的数值数量")
    print(f"✅ 目标噪声值加载完成（共{len(target_noise)}个频点）")

    # 步骤4：加载训练好的模型
    print("\n🔹 加载训练好的预测模型...")
    model, device = load_trained_model(MODEL_PATH, input_dim, output_dim)

    # 步骤5：核心预测（无梯度计算，提升速度，避免显存占用）
    print("\n🔹 开始模型预测...")
    with torch.no_grad():
        # 输入数据归一化（和训练时保持一致，+1e-8防止除0）
        pred_params_norm = (pred_params - input_mean) / (input_std + 1e-8)
        # 转为torch张量，适配模型输入格式：(1, 25) → 加batch维度
        pred_tensor = torch.tensor(pred_params_norm, dtype=torch.float32).to(device).unsqueeze(0)
        # 模型预测（输出归一化的噪声值）
        pred_norm = model(pred_tensor)
        # 反归一化，得到真实的噪声dB值，并展平为1维数组
        pred_noise = (pred_norm.cpu().numpy() * output_std + output_mean).flatten()
    # 预测值保留2位小数，符合工程精度
    pred_noise = np.round(pred_noise, 2)
    print(f"✅ 预测完成，输出{len(pred_noise)}个频点的噪声预测值")

    # 步骤6：计算预测值与目标值的MAPE误差 + 噪声曲线得分
    print("\n🔹 计算预测误差MAPE和噪声曲线得分...")
    mape = calculate_mape(target_noise, pred_noise)
    noise_score = calculate_noise_score(mape)  # 新增：计算噪声曲线得分
    print(f"✅ 模型预测值与目标值的MAPE：{mape}%")
    print(f"✅ 噪声曲线得分：{noise_score}")  # 新增：打印得分

    # 步骤7：绘制并保存预测值-目标值对比折线图（传入得分，在图中显示）
    print("\n🔹 绘制并保存对比折线图...")
    #plot_pred_vs_target(freq_labels, pred_noise, target_noise, mape, noise_score, PLOT_SAVE_PATH)

    print(f"📈 噪声曲线得分={noise_score}")
    print("=" * 60)
    return freq_labels, pred_noise, target_noise, mape, noise_score


# -------------------------- 运行入口 + 依赖检查 --------------------------
if __name__ == '__main__':
    # 自动检查必备依赖，缺少则提示安装
    try:
        import openpyxl  # Excel读写依赖
    except ModuleNotFoundError:
        print("❌ 缺少Excel读写依赖，请先执行命令安装：pip install openpyxl")
    else:

        # ====================== 请确认你的文件路径（仅修改这部分，其他不动） ======================
        MODEL_PATH = r'E:\一汽风噪\造型符合度评分\best_model.pth'  # 训练好的模型.pth路径
        INPUT_FILE = r'E:\一汽风噪\造型符合度评分\输入数据.xlsx'  # 训练用输入数据（求标准化参数）
        OUTPUT_FILE = r'E:\一汽风噪\造型符合度评分\输出数据.xlsx'  # 训练用输出数据（求标准化参数）
        PARAM_FILE_PATH = r'E:\一汽风噪\造型符合度评分\造型特征参数.xlsx'  # 造型特征参数Excel
        TARGET_FILE_PATH = r'E:\一汽风噪\造型符合度评分\目标噪声值.xlsx'  # 目标噪声Excel
        RESULT_SAVE_PATH = r'E:\一汽风噪\造型符合度评分\噪声预测结果.xlsx'  # 预测结果保存
        PLOT_SAVE_PATH = r'E:\一汽风噪\造型符合度评分\预测值_目标值对比图.png'  # 对比图保存路径
        # ======================================================================================
        # 执行主预测函数
        param_df = pd.read_excel(PARAM_FILE_PATH, sheet_name='Sheet1', header=None, engine='openpyxl')
        freq_labels, pred_noise, target_noise, mape, noise_score = calculate_rating(MODEL_PATH, INPUT_FILE, OUTPUT_FILE, PARAM_FILE_PATH, TARGET_FILE_PATH)