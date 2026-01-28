import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import qmc  # 引入拉丁超立方采样库


# ==========================================
# 1. 模型结构 (保持不变)
# ==========================================
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
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        x = self.xception_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# ==========================================
# 2. 工具函数
# ==========================================
def load_normalization_params(input_file_path, output_file_path):
    """加载用于标准化的均值和标准差参数"""
    input_df = pd.read_excel(input_file_path, header=0)
    output_df = pd.read_excel(output_file_path, header=0)

    input_mean = input_df.values.mean(axis=0)
    input_std = input_df.values.std(axis=0)
    output_mean = output_df.values.mean(axis=0)
    output_std = output_df.values.std(axis=0)

    return (input_df.columns.tolist(), output_df.columns.tolist(),
            input_mean, input_std, output_mean, output_std)


def load_original_data(original_data_path):
    """从Excel文件加载原始数据（目标数据）"""
    try:
        df = pd.read_excel(original_data_path, header=None)
        if len(df) < 2:
            raise ValueError("Excel文件至少需要包含2行数据")

        # 提取第二行（索引为1）的1-17列
        original_data = df.iloc[1, 0:17].values
        original_data = original_data.astype(float)
        print(f"成功加载目标原始数据，长度: {len(original_data)}")
        return original_data
    except Exception as e:
        print(f"加载原始数据失败: {str(e)}")
        return None


def generate_lhs_samples(input_file_path, n_samples=1000):
    """
    使用拉丁超立方生成样本
    特性：
    1. 自动适配 Excel 第2、3行作为范围
    2. 自动纠错（处理固定值）
    3. 【新增】强制最后 7 个特征只取值 1.0 或 2.0
    """
    #计算数据库中的最大最小值
    df = pd.read_excel(input_file_path, header=0)
    feature_names = df.columns.tolist()
    n_dims = len(feature_names)
    lower_bounds = df.min(axis=0)
    upper_bounds = df.max(axis=0)
    print(lower_bounds)
    print(upper_bounds)
    
    # # 读取 Excel
    # df = pd.read_excel(range_file_path, header=0)
    # if len(df) < 2:
    #     raise ValueError("Excel文件数据不足，至少需要2行数据（范围）。")

    # feature_names = df.columns.tolist()
    # n_dims = len(feature_names)

    # # -------------------------------------------------------
    # # 1. 准备常规特征的范围 (为了不报错，先处理所有列的范围)
    # # -------------------------------------------------------
    # try:
    #     row1 = df.iloc[0].values.astype(float)
    #     row2 = df.iloc[1].values.astype(float)
    # except ValueError:
    #     raise ValueError("Excel数据包含非数字字符，请检查。")

    # lower_bounds = np.minimum(row1, row2)
    # upper_bounds = np.maximum(row1, row2)

    # 处理固定值 (防止 Scipy 报错)
    diff = upper_bounds - lower_bounds 
    is_constant = diff < 1e-9
    if np.any(is_constant):
        upper_bounds[is_constant] += 1e-5

    # -------------------------------------------------------
    # 2. 生成基础样本 (0到1之间)
    # -------------------------------------------------------
    print(f"正在生成样本... (总特征数: {n_dims})")
    sampler = qmc.LatinHypercube(d=n_dims)
    sample = sampler.random(n=n_samples)  # 这里得到的是 0-1 的均匀分布

    # -------------------------------------------------------
    # 3. 缩放样本 (连续变量)
    # -------------------------------------------------------
    # 先把所有特征都按连续变量处理缩放一遍
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)

    # 还原固定值
    if np.any(is_constant):
        for i in range(n_dims):
            if is_constant[i]:
                scaled_samples[:, i] = lower_bounds[i]

    # -------------------------------------------------------
    # 4. 【核心修改】强制最后 7 个特征离散化 (只取 1 或 2)
    # -------------------------------------------------------
    if n_dims >= 7:
        print("【特殊处理】正在将最后 7 个特征强制转换为离散值 (1 或 2)...")

        # 获取最后 7 列的原始采样值 (0-1之间)
        # 这种做法比 round 更科学，因为它利用了 LHS 的概率分布
        raw_last_7 = sample[:, -7:]

        # 逻辑：概率 < 0.5 变成 1，概率 >= 0.5 变成 2
        discrete_values = np.where(raw_last_7 < 0.5, 1.0, 2.0)

        # 覆盖掉之前缩放的值
        scaled_samples[:, -7:] = discrete_values
    else:
        print("警告：特征总数少于 7 个，无法执行'倒数7个特征'的特殊处理逻辑。")

    print(f"成功生成 {n_samples} 组样本数据。")
    return scaled_samples, feature_names


def find_top_k_similar(predictions, inputs, target_data, top_k=10):
    """
    找出与目标数据最相似的 Top K 方案
    Similarity Metric: RMSE (均方根误差)
    """
    # 确保 target_data 形状正确 (1, 17)
    target = target_data.reshape(1, -1)

    # 计算每一组预测值与真实值的差的平方
    # predictions: (200, 17), target: (1, 17) -> broadcasting
    diff = predictions - target

    # 计算 RMSE (欧氏距离的变体)
    # axis=1 表示对每个样本的17个频点求均值
    mse = np.mean(diff ** 2, axis=1)
    rmse = np.sqrt(mse)

    # 获取排序后的索引 (从小到大)
    sorted_indices = np.argsort(rmse)

    # 取前 K 个
    top_k_indices = sorted_indices[:top_k]

    return top_k_indices, rmse[top_k_indices]


def plot_top10_vs_target(target_data, top_preds_data, save_path):
    """
    绘制 Top 10 预测方案与目标值的对比图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'STKAITI', 'Arial Unicode MS']  # 适配中文
    plt.rcParams['axes.unicode_minus'] = False

    # 定义标准频率轴 (1/3倍频程常用频率)
    # 如果你的数据点不是17个，代码会自动截取
    std_freqs = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                 2000, 2500, 3150, 4000, 5000, 6300, 8000]

    num_points = len(target_data)
    # 截取对应长度的频率标签
    x_labels = std_freqs[:num_points]
    x_axis = range(num_points)  # 用于绘图的均匀坐标

    plt.figure(figsize=(12, 7))

    # 1. 绘制 10 条预测线 (使用细虚线，颜色稍微淡一点)
    # 使用 colormap 生成渐变色，或者统一用蓝色
    colors = plt.cm.viridis(np.linspace(0, 1, 10))

    for i in range(len(top_preds_data)):
        # 只在第一条和最后一条加标签，避免图例太拥挤
        label = f'Top 10 方案' if i == 0 else None
        plt.plot(x_axis, top_preds_data[i],
                 color='gray',
                 linestyle='--',
                 linewidth=1,
                 alpha=0.6,  # 透明度
                 label=label)

    # 2. 绘制最强 Top 1 (可以高亮一下，比如用蓝色)
    plt.plot(x_axis, top_preds_data[0],
             color='blue',
             linestyle='--',
             linewidth=2,
             label='最佳预测')

    # 3. 绘制目标真实线 (粗红实线，最显眼)
    plt.plot(x_axis, target_data,
             color='red',
             marker='o',
             linewidth=2.5,
             label='目标数据 (Target)')

    # 设置 X 轴
    plt.xticks(x_axis, x_labels, rotation=45, fontsize=20)
    # 设置 Y 轴
    plt.yticks(fontsize=20)

    plt.title('Top 10 优选方案预测值 vs 目标值对比', fontsize=20)
    plt.xlabel('频率 (Hz)', fontsize=20)
    plt.ylabel('噪声 (dB)', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"对比图已保存至: {save_path}")

    plt.show()

# ==========================================
# 3. 主程序
# ==========================================
def make_top10_optimization(model_path, input_file_path, output_file_path, original_data_path, result_save_path):

    # ---------------- 1. 加载参数与模型 ----------------
    print("1. 加载标准化参数...")
    input_cols, output_cols, input_mean, input_std, output_mean, output_std = load_normalization_params(
        input_file_path, output_file_path
    )

    print("2. 加载目标原始数据 (Ground Truth)...")
    target_data = load_original_data(original_data_path)
    if target_data is None: return

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    output_dim = len(output_cols) if len(output_cols) > 0 else 1
    model = AdvancedModel(input_dim=len(input_cols), output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------------- 2. 拉丁超立方采样 ----------------
    print("\n3. 执行拉丁超立方采样 (LHS)...")
    # 生成 200 组物理值样本
    lhs_inputs, feature_names = generate_lhs_samples(input_file_path, n_samples=1000)

    # ---------------- 3. 批量预测 ----------------
    print("4. 对 200 组样本进行批量预测...")

    # 标准化输入 (输入到模型前必须归一化)
    # 广播机制: (200, n) - (n,) / (n,)
    norm_inputs = (lhs_inputs - input_mean) / (input_std + 1e-8)

    # 转 Tensor
    input_tensor = torch.tensor(norm_inputs, dtype=torch.float32).to(device)

    # 预测
    with torch.no_grad():
        preds_tensor = model(input_tensor)

    preds_norm = preds_tensor.cpu().numpy()

    # 反归一化输出 (还原为真实dB值)
    preds_real = preds_norm * output_std + output_mean

    # ---------------- 4. 相似度排序与筛选 ----------------
    print("5. 计算相似度并筛选 Top 10...")

    # 截取预测数据的前17列（如果模型输出多于17列）以匹配 target_data
    # 假设 output_cols 对应的就是频率点
    preds_compare = preds_real[:, :17]

    top_indices, top_rmse = find_top_k_similar(preds_compare, lhs_inputs, target_data, top_k=10)

    # ---------------- 5. 整理结果 ----------------
    print("\n========= Top 10 相似方案 =========")

    # 提取 Top 10 的输入特征
    top_inputs = lhs_inputs[top_indices]

    # 【新增】同时提取 Top 10 的预测结果用于画图 (虽然不存Excel，但要用来画图)
    # 截取前17个点（对应频率点）
    top_preds_for_plot = preds_real[top_indices][:, :17]

    # 构建 DataFrame 列表 (Excel 只存特征)
    results_list = []
    for i, idx in enumerate(top_indices):
        row_data = {
            'Rank': i + 1,
            'Sample_ID': idx,
            'RMSE_Score': top_rmse[i]
        }
        # 只添加特征数据
        for j, col in enumerate(feature_names):
            row_data[col] = top_inputs[i, j]

        results_list.append(row_data)

    df_results = pd.DataFrame(results_list)

    # ---------------- 6. 计算特征统计 ----------------
    print("6. 计算 Top 10 方案的特征范围统计...")
    feature_df = df_results[feature_names]
    stats_df = pd.DataFrame({
        'Feature': feature_names,
        'Top10_Min': feature_df.min().values,
        'Top10_Max': feature_df.max().values,
        'Top10_Mean': feature_df.mean().values,
        'Range_Width': (feature_df.max() - feature_df.min()).values
    })

    # ---------------- 7. 保存 Excel ----------------
    try:
        with pd.ExcelWriter(result_save_path) as writer:
            df_results.to_excel(writer, sheet_name='Top10_Designs', index=False)
            stats_df.to_excel(writer, sheet_name='Feature_Analysis', index=False)
        print(f"\n成功！Excel结果已保存至: {result_save_path}")

    except Exception as e:
        print(f"保存Excel时出错: {e}")

    # # ---------------- 8. 【新增】生成对比图 ----------------
    # print("7. 生成 Top 10 预测趋势对比图...")
    # plot_save_path = base_dir + 'Top10_趋势对比图.png'  # 图片保存路径

    # plot_top10_vs_target(target_data, top_preds_for_plot, plot_save_path)
    
    return df_results, feature_names, target_data, top_preds_for_plot


if __name__ == '__main__':

        # ---------------- 配置路径 ----------------
    base_dir = 'E:/一汽风噪/目标定义/目标定义/'
    model_path = base_dir + 'best_model1117.pth'
    input_file_path = base_dir + '处理后扩充造型数6-7-19-25.xlsx'  # 训练用输入(用于归一化参数)
    output_file_path = base_dir + '扩充风噪数据6-7-19-25.xlsx'  # 训练用输出(用于归一化参数)

    # ！！！注意：这个文件现在需要两行数据：Min值和Max值！！！
    range_input_path = base_dir + '造型数据.xlsx'


    original_data_path = base_dir + '目标风噪数据.xlsx'  # 目标数据
    result_save_path = base_dir + 'Top10_优化方案结果.xlsx'
    make_top10_optimization(model_path, input_file_path, output_file_path, original_data_path, result_save_path)