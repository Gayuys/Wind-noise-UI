import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP冲突

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform


# ====================== 响度计算函数（集成） ======================
def dBA_third_octave_to_loudness_sone(dBA_17bands, fieldtype='diffuse'):
    """
    输入：200Hz~8000Hz 的 17 个 1/3 倍频程 dB(A) 值（list 或 np.array）
    输出：ISO 532-1 总响度 N（单位：sone）
    fieldtype : 'diffuse'（默认，扩散声场）或 'free'（自由场）
    """
    if len(dBA_17bands) != 17:
        raise ValueError("必须正好输入 17 个值（200Hz - 8000Hz）")

    # 1. 反 A 计权，恢复真实 dB SPL
    A_weight = np.array([-25.5, -22.7, -19.2, -16.1, -13.4, -10.9, -8.6, -6.6,
                         -4.8, -3.2, -1.9, -0.8, 0.0, 1.0, 1.2, 1.0, -1.1])
    L_spl = np.array(dBA_17bands) - A_weight

    # 2. 补齐成 28 个带（25Hz~12.5kHz）
    LT = np.zeros(28)
    LT[8:25] = L_spl  # 200Hz 从第9个带开始（索引8）

    # 3. 核心响度计算（原 SWJTU 算法精简版）
    RAP = np.array([45, 55, 65, 71, 80, 90, 100, 120])
    DLL = np.array([[-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
                    [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
                    [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
                    [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
                    [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
                    [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
                    [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
                    [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]]).T
    LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    AO = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4.0, -1.5, 2.0, 5.0, 12.0])
    DDF = np.array([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3.0, 2.0, 0.0, -1.4, -2.0, -1.9, -1.0, 0.5, 3.0, 4.0, 4.3, 4.0])
    DCB = np.array(
        [-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0.0, -0.5])
    ZUP = np.array(
        [0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8, 22.7, 23.6,
         24.0])
    RNS = np.array(
        [21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10, 0.035, 0.0])
    USL = np.array([[13.00, 8.20, 6.30, 5.50, 5.50, 5.50, 5.50, 5.50],
                    [9.00, 7.50, 6.00, 5.10, 4.50, 4.50, 4.50, 4.50],
                    [7.80, 6.70, 5.60, 4.90, 4.40, 3.90, 3.90, 3.90],
                    [6.20, 5.40, 4.60, 4.00, 3.50, 3.20, 3.20, 3.20],
                    [4.50, 3.80, 3.60, 3.20, 2.90, 2.70, 2.70, 2.70],
                    [3.70, 3.00, 2.80, 2.35, 2.20, 2.20, 2.20, 2.20],
                    [2.90, 2.30, 2.10, 1.90, 1.80, 1.70, 1.70, 1.70],
                    [2.40, 1.70, 1.50, 1.35, 1.30, 1.30, 1.30, 1.30],
                    [1.95, 1.45, 1.30, 1.15, 1.10, 1.10, 1.10, 1.10],
                    [1.50, 1.20, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
                    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
                    [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
                    [0.40, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22],
                    [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
                    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
                    [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
                    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
                    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]])

    # 计算 TI、LCB
    TI = np.zeros(11)
    for i in range(11):
        j = 0
        while j < 8:
            if LT[i] <= RAP[j] - DLL[i, j]:
                TI[i] = 10 ** (0.1 * (LT[i] + DLL[i, j]))
                break
            j += 1
    GI = [TI[0:6].sum(), TI[6:9].sum(), TI[9:11].sum()]
    LCB = np.array([10 * np.log10(g) if g > 0 else -np.inf for g in GI])

    # 计算特定响度 NM
    LE = np.zeros(20)
    NM = np.zeros(21)
    for i in range(20):
        LE[i] = LCB[i] if i < 3 else LT[i + 8]
        LE[i] = LE[i] - AO[i]
        if fieldtype.lower() == 'diffuse':
            LE[i] += DDF[i]
        if LE[i] > LTQ[i]:
            LE[i] -= DCB[i]
            MP1 = 0.0635 * 10 ** (0.025 * LTQ[i])
            MP2 = (1 - 0.25 + 0.25 * 10 ** (0.1 * (LE[i] - LTQ[i]))) ** 0.25 - 1
            NM[i] = max(MP1 * MP2, 0)

    # 低电平修正
    KORRY = min(0.4 + 0.32 * NM[0] ** 0.2, 1.0)
    NM[0] *= KORRY

    # 积分得到总响度 N
    N = Z1 = N1 = J = 0.0
    for i in range(21):
        ZUP_i = ZUP[i] + 1e-4
        IG = min(i - 1, 7)
        while Z1 < ZUP_i:
            if N1 <= NM[i]:
                N += NM[i] * (ZUP_i - Z1)
                Z1 = ZUP_i
            else:
                while J < 17 and NM[i] >= RNS[J]: J += 1
                J = min(J, 17)
                slope = USL[J if J < 17 else 16, IG]
                DZ = (N1 - NM[i]) / slope
                Z2 = Z1 + DZ
                if Z2 > ZUP_i:
                    DZ = ZUP_i - Z1
                    Z2 = ZUP_i
                N += DZ * (N1 + N1 - DZ * slope) / 2
                Z1 = Z2
                N1 -= DZ * slope

    return round(N, 3) if N <= 16 else round(N, 2)


# -------------------------- 模型结构（保持不变） --------------------------
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
        x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        x = self.input_conv(x)
        x = self.xception_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# -------------------------- 核心功能函数 --------------------------
def load_normalization_params(input_file_path, output_file_path):
    """加载标准化参数（均值和标准差）"""
    input_df = pd.read_excel(input_file_path, header=0)
    output_df = pd.read_excel(output_file_path, header=0)

    input_mean = input_df.values.mean(axis=0)
    input_std = input_df.values.std(axis=0)
    output_mean = output_df.values.mean(axis=0)
    output_std = output_df.values.std(axis=0)

    return (input_df.columns.tolist(), output_df.columns.tolist(),
            input_mean, input_std, output_mean, output_std)


def predict_with_model(input_data, model, input_mean, input_std, device):
    """使用模型进行预测"""
    model.eval()
    input_data_norm = (input_data - input_mean) / (input_std + 1e-8)
    input_tensor = torch.tensor(input_data_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)

    return predictions.cpu().numpy()


def calculate_loudness_from_freq(freq_values):
    """从频点dB值计算总响度（sone）"""
    if len(freq_values) != 17:
        raise ValueError("频点值必须是17个（200Hz~8000Hz）")
    return dBA_third_octave_to_loudness_sone(freq_values)


def genetic_algorithm_optimization(
        base_params,  # 原始参数（1D数组）
        model,
        input_mean,
        input_std,
        output_mean,
        output_std,
        device,
        original_freq_values,  # 原始方案的频点值
        adjust_indices,  # 需要调整的参数索引列表
        param_range_min,  # 与adjust_indices顺序对应的最小值列表
        param_range_max,  # 与adjust_indices顺序对应的最大值列表
        target_loudness,  # 响度目标值（需要低于此值）
        pop_size=100,
        generations=50,
        crossover_prob=0.8,
        mutation_prob=0.1
):
    """
    遗传算法优化参数：目标是找到低于目标响度且总响度最小的方案
    """
    error = 0 #初始化错误指针
    
    if len(adjust_indices) != len(param_range_min) or len(adjust_indices) != len(param_range_max):
        raise ValueError(f"调整参数数量（{len(adjust_indices)}）与范围数量不匹配，请检查列表长度")

    # 计算原始响度
    original_loudness = calculate_loudness_from_freq(original_freq_values)
    print(f"原始方案总响度: {original_loudness} sone")
    print(f"目标响度: {target_loudness} sone (需要找到低于此值的方案)")

    param_min_dict = {idx: min_val for idx, min_val in zip(adjust_indices, param_range_min)}
    param_max_dict = {idx: max_val for idx, max_val in zip(adjust_indices, param_range_max)}

    param_bounds = []
    for i in range(len(base_params)):
        if i in param_min_dict:
            param_bounds.append((param_min_dict[i], param_max_dict[i]))
        else:
            param_bounds.append((base_params[i], base_params[i]))

    def init_population(size):
        """初始化种群"""
        population = []
        for _ in range(size):
            individual = []
            for (lower, upper) in param_bounds:
                individual.append(uniform.rvs(loc=lower, scale=upper - lower))
            population.append(np.round(np.array(individual), 2))
        return np.array(population)

    population = init_population(pop_size)

    # 最优方案记录（低于目标响度且响度最小）
    best_params = None
    best_loudness = float('inf')
    best_freq_values = None

    def calculate_fitness(params):
        """
        计算适应度：
        1. 若响度 >= 目标值，适应度为负（惩罚）
        2. 若响度 < 目标值，适应度 = 目标值 - 响度（值越大表示响度越小，越优）
        """
        # 预测频点值
        freq_norm = predict_with_model(params.reshape(1, -1), model, input_mean, input_std, device)
        freq_values = (freq_norm * output_std + output_mean).flatten()

        # 计算响度
        try:
            loudness = calculate_loudness_from_freq(freq_values)
        except:
            return -1e9  # 计算失败则惩罚

        # 适应度计算
        if loudness >= target_loudness:
            # 高于目标值，惩罚（距离越远惩罚越大）
            fitness = - (loudness - target_loudness + 1)
        else:
            # 低于目标值，适应度=目标值-响度（值越大响度越小）
            fitness = target_loudness - loudness

        return fitness, loudness, freq_values

    # 遗传算法主循环
    for gen in range(generations):
        # 计算所有个体的适应度、响度、频点值
        fitness_info = []
        valid_fitness = []
        valid_individuals = []
        valid_loudness = []
        valid_freq_values = []

        for ind in population:
            fit, loud, freq = calculate_fitness(ind)
            fitness_info.append((fit, loud, freq, ind))

            # 只保留有效的个体（响度计算成功）
            if not np.isinf(fit) and not np.isnan(fit):
                valid_fitness.append(fit)
                valid_individuals.append(ind)
                valid_loudness.append(loud)
                valid_freq_values.append(freq)

        # 更新最优方案
        for i in range(len(valid_fitness)):
            loud = valid_loudness[i]
            if loud < target_loudness and loud < best_loudness:
                best_loudness = loud
                best_params = valid_individuals[i]
                best_freq_values = valid_freq_values[i]

        # 处理无有效个体的情况
        if not valid_fitness:
            print(f"第{gen + 1}代所有个体适应度无效，重新初始化种群。")
            population = init_population(pop_size)
            continue

        # 适应度归一化（用于选择）
        min_valid_fitness = min(valid_fitness)
        if min_valid_fitness < 0:
            valid_fitness = [f - min_valid_fitness + 1e-8 for f in valid_fitness]

        probs = np.array(valid_fitness) / np.sum(valid_fitness)
        selected_indices = np.random.choice(len(valid_individuals), size=pop_size, p=probs)

        selected_pop = [valid_individuals[i] for i in selected_indices]
        selected_loudness = [valid_loudness[i] for i in selected_indices]

        # 交叉操作
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected_pop[i]
            parent2 = selected_pop[i + 1] if (i + 1) < pop_size else parent1

            if np.random.rand() < crossover_prob:
                cross_point = np.random.randint(1, len(base_params))
                child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # 变异操作
        for i in range(pop_size):
            if np.random.rand() < mutation_prob:
                if adjust_indices:
                    mutate_idx = np.random.choice(adjust_indices)
                    lower, upper = param_bounds[mutate_idx]
                    new_population[i][mutate_idx] = np.round(uniform.rvs(loc=lower, scale=upper - lower), 2)

        # 边界裁剪和精度处理
        for i in range(pop_size):
            for j in range(len(base_params)):
                lower, upper = param_bounds[j]
                new_population[i][j] = np.clip(new_population[i][j], lower, upper)
            new_population[i] = np.round(new_population[i], 2)

        population = np.array(new_population[:pop_size])

        # 打印进度
        if (gen + 1) % 10 == 0:
            if best_params is not None:
                print(
                    f"第{gen + 1}代 | 最优响度: {best_loudness:.4f} sone | "
                    f"目标响度: {target_loudness} sone | "
                    f"满足条件: {best_loudness < target_loudness}"
                )
            else:
                print(f"第{gen + 1}代 | 暂未找到满足条件的方案")

    # 最终检查：如果没有找到满足条件的方案，返回原始方案
    if best_params is None:
        print("警告：未找到低于目标响度的方案，返回原始方案")
        error = 1
        best_params = base_params
        best_loudness = original_loudness
        best_freq_values = original_freq_values

    return best_params, best_freq_values, best_loudness, param_min_dict, param_max_dict, error


def visualize_freq_comparison(original, optimized, original_loudness, optimized_loudness, save_path=None):
    """折线图：原始与优化方案的频点对比，显示响度信息"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    freq_labels = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                   2000, 2500, 3150, 4000, 5000, 6300, 8000][:len(original)]
    x = np.arange(len(freq_labels))

    plt.figure(figsize=(14, 7))
    plt.plot(x, original, 'ro-', linewidth=2, markersize=6, label=f'原始方案 (响度: {original_loudness} sone)')
    plt.plot(x, optimized, 'bo-', linewidth=2, markersize=6, label=f'优化方案 (响度: {optimized_loudness} sone)')

    plt.xticks(x, freq_labels, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('频率(Hz)', fontsize=20)
    plt.ylabel('噪声值(dB)', fontsize=20)
    plt.title('原始方案与优化方案的频点对比（响度优化）', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"频点对比折线图已保存至: {save_path}")
    plt.show()


def visualize_param_changes(original_params, optimized_params, adjust_indices, param_min_dict, param_max_dict,
                            save_path=None):
    """柱状图：显示调整参数的前后对比及调整范围"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    param_indices = adjust_indices
    original_values = [original_params[i] for i in param_indices]
    optimized_values = [optimized_params[i] for i in param_indices]
    param_ranges = [f"{param_min_dict[i]}-{param_max_dict[i]}" for i in param_indices]

    x = np.arange(len(param_indices))
    width = 0.35

    plt.figure(figsize=(14, 7))
    plt.bar(x - width / 2, original_values, width, label='原始参数值', alpha=0.7)
    plt.bar(x + width / 2, optimized_values, width, label='优化参数值', alpha=0.7)

    plt.xticks(x, [f'参数{i}' for j, i in enumerate(param_indices)], fontsize=12)
    plt.yticks(fontsize=20)
    plt.xlabel('参数索引及调整范围', fontsize=20)
    plt.ylabel('参数值', fontsize=20)
    plt.title('调整参数的前后对比', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(axis='y', alpha=0.3)

    for i, (orig, opt) in enumerate(zip(original_values, optimized_values)):
        plt.text(i - width / 2, orig + max(original_values) * 0.01, f'{orig:.2f}',
                 ha='center', fontsize=12)
        plt.text(i + width / 2, opt + max(optimized_values) * 0.01, f'{opt:.2f}',
                 ha='center', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"参数对比图已保存至: {save_path}")
    plt.show()


# -------------------------- 主函数 --------------------------
def optimization_program(model_path, input_file_path, output_file_path, new_input_path, result_save_path, target_loudness, pop_size, generations):
    # -------------------------- 配置参数（请根据需要修改） --------------------------
    # 文件路径配置
    # model_path = 'D:/yan1/风噪/优化/Model1/best_model.pth'
    # input_file_path = 'D:/yan1/风噪/优化/Model1/输入数据.xlsx'
    # output_file_path = 'D:/yan1/风噪/优化/Model1/输出数据.xlsx'
    # new_input_path = 'D:/yan1/风噪/优化/Model1/需要优化的造型数据.xlsx'
    # result_save_path = 'D:/yan1/风噪/优化/Model1/参数优化结果_响度.xlsx'
    # freq_plot_path = 'D:/yan1/风噪/优化/Model1/频点对比折线图_响度.png'
    # param_plot_path = 'D:/yan1/风噪/优化/Model1/参数调整对比图_响度.png'

    # # 响度优化配置
    # target_loudness = 17  # 目标响度值（需要找到低于此值的方案）
    # pop_size = 150  # 种群大小
    # generations = 200  # 迭代次数

    # -------------------------- 加载数据和模型 --------------------------
    print("从Excel读取参数范围...")
    param_df = pd.read_excel(new_input_path, sheet_name='Sheet1')
    if not all(col in param_df.columns for col in ['原始值', '最小值', '最大值']):
        raise ValueError("Excel文件的sheet1必须包含'原始值'、'最小值'、'最大值'列")

    base_params = param_df['原始值'].values
    param_min = param_df['最小值'].values
    param_max = param_df['最大值'].values

    adjust_indices = []
    param_range_min = []
    param_range_max = []
    for idx in range(len(base_params)):
        if param_min[idx] != param_max[idx]:
            adjust_indices.append(idx)
            param_range_min.append(param_min[idx])
            param_range_max.append(param_max[idx])

    print("识别到可调整参数：")
    for i, idx in enumerate(adjust_indices):
        print(f"  参数索引{idx} | 范围: [{param_range_min[i]}, {param_range_max[i]}]")
    print("-" * 50)

    print("加载标准化参数...")
    input_cols, output_cols, input_mean, input_std, output_mean, output_std = load_normalization_params(
        input_file_path, output_file_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = AdvancedModel(input_dim=len(input_cols), output_dim=len(output_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式

    print("计算原始方案的频点值和响度...")
    original_freq_values = predict_with_model(base_params.reshape(1, -1), model, input_mean, input_std, device)
    original_freq_values = (original_freq_values * output_std + output_mean).flatten()
    original_loudness = calculate_loudness_from_freq(original_freq_values)


    # -------------------------- 开始优化 --------------------------
    print("开始遗传算法参数优化（响度目标）...")
    best_params, best_freq_values, best_loudness, param_min_dict, param_max_dict, error = genetic_algorithm_optimization(
        base_params=base_params,
        model=model,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        device=device,
        original_freq_values=original_freq_values,
        adjust_indices=adjust_indices,
        param_range_min=param_range_min,
        param_range_max=param_range_max,
        target_loudness=target_loudness,
        pop_size=pop_size,
        generations=generations
    )

    # -------------------------- 结果处理 --------------------------
    print("目标响度为...")
    print(target_loudness)
    # 参数对比表
    result_df = pd.DataFrame({
        "参数索引": list(range(len(base_params))),
        "原始参数值": base_params,
        "优化参数值": best_params,
        "调整量": [best_params[i] - base_params[i] for i in range(len(base_params))],
        "调整范围": [f"{param_min_dict.get(i, base_params[i])}-{param_max_dict.get(i, base_params[i])}"
                 for i in range(len(base_params))]
    })

    # 频点对比表
    freq_labels = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                   2000, 2500, 3150, 4000, 5000, 6300, 8000][:len(original_freq_values)]
    freq_df = pd.DataFrame({
        "频率(Hz)": freq_labels,
        "原始频点值(dB)": original_freq_values,
        "优化频点值(dB)": best_freq_values,
        "降低量(dB)": original_freq_values - best_freq_values
    })

    # 响度汇总表
    summary_df = pd.DataFrame({
        "指标": ["原始总响度(sone)", "优化后总响度(sone)", "响度降低量(sone)", "目标响度(sone)", "是否满足目标"],
        "数值": [
            original_loudness,
            best_loudness,
            original_loudness - best_loudness,
            target_loudness,
            "是" if best_loudness < target_loudness else "否"
        ]
    })

    with pd.ExcelWriter(result_save_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name="参数对比", index=False)
        freq_df.to_excel(writer, sheet_name="频点对比", index=False)
        summary_df.to_excel(writer, sheet_name="响度汇总", index=False)
    print(f"优化结果已保存至: {result_save_path}")

    return original_freq_values, best_freq_values, original_loudness, best_loudness, base_params, best_params, adjust_indices, param_min_dict, param_max_dict, error

if __name__ == '__main__':
    model_path = 'E:/一汽风噪/UI界面/model4/best_model.pth'
    input_file_path = 'E:/一汽风噪/UI界面/model4/输入数据.xlsx'
    output_file_path = 'E:/一汽风噪/UI界面/model4/输出数据.xlsx'
    new_input_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/优化方案.xlsx'
    result_save_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果//参数优化结果_响度.xlsx'
    freq_plot_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/频点对比折线图_响度.png'
    param_plot_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/参数调整对比图_响度.png'

    # 响度优化配置
    target_loudness = 15  # 目标响度值（需要找到低于此值的方案）
    pop_size = 2  # 种群大小
    generations = 2  # 迭代次数
    original_freq_values, best_freq_values, original_loudness, best_loudness, base_params, best_params, adjust_indices, param_min_dict, param_max_dict, error = optimization_program(model_path, input_file_path, output_file_path, new_input_path, result_save_path, target_loudness, pop_size, generations)
    if error ==1:
        print("未找到低于目标响度的方案，返回原始方案！")
    visualize_freq_comparison(
        original_freq_values, best_freq_values,
        original_loudness, best_loudness,
        freq_plot_path
    )

    visualize_param_changes(
            base_params, best_params, adjust_indices,
            param_min_dict, param_max_dict,
            param_plot_path
        )
