import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP冲突

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform


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
        target_indices,  # 新增：目标优化频段的索引列表
        pop_size=100,
        generations=50,
        crossover_prob=0.8,
        mutation_prob=0.1
):
    """遗传算法优化参数：支持分频段优化"""
    if len(adjust_indices) != len(param_range_min) or len(adjust_indices) != len(param_range_max):
        raise ValueError(f"调整参数数量（{len(adjust_indices)}）与范围数量不匹配，请检查列表长度")

    # 只计算目标频段的原始均值
    original_target_mean = np.mean(original_freq_values[target_indices])

    param_min_dict = {idx: min_val for idx, min_val in zip(adjust_indices, param_range_min)}
    param_max_dict = {idx: max_val for idx, max_val in zip(adjust_indices, param_range_max)}

    param_bounds = []
    for i in range(len(base_params)):
        if i in param_min_dict:
            param_bounds.append((param_min_dict[i], param_max_dict[i]))
        else:
            param_bounds.append((base_params[i], base_params[i]))

    def init_population(size):
        population = []
        for _ in range(size):
            individual = []
            for (lower, upper) in param_bounds:
                individual.append(uniform.rvs(loc=lower, scale=upper - lower))
            population.append(np.round(np.array(individual), 2))
        return np.array(population)

    population = init_population(pop_size)
    best_fitness = -np.inf
    best_params = None
    best_freq_values = None

    def calculate_fitness(params):
        """
        计算适应度：仅以目标频段的频点均值降低量为基础。
        """
        freq_norm = predict_with_model(params.reshape(1, -1), model, input_mean, input_std, device)
        freq_values = freq_norm * output_std + output_mean
        freq_values = freq_values.flatten()

        # 仅关注目标频段
        current_target_freq = freq_values[target_indices]
        current_target_mean = np.mean(current_target_freq)

        # 适应度 = 原始均值 - 当前均值（值越大表示改善越明显）
        fitness = original_target_mean - current_target_mean

        # 惩罚：如果目标频段内有频点反而升高，降低适应度
        worse_count = np.sum(current_target_freq > original_freq_values[target_indices])
        fitness -= worse_count * 0.5  # 每个升高的频点惩罚0.5

        return fitness

    # 遗传算法主循环
    for gen in range(generations):
        fitness_scores = [calculate_fitness(ind) for ind in population]

        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_params = population[current_best_idx]
            freq_norm = predict_with_model(best_params.reshape(1, -1), model, input_mean, input_std, device)
            best_freq_values = (freq_norm * output_std + output_mean).flatten()

        valid_indices = [i for i, f in enumerate(fitness_scores) if not np.isinf(f)]
        if not valid_indices:
            print(f"第{gen + 1}代所有个体适应度无效，重新初始化种群。")
            population = init_population(pop_size)
            continue

        valid_fitness = [fitness_scores[i] for i in valid_indices]
        valid_pop = population[valid_indices]
        min_valid_fitness = min(valid_fitness)
        if min_valid_fitness < 0:
            valid_fitness = [f - min_valid_fitness + 1e-8 for f in valid_fitness]
        probs = np.array(valid_fitness) / np.sum(valid_fitness)
        selected_indices = np.random.choice(len(valid_pop), size=pop_size, p=probs)
        selected_pop = valid_pop[selected_indices]

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

        for i in range(pop_size):
            if np.random.rand() < mutation_prob:
                if adjust_indices:
                    mutate_idx = np.random.choice(adjust_indices)
                    lower, upper = param_bounds[mutate_idx]
                    new_population[i][mutate_idx] = np.round(uniform.rvs(loc=lower, scale=upper - lower), 2)

        for i in range(pop_size):
            for j in range(len(base_params)):
                lower, upper = param_bounds[j]
                new_population[i][j] = np.clip(new_population[i][j], lower, upper)
            new_population[i] = np.round(new_population[i], 2)

        population = np.array(new_population[:pop_size])

        if (gen + 1) % 10 == 0:
            print(
                f"第{gen + 1}代 | 最优适应度: {best_fitness:.4f} | "
                f"目标频段原始均值: {original_target_mean:.2f} | "
                f"最优方案目标频段均值: {np.mean(best_freq_values[target_indices]):.2f}"
            )

    return best_params, best_freq_values, param_min_dict, param_max_dict

# 可视化原始与优化方案的频点对比
def visualize_freq_comparison(original, optimized, target_indices, save_path=None):
    """折线图：原始与优化方案的频点对比，高亮显示目标频段"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    freq_labels = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                   2000, 2500, 3150, 4000, 5000, 6300, 8000][:len(original)]
    x = np.arange(len(freq_labels))

    plt.figure(figsize=(14, 7))
    plt.plot(x, original, 'ro-', linewidth=2, markersize=6, label='原始方案')
    plt.plot(x, optimized, 'bo-', linewidth=2, markersize=6, label='优化方案')

    # 高亮目标频段
    if target_indices:
        target_x = np.array(target_indices)
        plt.fill_between(target_x, original[target_x], optimized[target_x],
                         color='green', alpha=0.3, label='优化目标频段')

    plt.xticks(x, freq_labels, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('频率(Hz)', fontsize=20)
    plt.ylabel('噪声值(dB)', fontsize=20)
    plt.title('原始方案与优化方案的频点对比', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"频点对比折线图已保存至: {save_path}")
    plt.show()

# 可视化调整参数的前后
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

    plt.xticks(x, [f'参数{i}' for j, i in enumerate(param_indices)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('参数索引及调整范围', fontsize=20)
    plt.ylabel('参数值', fontsize=14)
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
def optimization_program(model_path, input_file_path, output_file_path, new_input_path, result_save_path, full_freq_table_path, target_freq_min, target_freq_max, pop_size, generations):
    # -------------------------- 文件路径配置 --------------------------
    # model_path = 'D:/yan1/风噪/优化/Model2/best_model.pth' #模型路径
    # input_file_path = 'D:/yan1/风噪/优化/Model2/处理后有噪声扩充造型数据6-8-18-24-31-32-33.xlsx' #输入归一化
    # output_file_path = 'D:/yan1/风噪/优化/Model2/有噪声扩充风噪数据6-8-18-24-31-32-33.xlsx' #输出归一化
    # new_input_path = 'D:/yan1/风噪/优化/Model2/需要优化的造型数据.xlsx' #获取优化参数
    # result_save_path = 'D:/yan1/风噪/优化/Model2/参数优化结果.xlsx' #优化结果保存路径
    # freq_plot_path = 'D:/yan1/风噪/优化/Model2/频点对比折线图.png' #频点对比折线图保存路径
    # param_plot_path = 'D:/yan1/风噪/优化/Model2/参数调整对比图.png' #参数调整对比图保存路径
    # full_freq_table_path = 'D:/yan1/风噪/优化/Model2/200-8000Hz噪声值对比表.xlsx' #200-8000Hz噪声值对比表保存路径

    # -------------------------- 分频段优化配置（在此处修改目标频段） --------------------------
    target_freq_range = (target_freq_min, target_freq_max)
    # 完整的200-8000Hz频点列表
    all_freq_labels = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                       2000, 2500, 3150, 4000, 5000, 6300, 8000]

    # 自动找到目标频段对应的索引
    target_indices = [i for i, freq in enumerate(all_freq_labels)
                      if target_freq_range[0] <= freq <= target_freq_range[1]]

    if not target_indices:
        raise ValueError(f"指定的频段 {target_freq_range} Hz 在支持的频点列表中找不到，请检查配置。")

    print(f"目标优化频段: {target_freq_range[0]}-{target_freq_range[1]} Hz")
    print(f"对应的频点索引: {target_indices}")
    print(f"对应的具体频率: {[all_freq_labels[i] for i in target_indices]} Hz")
    print("-" * 50)

    # -------------------------- 加载数据和模型 --------------------------
    print("从Excel读取参数范围...")
    param_df = pd.read_excel(new_input_path, sheet_name='Sheet1')
    if not all(col in param_df.columns for col in ['原始值', '最小值', '最大值']):
        raise ValueError("Excel文件的Sheet1必须包含'原始值'、'最小值'、'最大值'列")

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

    print("计算原始方案的频点值...")
    original_freq_values = predict_with_model(base_params.reshape(1, -1), model, input_mean, input_std, device)
    original_freq_values = (original_freq_values * output_std + output_mean).flatten()

    # 确保频点值长度与完整频率列表匹配
    if len(original_freq_values) < len(all_freq_labels):
        # 如果模型输出频点少于完整列表，补充NaN
        original_freq_values = np.pad(original_freq_values, (0, len(all_freq_labels) - len(original_freq_values)),
                                      mode='constant', constant_values=np.nan)
    elif len(original_freq_values) > len(all_freq_labels):
        # 如果模型输出频点多于完整列表，截断到8000Hz
        original_freq_values = original_freq_values[:len(all_freq_labels)]

    print(f"原始方案200-8000Hz全频段均值: {np.nanmean(original_freq_values):.2f} dB")
    print(
        f"原始方案目标频段({target_freq_range[0]}-{target_freq_range[1]}Hz)均值: {np.nanmean(original_freq_values[target_indices]):.2f} dB")
    print("-" * 50)

    # -------------------------- 开始优化 --------------------------
    print("开始遗传算法参数优化...")
    best_params, best_freq_values, param_min_dict, param_max_dict = genetic_algorithm_optimization(
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
        target_indices=target_indices,  # 将目标频段索引传入优化函数
        pop_size=pop_size,
        generations=generations
    )

    # 确保优化后的频点值长度与完整频率列表匹配
    if len(best_freq_values) < len(all_freq_labels):
        best_freq_values = np.pad(best_freq_values, (0, len(all_freq_labels) - len(best_freq_values)),
                                  mode='constant', constant_values=np.nan)
    elif len(best_freq_values) > len(all_freq_labels):
        best_freq_values = best_freq_values[:len(all_freq_labels)]

    # -------------------------- 生成噪声值对比表 --------------------------
    print("\n生成噪声值对比表...")

    # 创建详细的全频段对比表
    full_freq_df = pd.DataFrame({
        '频率(Hz)': all_freq_labels,
        '原始噪声值(dB)': np.round(original_freq_values, 2),
        '优化噪声值(dB)': np.round(best_freq_values, 2),
        '噪声降低量(dB)': np.round(original_freq_values - best_freq_values, 2),
        '优化效果': np.where(best_freq_values < original_freq_values, '降低',
                         np.where(best_freq_values > original_freq_values, '升高', '不变')),
        '是否目标频段': ['是' if i in target_indices else '否' for i in range(len(all_freq_labels))]
    })

    # 添加统计行
    stats_data = {
        '频率(Hz)': ['200-8000Hz均值', f'{target_freq_range[0]}-{target_freq_range[1]}Hz均值'],
        '原始噪声值(dB)': [np.round(np.nanmean(original_freq_values), 2),
                      np.round(np.nanmean(original_freq_values[target_indices]), 2)],
        '优化噪声值(dB)': [np.round(np.nanmean(best_freq_values), 2),
                      np.round(np.nanmean(best_freq_values[target_indices]), 2)],
        '噪声降低量(dB)': [np.round(np.nanmean(original_freq_values) - np.nanmean(best_freq_values), 2),
                      np.round(np.nanmean(original_freq_values[target_indices]) - np.nanmean(
                          best_freq_values[target_indices]), 2)],
        '优化效果': ['-', '-'],
        '是否目标频段': ['否', '是']
    }
    stats_df = pd.DataFrame(stats_data)

    # 合并详细数据和统计数据
    final_freq_df = pd.concat([full_freq_df, stats_df], ignore_index=True)

    # 保存全频段对比表
    with pd.ExcelWriter(full_freq_table_path, engine='openpyxl') as writer:
        final_freq_df.to_excel(writer, sheet_name='200-8000Hz噪声值对比', index=False)

        # 添加汇总统计表
        summary_data = {
            '统计项': [
                '200-8000Hz全频段原始均值',
                '200-8000Hz全频段优化均值',
                '全频段均值降低量',
                f'{target_freq_range[0]}-{target_freq_range[1]}Hz原始均值',
                f'{target_freq_range[0]}-{target_freq_range[1]}Hz优化均值',
                '目标频段均值降低量',
                '目标频段降低频点数量',
                '目标频段升高频点数量',
                '全频段降低频点数量',
                '全频段升高频点数量'
            ],
            '数值': [
                np.round(np.nanmean(original_freq_values), 2),
                np.round(np.nanmean(best_freq_values), 2),
                np.round(np.nanmean(original_freq_values) - np.nanmean(best_freq_values), 2),
                np.round(np.nanmean(original_freq_values[target_indices]), 2),
                np.round(np.nanmean(best_freq_values[target_indices]), 2),
                np.round(
                    np.nanmean(original_freq_values[target_indices]) - np.nanmean(best_freq_values[target_indices]), 2),
                np.sum((best_freq_values[target_indices] < original_freq_values[target_indices]) & (
                    ~np.isnan(best_freq_values[target_indices]))),
                np.sum((best_freq_values[target_indices] > original_freq_values[target_indices]) & (
                    ~np.isnan(best_freq_values[target_indices]))),
                np.sum((best_freq_values < original_freq_values) & (~np.isnan(best_freq_values))),
                np.sum((best_freq_values > original_freq_values) & (~np.isnan(best_freq_values)))
            ],
            '单位': [
                'dB', 'dB', 'dB', 'dB', 'dB', 'dB', '个', '个', '个', '个'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='优化效果汇总', index=False)

    print(f"200-8000Hz全频段噪声值对比表已保存至: {full_freq_table_path}")

    # -------------------------- 原有结果处理 --------------------------
    print("\n保存优化结果...")
    result_df = pd.DataFrame({
        "参数索引": list(range(len(base_params))),
        "原始参数值": base_params,
        "优化参数值": best_params,
        "调整量": [best_params[i] - base_params[i] for i in range(len(base_params))],
        "调整范围": [f"{param_min_dict.get(i, base_params[i])}-{param_max_dict.get(i, base_params[i])}"
                 for i in range(len(base_params))]
    })

    freq_df = pd.DataFrame({
        "频率(Hz)": all_freq_labels[:len(original_freq_values)],
        "原始频点值(dB)": original_freq_values,
        "优化频点值(dB)": best_freq_values,
        "降低量(dB)": original_freq_values - best_freq_values
    })

    with pd.ExcelWriter(result_save_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name="参数对比", index=False)
        freq_df.to_excel(writer, sheet_name="频点对比", index=False)
    print(f"优化结果已保存至: {result_save_path}")




    print("\n" + "=" * 60)
    print("优化结果 Summary:")
    print("=" * 60)
    print(f"优化目标: 降低 {target_freq_range[0]}-{target_freq_range[1]} Hz 频段噪声")
    print(f"200-8000Hz全频段原始均值: {np.nanmean(original_freq_values):.2f} dB")
    print(f"200-8000Hz全频段优化均值: {np.nanmean(best_freq_values):.2f} dB")
    print(f"全频段均值降低量: {np.nanmean(original_freq_values) - np.nanmean(best_freq_values):.2f} dB")
    print(
        f"目标频段({target_freq_range[0]}-{target_freq_range[1]}Hz)原始均值: {np.nanmean(original_freq_values[target_indices]):.2f} dB")
    print(
        f"目标频段({target_freq_range[0]}-{target_freq_range[1]}Hz)优化均值: {np.nanmean(best_freq_values[target_indices]):.2f} dB")
    print(
        f"目标频段均值降低量: {np.nanmean(original_freq_values[target_indices]) - np.nanmean(best_freq_values[target_indices]):.2f} dB")
    print(
        f"目标频段内所有频点是否均降低: {np.all((best_freq_values[target_indices] < original_freq_values[target_indices]) | np.isnan(best_freq_values[target_indices]))}")
    print(f"全频段降低频点数量: {np.sum((best_freq_values < original_freq_values) & (~np.isnan(best_freq_values)))} 个")
    print(f"全频段升高频点数量: {np.sum((best_freq_values > original_freq_values) & (~np.isnan(best_freq_values)))} 个")
    print("=" * 60)
    
    return original_freq_values, best_freq_values, target_indices, base_params, best_params, adjust_indices, param_min_dict, param_max_dict


if __name__ == '__main__':
    model_path = 'E:/一汽风噪/UI界面/model4/best_model.pth' #模型路径
    input_file_path = 'E:/一汽风噪/UI界面/model4/输入数据.xlsx' #输入归一化
    output_file_path = 'E:/一汽风噪/UI界面/model4/输出数据.xlsx' #输出归一化
    new_input_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/优化方案.xlsx' #获取优化参数
    result_save_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/参数优化结果.xlsx' #优化结果保存路径
    freq_plot_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/频点对比折线图.png' #频点对比折线图保存路径
    param_plot_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/参数调整对比图.png' #参数调整对比图保存路径
    full_freq_table_path = 'E:/一汽风噪/UI界面/灵敏度分析/结果/200-8000Hz噪声值对比表.xlsx' #200-8000Hz噪声值对比表保存路径
    target_freq_min = 200
    target_freq_max = 8000
    pop_size = 2
    generations = 2
    original_freq_values, best_freq_values, target_indices, base_params, best_params, adjust_indices, param_min_dict, param_max_dict = optimization_program(model_path, input_file_path, output_file_path, new_input_path, result_save_path, full_freq_table_path, target_freq_min, target_freq_max, pop_size, generations)
    print("\n生成频点对比折线图...")
    visualize_freq_comparison(original_freq_values, best_freq_values, target_indices, freq_plot_path)
    if adjust_indices:
        print("生成参数调整对比图...")
        visualize_param_changes(base_params, best_params, adjust_indices, param_min_dict, param_max_dict,
                                param_plot_path)
