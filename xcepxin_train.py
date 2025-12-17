import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from scipy.stats import uniform  # 用于超参数搜索空间初始化

# -------------------------- 路径配置 --------------------------
# 定义项目根目录（相对于当前脚本的位置）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义各子目录的相对路径
DATA_DIR = os.path.join(BASE_DIR, 'data')          # 数据目录
MODEL_DIR = os.path.join(BASE_DIR, '缓存')        # 模型保存目录
RESULT_DIR = os.path.join(BASE_DIR, '缓存')     # 结果保存目录

# 创建目录（如果不存在）
for dir_path in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# -------------------------- 1. 模型结构（保持不变） --------------------------
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv1d, self).__init__()
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


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(XceptionBlock, self).__init__()
        self.separable_conv1 = SeparableConv1d(in_channels, out_channels, stride=stride)
        self.separable_conv2 = SeparableConv1d(out_channels, out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        return x + residual


class AdvancedModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):  # 新增dropout_rate超参数
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.xception_blocks = nn.Sequential(
            XceptionBlock(32, 64),
            XceptionBlock(64, 128),
            XceptionBlock(128, 256)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 使用优化后的dropout_rate
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        x = self.input_conv(x)
        x = self.xception_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# -------------------------- 2. 数据加载/可视化工具 --------------------------
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


def plot_loss_curve(losses, save_dir=MODEL_DIR):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.title('Training Loss Curve', fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=20)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss 曲线已保存到 {save_dir}/loss_curve.png")


def save_predictions(predictions, output_cols, output_mean, output_std,
                     save_path=os.path.join(RESULT_DIR, '预测结果4.xlsx')):
    denormalized = predictions * output_std + output_mean
    pred_df = pd.DataFrame(denormalized, columns=output_cols)
    pred_df.to_excel(save_path, index=False)
    print(f"预测结果已保存到 {save_path}")


def plot_error_boxplot(y_true, y_pred, save_dir=RESULT_DIR):
    """绘制预测值与真实值误差的箱型图，横坐标使用频率值"""
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 计算误差（预测值 - 真实值）
    errors = y_pred - y_true
    print('误差的维度为',errors.shape)

    # 定义频率刻度（与预测对比图保持一致）
    frequencies = [200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                   2000, 2500, 3150, 4000, 5000, 6300, 8000]

    # 创建箱型图
    plt.figure(figsize=(14, 6))  # 加宽画布以容纳17个频率点
    bp = plt.boxplot(
        [errors[:, i] for i in range(errors.shape[1])],
        patch_artist=True,
        widths=0.6
    )

    # 设置箱体颜色
    for patch in bp['boxes']:
        patch.set_facecolor('#99CCFF')

    # 添加零误差参考线
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='零误差线')

    # 设置图表属性（关键修改：使用频率作为横坐标）
    plt.xticks(range(1, len(frequencies) + 1), frequencies, fontsize=10, rotation=45)  # 旋转45度避免重叠
    plt.xlabel('频率(Hz)', fontsize=14)
    plt.ylabel('误差值（预测值 - 真实值）(dB)', fontsize=14)
    plt.title('各频率点预测误差分布', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # 自动调整布局，防止标签被截断

    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/test_error_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集预测误差箱型图已保存到 {save_dir}/test_error_boxplot.png")


# -------------------------- 3. 遗传算法超参数优化核心模块（保持不变） --------------------------
def genetic_algorithm_hpo(
        X, y,
        pop_size=1,  # 遗传算法种群规模
        max_generations=100,  # 迭代代数
        crossover_prob=0.7,  # 交叉概率
        mutation_prob=0.1,  # 变异概率
        save_dir='D:/一汽风噪/AI/方案V2/'  # 保存适应度曲线的路径
):
    """
    遗传算法优化超参数：
    优化目标：最小化模型在验证集上的MSE损失
    优化的超参数空间：
    1. learning_rate: [1e-4, 1e-1]（对数均匀分布）
    2. batch_size: [16, 32, 64, 128]（离散值）
    3. dropout_rate: [0.2, 0.8]（连续值）
    4. weight_decay: [1e-5, 1e-2]（对数均匀分布）
    """
    # 步骤1：划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=42, shuffle=True
    )
    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 步骤2：定义超参数搜索空间
    param_space = {
        'lr': ('log', (1e-4, 1e-1)),  # 对数均匀分布
        'batch_size': ('discrete', [16, 32, 64, 128]),  # 离散值
        'dropout': ('uniform', (0.2, 0.6)),  # 均匀分布
        'weight_decay': ('log', (1e-5, 1e-2))  # 对数均匀分布
    }
    n_params = len(param_space)  # 超参数维度（4维）

    # 步骤3：初始化种群
    def init_population(pop_size):
        pop = []
        for _ in range(pop_size):
            individual = []
            for param_name, (param_type, param_range) in param_space.items():
                if param_type == 'log':
                    low, high = np.log(param_range[0]), np.log(param_range[1])
                    val = np.exp(uniform.rvs(loc=low, scale=high - low))
                elif param_type == 'discrete':
                    val = np.random.choice(param_range)
                elif param_type == 'uniform':
                    low, high = param_range
                    val = uniform.rvs(loc=low, scale=high - low)
                individual.append(val)
            pop.append(individual)
        return np.array(pop, dtype=np.float32)

    population = init_population(pop_size)
    best_val_loss = float('inf')
    best_hparams = None
    hparam_history = []  # 记录每代最优超参数和损失
    best_fitness_history = []  # 记录每代最优适应度
    avg_fitness_history = []  # 记录每代平均适应度

    # 步骤4：超参数解码
    def decode_hparams(individual):
        hparams = {}
        for i, (param_name, (param_type, _)) in enumerate(param_space.items()):
            val = individual[i]
            if param_name == 'batch_size':
                hparams[param_name] = int(round(val))  # 离散值需转为整数
            else:
                hparams[param_name] = val
        return hparams

    # 步骤5：模型训练与验证（评估单个超参数组合的性能）
    def evaluate_hparams(hparams):
        # 初始化模型
        model = AdvancedModel(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            dropout_rate=hparams['dropout']
        ).to(device)

        # 配置优化器和损失函数
        optimizer = optim.SGD(
            model.parameters(),
            lr=hparams['lr'],
            momentum=0.9,
            weight_decay=hparams['weight_decay']
        )
        criterion = nn.MSELoss()

        # 训练数据加载
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=hparams['batch_size'],
            shuffle=True
        )

        # 短期训练（仅验证超参数趋势）
        model.train()
        for epoch in range(30):  # 超参数评估用30轮训练
            total_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_dataloader)
        return val_loss

    # 步骤6：遗传算法主循环（选择→交叉→变异）
    print(f"\n=== 开始遗传算法超参数优化（共{max_generations}代，种群规模{pop_size}）===")
    for gen in range(max_generations):
        print(f"\n--- 第{gen + 1}/{max_generations}代 ---")

        # 6.1 计算种群适应度（损失越小，适应度越高）
        fitness = [1 / (evaluate_hparams(decode_hparams(ind)) + 1e-8) for ind in population]
        val_losses = [1 / fit - 1e-8 for fit in fitness]  # 还原为损失值
        current_best_idx = np.argmin(val_losses)
        current_best_loss = val_losses[current_best_idx]
        current_best_hparams = decode_hparams(population[current_best_idx])

        # 记录当前代的最优适应度和平均适应度
        best_fitness = fitness[current_best_idx]
        avg_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        print(f"当前代最优适应度: {best_fitness:.4f} | 平均适应度: {avg_fitness:.4f}")

        # 更新全局最优
        if current_best_loss < best_val_loss:
            best_val_loss = current_best_loss
            best_hparams = current_best_hparams
        hparam_history.append({
            'generation': gen + 1,
            'best_val_loss': best_val_loss,
            'best_hparams': best_hparams
        })
        print(f"当前代最优损失: {current_best_loss:.4f}")
        print(f"当前代最优超参数: {current_best_hparams}")

        # 6.2 选择操作（轮盘赌选择）
        total_fitness = sum(fitness)
        selection_probs = [fit / total_fitness for fit in fitness]
        selected_indices = np.random.choice(pop_size, size=pop_size, p=selection_probs)
        selected_pop = population[selected_indices]

        # 6.3 交叉操作（单点交叉）
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected_pop[i]
            parent2 = selected_pop[i + 1] if (i + 1) < pop_size else selected_pop[i]  # 处理奇数种群

            if np.random.rand() < crossover_prob:
                cross_point = np.random.randint(1, n_params)  # 交叉点
                child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # 6.4 变异操作（随机扰动）
        for i in range(pop_size):
            if np.random.rand() < mutation_prob:
                mutate_dim = np.random.randint(n_params)  # 随机选择变异维度
                param_name, (param_type, param_range) = list(param_space.items())[mutate_dim]

                # 按参数类型生成变异值
                if param_type == 'log':
                    low, high = np.log(param_range[0]), np.log(param_range[1])
                    mutate_val = np.exp(uniform.rvs(loc=low, scale=high - low))
                elif param_type == 'discrete':
                    mutate_val = np.random.choice(param_range)
                elif param_type == 'uniform':
                    low, high = param_range
                    mutate_val = uniform.rvs(loc=low, scale=high - low)

                # 更新变异维度的值
                new_population[i][mutate_dim] = mutate_val

        # 6.5 裁剪超参数范围（防止超出合理区间）
        for i in range(pop_size):
            for d, (param_name, (param_type, param_range)) in enumerate(param_space.items()):
                if param_type == 'log' or param_type == 'uniform':
                    low, high = param_range
                elif param_type == 'discrete':
                    low, high = min(param_range), max(param_range)
                new_population[i][d] = np.clip(new_population[i][d], low, high)

        # 更新种群
        population = np.array(new_population[:pop_size])

    # # 绘制遗传算法适应度曲线
    # os.makedirs(save_dir, exist_ok=True)
    # plt.figure(figsize=(10, 6))

    # # 设置中文字体支持
    # plt.rcParams["font.family"] = ["SimHei"]
    # plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # plt.plot(range(1, max_generations + 1), best_fitness_history, 'r-', linewidth=2,
    #          label='每代最优适应度')
    # plt.plot(range(1, max_generations + 1), avg_fitness_history, 'b--', linewidth=2,
    #          label='每代平均适应度')
    # plt.xlabel('迭代代数', fontsize=14)
    # plt.ylabel('适应度值', fontsize=14)
    # plt.title('遗传算法优化过程中的适应度曲线', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.tick_params(axis='both', labelsize=12)
    # plt.savefig(f'{save_dir}/ga_fitness_curve.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"遗传算法适应度曲线已保存到 {save_dir}/ga_fitness_curve.png")

    # # 输出优化结果
    # print(f"\n=== 超参数优化完成 ===")
    # print(f"最优超参数: {best_hparams}")
    # print(f"最优验证集损失: {best_val_loss:.4f}")
    return best_hparams, best_fitness_history, avg_fitness_history #输出最优参数、每代最优适应度、每代平均适应度


# -------------------------- 4. 模型训练函数（保持不变） --------------------------
def train_model(model, dataloader, val_dataloader, criterion, optimizer,
                num_epochs=200, patience=10, save_dir=MODEL_DIR):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')
    counter = 0
    losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_val_loss += criterion(outputs, labels).item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # 早停与最优模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            print(f"Epoch {epoch + 1}: 验证损失下降至 {avg_val_loss:.4f}，保存最优模型")
        else:
            counter += 1

        print(f"Epoch {epoch + 1} | 训练损失: {avg_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
        if counter >= patience:
            print(f"早停：在 {epoch + 1} 轮未改善")
            break

    # 绘制训练+验证损失曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', linewidth=2)
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linewidth=2)
    # plt.xlabel('Epoch', fontsize=24)
    # plt.ylabel('Loss', fontsize=24)
    # plt.title('Training & Validation Loss Curve', fontsize=24)
    # plt.legend(fontsize=24)
    # plt.grid(True)
    # plt.tick_params(axis='both', labelsize=20)
    # plt.savefig(f'{save_dir}/train_val_loss.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"训练+验证损失曲线已保存到 {save_dir}/train_val_loss.png")

    return model, losses, val_losses


# -------------------------- 5. 预测函数（保持不变） --------------------------
def predict(new_input_data, input_cols, output_cols, input_mean, input_std,
            model_path=os.path.join(MODEL_DIR, 'best_model.pth')):
    model = AdvancedModel(input_dim=len(input_cols), output_dim=len(output_cols))
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 标准化输入
    new_input_data = (new_input_data - input_mean) / (input_std + 1e-8)
    input_tensor = torch.tensor(new_input_data, dtype=torch.float32).to(device)

    # 预测
    with torch.no_grad():
        predictions = model(input_tensor)
    return predictions.cpu().numpy()


# ---------------------------------- 6. 主函数----------------------------------
def model_Train_main(
        input_file_path=os.path.join(DATA_DIR, '处理后有噪声扩充造型数据6-8-18-24-31-32-33.xlsx'),
        output_file_path=os.path.join(DATA_DIR, '有噪声扩充风噪数据6-8-18-24-31-32-33.xlsx'),
        ga_max_generations=20,  # 遗传算法迭代代数（用户可自定义）
        ga_pop_size=80,  # 遗传算法种群规模（用户可自定义）
):
    # 步骤1：加载数据
    print("=== 加载数据 ===")
    X, y, input_cols, output_cols, input_mean, input_std, output_mean, output_std = load_data_from_excels(
        input_file_path, output_file_path
    )

    # 步骤2：遗传算法优化超参数
    best_hparams, best_fitness_history, avg_fitness_history = genetic_algorithm_hpo(
                                                                                        X, y,
                                                                                        pop_size=ga_pop_size,  # 种群规模
                                                                                        max_generations=ga_max_generations,  # 迭代代数
                                                                                        crossover_prob=0.8,  # 推荐交叉概率
                                                                                        mutation_prob=0.03,  # 推荐变异概率
                                                                                        save_dir=MODEL_DIR  # 传递保存路径
                                                                                    )

    # 步骤3：划分最终训练/验证/测试集
    print("\n=== 划分最终数据集 ===")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=0.8, random_state=42, shuffle=True
    )

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_hparams['batch_size'],
        shuffle=True
    )
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 步骤4：用最优超参数训练模型
    print("\n=== 用最优超参数训练模型 ===")
    model = AdvancedModel(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        dropout_rate=best_hparams['dropout']
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=best_hparams['lr'],
        momentum=0.9,
        weight_decay=best_hparams['weight_decay']
    )
    criterion = nn.MSELoss()
    trained_model, losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=200, patience=10
    )

    # 步骤5：测试集最终评估（绘制误差箱型图）
    print("\n=== 测试集最终评估 ===")
    trained_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth')))
    trained_model.eval()
    device = next(trained_model.parameters()).device
    test_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            test_loss += criterion(outputs, labels).item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    test_loss /= len(test_loader)

    # 反归一化计算指标
    y_true_denorm = np.array(y_true) * output_std + output_mean
    y_pred_denorm = np.array(y_pred) * output_std + output_mean
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    print(f"测试集MSE损失: {test_loss:.4f}")
    print(f"测试集RMSE（dB）: {rmse:.2f}")
    print(f"测试集R²: {r2:.4f}")

    # 绘制测试集预测误差箱型图（横坐标为频率值）
    plot_error_boxplot(
        y_true_denorm,
        y_pred_denorm,
        save_dir=RESULT_DIR
    )

    return best_fitness_history, avg_fitness_history, losses, val_losses, y_true_denorm, y_pred_denorm


# -------------------------- 7. 执行入口 --------------------------
if __name__ == '__main__':
    # 关键参数：ga_max_generations=100（遗传算法迭代次数，可根据需求调整）
    best_fitness_history, avg_fitness_history, losses, val_losses, y_true_denorm, y_pred_denorm = model_Train_main(ga_max_generations=1)
    
#best_fitness_history, avg_fitness_history, losses, val_losses, y_true_denorm, y_pred_denorm = model_Train_main(ga_max_generations=1)