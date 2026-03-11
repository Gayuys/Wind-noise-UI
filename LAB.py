import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def save_with_frequency_header(series_data, filename, base_path):
    """
    将一列数据转置为一行，并保存为Excel，
    列名使用图片所示的频率表头（200, 250, ..., 8000），不带行索引
    """
    # 如果提供了基础路径，则组合完整路径
    full_path = os.path.join(base_path, filename)
    # 转置为一行，列名使用 frequency_columns（即图片中的频率表头）
    transposed_df = pd.DataFrame([series_data.values], columns=series_data.index)
    transposed_df.to_excel(full_path, header=True, index=False)
    print(f"数据保存至: {full_path}")

def draw_chart(result_df, car_rows, df,target_column, title_text):

    fig, ax = plt.subplots(figsize=(12, 6))

    x_indices = range(len(result_df))
    frequencies = [str(int(float(f))) for f in result_df['Frequency']]

    # 动态生成颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(car_rows)))

    # 绘制背景线（各车辆噪声曲线）
    for i, car in enumerate(car_rows):
        ax.plot(x_indices, df.loc[car],
                linestyle='--', linewidth=0.8, alpha=0.6,
                marker='o', markersize=2, color=colors[i % len(colors)])

    # 绘制目标线（Target L/A/B，突出显示）
    ax.plot(x_indices, result_df[target_column],
            label=target_column, color='red', linewidth=3.0,
            linestyle='-', marker='o', markersize=6,
            markerfacecolor='orange', zorder=10)

    # 图表样式设置
    ax.set_xticks(x_indices)
    ax.set_xticklabels(frequencies, rotation=45, fontsize=11)
    ax.set_xlabel('频率 (Hz)', fontsize=13)
    ax.set_ylabel('噪声 (dB)', fontsize=13)
    ax.set_title(f'{title_text}', fontsize=15, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, ncol=3, fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_LAB(file_path, save_path):
    # ==========================================
    # 第一步：读取 竞品车噪声 数据
    # ==========================================

    filename = file_path

    if not os.path.exists(filename):
        print(f"错误：找不到文件 '{filename}'，请确认文件在当前目录下。")
        exit()

    # 1. 读取数据：第一列是车型，第一行是频率
    df = pd.read_excel(filename, header=0, index_col=None)

    # 2. 强制重命名和定义数据列（频率列）
    frequency_columns = [col for col in df.columns]
    car_rows = list(df.index)

    # 3. 数据清洗：确保数据都是数值型
    for col in frequency_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    print(f"数据读取成功。频率列: {len(frequency_columns)}列, 车型行: {len(car_rows)}行")

    # ==========================================
    # 第二步：计算 Target L, Target A 和 Target B
    # ==========================================

    # 1. 计算 Target L (200-1000Hz 平均值, 1000Hz以上 最小值)
    target_l_list = []
    for freq_col in frequency_columns:
        freq = float(freq_col)
        values = df[freq_col]
        if freq <= 1000:
            target_l_list.append(values.mean())
        else:
            target_l_list.append(values.min())

    # 2. 计算 Target A (所有频段的平均值)
    target_a_list = df.mean(axis=0).tolist()

    # 3. 计算 Target B (所有频段的最大值 - 1dB)
    target_b_list = (df.max(axis=0) - 1).tolist()

    # 构建结果 DataFrame（频率为行，Target 为列）
    result_df = pd.DataFrame({
        'Frequency': frequency_columns,
        'Target L': target_l_list,
        'Target A': target_a_list,
        'Target B': target_b_list
    })


    # # 保存完整数据到「目标噪声.xlsx」
    # full_output_filename = '目标噪声.xlsx'
    # result_df.to_excel(full_output_filename, index=False)
    # print(f"噪声数据已保存至: {full_output_filename}")

    # ==========================================
    # 第三步：保存数据
    # ==========================================


    # 保存各Target的数据（带频率表头）
    save_with_frequency_header(result_df.set_index('Frequency')['Target L'], 'Target_L.xlsx', save_path)
    save_with_frequency_header(result_df.set_index('Frequency')['Target A'], 'Target_A.xlsx', save_path)
    save_with_frequency_header(result_df.set_index('Frequency')['Target B'], 'Target_B.xlsx', save_path)
    return result_df, car_rows, df
    
if __name__ == '__main__':
    # 示例调用
    file_path = r'E:\一汽风噪\目标定义\目标定义\竞品车噪声.xlsx'
    save_path = r'E:\一汽风噪\目标定义\目标定义'
    result_df, car_rows, df = calculate_LAB(file_path, save_path)
    # 分别绘制并显示三张图
    draw_chart(result_df, car_rows, df, 'Target L', 'Target L')
    draw_chart(result_df, car_rows, df, 'Target A', 'Target A')
    draw_chart(result_df, car_rows, df, 'Target B', 'Target B')
