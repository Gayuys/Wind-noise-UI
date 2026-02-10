import os
import re
import traceback
import pandas as pd
import numpy as np


# ===================== 全局配置 - 25个目标参数 =====================
TARGET_PARAMS = [
    "A柱上端Y向尺寸(mm)",
    "A柱上端与前风挡阶差(mm)",
    "A柱上端平行段(mm)",
    "A柱下端Y向尺寸(mm)",
    "A柱下端与前风挡阶差(mm)",
    "A柱下端平行段(mm)",
    "A柱与X轴空间角度",
    "A柱与Y轴空间角度",
    "A柱与Z轴空间角度",
    "后视镜X向尺寸(mm)",
    "后视镜Y向尺寸(mm)",
    "后视镜Z向尺寸(mm)",
    "后视镜到车身距离(mm)",
    "侧窗倾角(°)",
    "后视镜前端角度(°)",
    "后视镜后端角度(°)",
    "前风挡角度(°)",
    "引擎盖角度(°)",
    "前风挡上端R角(°)",
    "引擎盖前端弧度倾角(°)-最大角度",
    "前格栅角度(°)",
    "后风挡角度(°)",
    "后风挡实际作用角(°)",
    "后视镜镜柄厚度（水切式）(mm)"
]
VALID_PARAM_COUNT = len(TARGET_PARAMS)  # 固定25个参数


# ===================== 工具函数 =====================
def get_param_column_indices(ref_file_path):
    """
    从参考Excel（造型优化特征.xlsx）中获取25个目标参数对应的列索引
    :param ref_file_name: 参考Excel文件名
    :return: 列索引列表（长度25），按TARGET_PARAMS顺序
    """
    ref_path = ref_file_path

    if not os.path.exists(ref_path):
        print(f"❌ 参考文件{ref_path}不存在！请放在代码同文件夹")
        return None

    try:
        # 读取参考Excel（无表头，第一行是参数名称/序号）
        df_ref = pd.read_excel(ref_path, engine="openpyxl", header=None)

        # 假设第一行是参数标识（序号/名称），遍历匹配目标参数
        param_indices = []

        # 标准化参考列名的函数
        def std_name(name):
            if pd.isna(name):
                return ""
            return re.sub(r'\s+|°|mm|\(.*?\)|（.*?）', '', str(name)).lower()

        # 标准化目标参数
        target_std = [std_name(p) for p in TARGET_PARAMS]

        # 遍历参考Excel的列，匹配目标参数
        for col_idx in range(df_ref.shape[1]):
            # 取列的第一个非空值作为参数标识
            col_vals = df_ref[col_idx].dropna()
            if col_vals.empty:
                continue
            col_name = col_vals.iloc[0]
            col_std = std_name(col_name)

            # 匹配目标参数
            for i, t_std in enumerate(target_std):
                if t_std in col_std or col_std in t_std:
                    param_indices.append(col_idx)
                    break

            # 匹配完成25个参数则停止
            if len(param_indices) == VALID_PARAM_COUNT:
                break

        # 校验匹配结果
        if len(param_indices) != VALID_PARAM_COUNT:
            print(f"⚠️ 仅匹配到{len(param_indices)}个参数（目标25个）！请检查参考Excel的参数名称")
            print(f"已匹配的列索引：{param_indices}")
            return None

        print(f"✅ 成功获取25个参数的列索引：{param_indices[:5]}...（后略）")
        return param_indices

    except Exception as e:
        print(f"❌ 读取参考Excel失败：{str(e)}")
        traceback.print_exc()
        return None


def extract_horizontal_excel_data(file_path, value_row_idx, col_indices, file_desc):
    """
    提取横向Excel的目标列数据（适配66列Excel）
    :param file_path: Excel文件路径
    :param value_row_idx: 数值所在行索引（从0开始）
    :param col_indices: 要提取的列索引列表（长度25）
    :param file_desc: 文件描述
    :return: 整理后的DataFrame
    """
    file_path

    if not os.path.exists(file_path):
        print(f"❌文件{file_path}不存在！")
        return None

    try:
        # 读取Excel：无表头，仅读取指定列
        df = pd.read_excel(
            file_path,
            engine="openpyxl",
            header=None,
            usecols=col_indices  # 仅读取目标列
        )

        # 校验数据行
        if value_row_idx >= len(df):
            print(f"❌ 【{file_desc}】数据行{value_row_idx}超出范围（共{len(df)}行）")
            return None

        # 提取数值行
        value_row = df.iloc[value_row_idx]
        values = value_row.tolist()

        # 构造结果DataFrame
        df_result = pd.DataFrame({
            "参数名称": TARGET_PARAMS,
            file_desc: values
        })

        # 转换为数值类型
        df_result[file_desc] = pd.to_numeric(df_result[file_desc], errors='coerce')
        df_result = df_result.dropna(subset=[file_desc])

        if df_result.empty:
            print(f"❌ 【{file_desc}】无有效数值！")
            return None

        print(f"✅ 【{file_desc}】提取完成：{len(df_result)}个有效参数")
        return df_result

    except Exception as e:
        print(f"❌ 【{file_desc}】提取失败：{str(e)}")
        traceback.print_exc()
        return None

def extract_horizontal_excel_data_head(file_path, value_row_idx, col_indices, file_desc):
    """
    提取横向Excel的目标列数据（适配66列Excel）有表头版
    :param file_path: Excel文件路径
    :param value_row_idx: 数值所在行索引（从0开始）
    :param col_indices: 要提取的列索引列表（长度25）
    :param file_desc: 文件描述
    :return: 整理后的DataFrame
    """
    file_path

    if not os.path.exists(file_path):
        print(f"❌文件{file_path}不存在！")
        return None

    try:
        # 读取Excel：无表头，仅读取指定列
        df = pd.read_excel(
            file_path,
            engine="openpyxl",
            header=0,
            usecols=col_indices  # 仅读取目标列
        )

        # 校验数据行
        if value_row_idx >= len(df):
            print(f"❌ 【{file_desc}】数据行{value_row_idx}超出范围（共{len(df)}行）")
            return None

        # 提取数值行
        value_row = df.iloc[value_row_idx]
        values = value_row.tolist()

        # 构造结果DataFrame
        df_result = pd.DataFrame({
            "参数名称": TARGET_PARAMS,
            file_desc: values
        })

        # 转换为数值类型
        df_result[file_desc] = pd.to_numeric(df_result[file_desc], errors='coerce')
        df_result = df_result.dropna(subset=[file_desc])

        if df_result.empty:
            print(f"❌ 【{file_desc}】无有效数值！")
            return None

        print(f"✅ 【{file_desc}】提取完成：{len(df_result)}个有效参数")
        return df_result

    except Exception as e:
        print(f"❌ 【{file_desc}】提取失败：{str(e)}")
        traceback.print_exc()
        return None



# ===================== 第一部分：几何参数整合 =====================
def integrate_geometric_params(characteristic_name_path, characteristic_path, range_path, weight_path):
    """整合横向Excel的几何参数，生成「几何参数整合结果.xlsx」（纯数据，无任何表头/索引）"""
    # 第一步：从参考Excel获取25个参数的列索引
    col_indices = get_param_column_indices(characteristic_name_path)
    if col_indices is None:
        print("❌ 无法获取参数列索引，整合终止")
        return None

    # 第二步：配置三个Excel的读取规则
    config = {
        "数值": {
            "file_path": characteristic_path,
            "value_row_idx": 0,  # 数值在第1行（索引0）
            "file_desc": "数值"
        },
        "最小值": {
            "file_path": range_path,
            "value_row_idx": 0,  # 最小值在第1行（索引0）
            "file_desc": "最小值"
        },
        "最大值": {
            "file_path": range_path,
            "value_row_idx": 1,  # 最大值在第2行（索引1）
            "file_desc": "最大值"
        },
        "权重系数": {
            "file_path": weight_path,
            "value_row_idx": 0,  # 权重系数在第1行（索引0）
            "file_desc": "权重系数",
            "is_weight": True  # 权重表直接读取25列，无需参考索引
        }
    }

    # 第三步：逐个提取数据
    data_frames = {}
    for key, cfg in config.items():
        if cfg.get("is_weight"):
            # 权重系数表：直接读取前25列，无需参考索引
            df = extract_horizontal_excel_data(
                file_path=cfg["file_path"],
                value_row_idx=cfg["value_row_idx"],
                col_indices=range(VALID_PARAM_COUNT),  # 0-24列
                file_desc=cfg["file_desc"],
            )
        else:
            # 造型特征/范围：使用参考列索引
            if key == "数值":
                df = extract_horizontal_excel_data_head(
                    file_path=cfg["file_path"],
                    value_row_idx=cfg["value_row_idx"],
                    col_indices=col_indices,
                    file_desc=cfg["file_desc"],
                )
            else:
                df = extract_horizontal_excel_data(
                    file_path=cfg["file_path"],
                    value_row_idx=cfg["value_row_idx"],
                    col_indices=col_indices,
                    file_desc=cfg["file_desc"],
                )

        if df is None:
            print(f"❌ {cfg['file_desc']}提取失败，整合终止")
            return None
        data_frames[key] = df

    # 第四步：合并所有数据（纵向格式，保留列名）
    df_merged = data_frames["数值"]
    for key in ["最小值", "最大值", "权重系数"]:
        df_merged = df_merged.merge(data_frames[key], on="参数名称", how="inner")

    # # ========== 关键修改1：纯数据转置（移除所有表头/索引） ==========
    # # 1. 设置参数名称为索引，转置
    # df_transposed = df_merged.set_index("参数名称").T
    # # 2. 重置索引并删除默认的index列（彻底移除行表头）
    # df_transposed = df_transposed.reset_index(drop=True)
    # # 3. 确保无列名（columns.name=None）
    # df_transposed.columns.name = None

    # # 第五步：保存整合结果（纯数据，无任何表头/索引）
    # output_file = os.path.join(os.path.dirname(__file__), "几何参数整合结果.xlsx")
    # # header=None：无列表头；index=False：无行索引；仅保存纯数据
    # df_transposed.to_excel(output_file, index=False, header=None, engine="openpyxl")

    # print(f"\n🎉 几何参数整合成功！")
    # print(f"📁 结果保存至：{output_file}")
    # print(f"📊 有效参数数：{len(df_merged)}个（目标25个）")
    # print(f"📋 保存格式：纯数据（无列表头、无行表头/索引）")

    # 返回纵向的原始数据（供评分函数使用）
    return df_merged


# ===================== 造型参数评分函数 =====================
def calculate_single_param_score(row):
    """造型参数单项基础评分计算逻辑"""
    val = row['数值']
    min_val = row['最小值']
    max_val = row['最大值']

    try:
        val = float(val)
        min_val = float(min_val)
        max_val = float(max_val)
    except (ValueError, TypeError):
        return None

    if min_val <= val <= max_val:
        return 100.0
    elif val < min_val:
        if min_val == 0:
            return 0.0
        ratio = (min_val - val) / min_val
        return 0.0 if ratio >= 1 else ((1 - ratio) ** 2) * 100
    elif val > max_val:
        if max_val == 0:
            return 0.0
        ratio = (val - max_val) / max_val
        return 0.0 if ratio >= 1 else ((1 - ratio) ** 2) * 100
    return 0.0


def process_modeling_params(df_merged):
    """
    处理造型参数评分（直接接收纵向整合数据，纯数据保存无任何表头/索引）
    :param df_merged: integrate_geometric_params返回的纵向整合数据
    :return: 最终得分、是否成功
    :shape_score 最终造型参数得分
    :df_transposed 第一行为原始造型值，第二行为最小值，第三行为最大值，第四行为权重系数，第五行为单项得分，第六行归一化权重，第七行为单项加权得分
    """
    try:
        # 直接使用传入的纵向数据，列名完整
        df = df_merged.copy()

        # 数据清洗（保留关键列，去除空值）
        df = df.dropna(subset=['权重系数', '数值', '最小值', '最大值'])
        df['权重系数'] = pd.to_numeric(df['权重系数'], errors='coerce')
        df = df[df['权重系数'] > 0]

        if df.empty:
            print("⚠️ [造型参数] 无有效数据！")
            return 0.0, False

        # 计算基础单项得分
        df['基础单项得分'] = df.apply(calculate_single_param_score, axis=1)
        df['基础单项得分'] = df['基础单项得分'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
        df = df.dropna(subset=['基础单项得分'])

        # 计算归一化权重和加权得分
        total_weight = df['权重系数'].sum()
        df['归一化权重(和为1)'] = (df['权重系数'] / total_weight).round(6)
        df['单项加权得分'] = (df['权重系数'] * df['基础单项得分']).round(4)

        # 计算最终加权平均分
        total_weighted_score = df['单项加权得分'].sum()
        shape_score = round(total_weighted_score / total_weight, 2)

        # 构造汇总行（仅保留数值，移除文本描述）
        summary_row = pd.DataFrame([{
            '参数名称': '【加权总平均分】',
            '数值': '', '最小值': '', '最大值': '',
            '权重系数': total_weight,
            '基础单项得分': '', '归一化权重(和为1)': '',
            '单项加权得分': shape_score
        }])
        df_with_summary = pd.concat([df, summary_row], ignore_index=True)

        # ========== 关键修改2：纯数据转置（移除所有表头/索引） ==========
        # 1. 设置参数名称为索引，转置
        df_transposed = df_with_summary.set_index("参数名称").T
        # 2. 重置索引并删除默认的index列（彻底移除行表头）
        df_transposed = df_transposed.reset_index(drop=True)
        # 3. 确保无列名
        df_transposed.columns.name = None

        # 保存评分结果（纯数据，无任何表头/索引）
        output_file = os.path.join(os.path.dirname(__file__), '造型参数各项得分.xlsx')
        df_transposed.to_excel(output_file, index=False, header=None, engine="openpyxl")

        print(f"✅ [造型参数] 评分结果已保存：造型参数各项得分.xlsx")
        print(f"🏆 造型参数得分：{shape_score} 分")
        return shape_score, True, df_transposed

    except Exception as e:
        print(f"❌ [造型参数] 处理失败：{str(e)}")
        traceback.print_exc()
        return 0.0, False



# ===================== 主函数 =====================
def calculate_rating(characteristic_name_path, characteristic_path, range_path, weight_path):
    """主函数"""

    # 第一步：整合几何参数（返回纵向原始数据）
    df_merged = integrate_geometric_params(characteristic_name_path, characteristic_path, range_path, weight_path)
    if df_merged is None:
        print("\n❌ 全流程终止：几何参数整合失败")
        return

    # 第二步：计算造型参数评分（直接传入纵向数据）
    shape_score, is_success, df_transposed = process_modeling_params(df_merged)
    if is_success:
        print("\n🎊 全流程执行完成！")
        return shape_score, df_transposed
    else:
        print("\n❌ 全流程终止：造型参数评分失败")


if __name__ == "__main__":
    characteristic_name_path = 'E:\一汽风噪\造型符合度评分/造型优化特征.xlsx'
    characteristic_path = 'E:\一汽风噪\造型符合度评分/造型特征参数.xlsx'
    range_path = 'E:\一汽风噪\造型符合度评分/造型范围.xlsx'
    weight_path = 'E:\一汽风噪\造型符合度评分/权重系数表.xlsx'
    shape_score, df_transposed = calculate_rating(characteristic_name_path, characteristic_path, range_path, weight_path)
    print(df_transposed)
    # origin_data_rounded = pd.Series(numeric_data).round(2).values
    # origin_data_list = origin_data_rounded.tolist()
    # print("完成计算")
    # print(origin_data_list)
