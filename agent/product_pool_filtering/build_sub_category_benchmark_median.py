#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 sub_category 级别的 Benchmark（使用中位数）

输入: agent/product_pool_filtering/raw_products.csv
输出: agent/product_pool_filtering/outputs/sub_category_benchmark_median.csv
"""

import pandas as pd
import os
from typing import List

# 定义数值字段列表
NUMERIC_FIELDS = [
    'return_1y', 'return_3y', 'return_5y',
    'volatility_1y', 'volatility_3y', 'volatility_5y',
    'max_drawdown_1y', 'max_drawdown_2y', 'max_drawdown_3y',
    'sharpe_ratio_1y', 'sharpe_ratio_3y', 'sharpe_ratio_5y'
]

RETURN_FIELDS = ['return_1y', 'return_3y', 'return_5y']

# 需要单位归一化的字段（绝对值≥1则除以100）
NORMALIZE_FIELDS = [
    'return_1y', 'return_3y', 'return_5y',
    'volatility_1y', 'volatility_3y', 'volatility_5y',
    'max_drawdown_1y', 'max_drawdown_2y', 'max_drawdown_3y'
]

# 输出字段顺序（严格按此顺序）
OUTPUT_COLUMNS = [
    'sub_category',
    'return_1y', 'return_3y', 'return_5y',
    'volatility_1y', 'volatility_3y', 'volatility_5y',
    'max_drawdown_1y', 'max_drawdown_2y', 'max_drawdown_3y',
    'sharpe_ratio_1y', 'sharpe_ratio_3y', 'sharpe_ratio_5y'
]


def normalize_missing_values(value):
    """
    将缺失值标识符转换为 None（后续会被 pandas 识别为 NaN）
    
    Args:
        value: 待处理的值
        
    Returns:
        处理后的值（缺失值标识符返回 None，否则返回原值）
    """
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        value_upper = value.strip().upper()
        if value_upper in ['NA', 'N/A', 'NULL', 'NONE', '']:
            return None
    
    return value


def main():
    # 文件路径
    input_file = 'raw_products.csv'
    output_dir = 'outputs'
    output_file = os.path.join(output_dir, 'sub_category_benchmark_median.csv')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取数据
    print(f"正在读取输入文件: {input_file}")
    df = pd.read_csv(input_file)
    input_rows = len(df) + 1  # 包括表头
    
    print(f"输入文件总行数（包括表头）: {input_rows}")
    
    # 2. 处理 sub_category 缺失值
    df['sub_category'] = df['sub_category'].fillna('UNKNOWN')
    df['sub_category'] = df['sub_category'].replace('', 'UNKNOWN')
    
    # 3. 处理数值字段的缺失值
    print("正在处理缺失值...")
    for field in NUMERIC_FIELDS:
        if field in df.columns:
            # 应用缺失值标准化
            df[field] = df[field].apply(normalize_missing_values)
            # 转换为数值类型（errors="coerce" 会将无效值转为 NaN）
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    # 4. 按 sub_category 分组计算中位数
    print("正在按 sub_category 分组计算中位数...")
    
    # 准备聚合字典：对每个数值字段计算中位数
    agg_dict = {}
    for field in NUMERIC_FIELDS:
        if field in df.columns:
            agg_dict[field] = 'median'
    
    # 分组聚合
    result_df = df.groupby('sub_category', as_index=False).agg(agg_dict)
    
    # 5. 过滤：剔除所有 return_* 字段均为空的 sub_category
    print("正在过滤无效的 sub_category...")
    
    # 检查每个 sub_category 是否至少有一个 return_* 字段有有效值
    has_valid_return = result_df[RETURN_FIELDS].notna().any(axis=1)
    result_df = result_df[has_valid_return].copy()
    
    # 6. 单位归一化：对指定字段，统一除以100
    print("正在执行单位归一化...")
    for field in NORMALIZE_FIELDS:
        if field in result_df.columns:
            # 对非空值统一除以100
            result_df[field] = result_df[field] / 100.0
    
    # 7. 数据格式化
    print("正在格式化数据...")
    
    # 对数值字段保留 6 位小数
    for field in NUMERIC_FIELDS:
        if field in result_df.columns:
            result_df[field] = result_df[field].round(6)
    
    # 按 sub_category 升序排序
    result_df = result_df.sort_values('sub_category', ascending=True).reset_index(drop=True)
    
    # 8. 确保输出字段顺序正确
    # 只保留需要的列，并按指定顺序排列
    available_columns = [col for col in OUTPUT_COLUMNS if col in result_df.columns]
    result_df = result_df[available_columns]
    
    # 9. 输出到 CSV
    print(f"正在保存结果到: {output_file}")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 10. 控制台输出统计信息
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"输入文件总行数（包括表头）: {input_rows}")
    print(f"sub_category 数量: {len(result_df)}")
    print(f"输出文件路径: {os.path.abspath(output_file)}")
    print("\n输出结果前 10 行:")
    print("-"*60)
    print(result_df.head(10).to_string(index=False))
    print("="*60)


if __name__ == '__main__':
    main()

