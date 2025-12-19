#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
产品池筛选与分桶脚本
基于 bucketing_mechanism.md 规则实现

兼容：Python 3.9 + pandas 2.2.3 + numpy 1.26.4
"""

import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


# =============================================================================
# 配置常量
# =============================================================================

# 关键指标列
REQUIRED_COLUMNS = ['return_1y', 'return_3y', 'volatility_3y', 'sharpe_ratio_3y']

# 原始数据列（用于 raw_format 输出）
RAW_COLUMNS = [
    'product_name', 'product_code', 'currency', 'asset_class', 'sub_category',
    'risk_level', 'return_1y', 'return_3y', 'return_5y', 'volatility_1y',
    'volatility_3y', 'volatility_5y', 'max_drawdown_1y', 'max_drawdown_2y',
    'max_drawdown_3y', 'sharpe_ratio_1y', 'sharpe_ratio_3y', 'sharpe_ratio_5y'
]

# 分桶数量
NUM_BUCKETS = 5

# Top Alpha 数量
TOP_ALPHA_COUNT = 10

# 剔除阈值
FILTER_BOTTOM_PERCENTILE = 0.10  # 最差 10%
FILTER_TOP_VOL_PERCENTILE = 0.90  # 波动率最高 10%
FILTER_RETURN_MEDIAN = 0.50  # 收益中位数

# 桶内优选阈值
BUCKET_TOP_PERCENTILE = 0.80  # Top 20% (即分位数 >= 0.80)

# 需要保持字符串类型的列（防止前导零丢失）
STRING_COLUMNS = {'product_code': str}


# =============================================================================
# 数据加载与清洗
# =============================================================================

def load_and_clean_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载 CSV 并清洗数据
    
    Returns:
        cleaned_df: 清洗后的数据
        removed_df: 被移除的数据（用于记录）
    """
    # 显式指定 product_code 为字符串类型，防止前导零丢失
    df = pd.read_csv(filepath, dtype=STRING_COLUMNS)
    
    # 确保所有必需列存在
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    # 标记缺失值
    mask_valid = df[REQUIRED_COLUMNS].notna().all(axis=1)
    
    cleaned_df = df[mask_valid].copy()
    removed_df = df[~mask_valid].copy()
    
    print(f"[数据清洗] 原始产品数: {len(df)}")
    print(f"[数据清洗] 有效产品数: {len(cleaned_df)}")
    print(f"[数据清洗] 移除产品数: {len(removed_df)} (关键指标缺失)")
    
    return cleaned_df, removed_df


# =============================================================================
# Stage A: 强 Alpha 产品识别
# =============================================================================

def identify_top_alpha(df: pd.DataFrame, top_n: int = TOP_ALPHA_COUNT) -> pd.DataFrame:
    """
    识别收益率 Top N 的产品
    
    按 return_1y 降序排列，取前 N 个
    """
    sorted_df = df.sort_values('return_1y', ascending=False)
    top_alpha = sorted_df.head(top_n).copy()
    
    print(f"\n[Stage A] Top {top_n} Alpha 产品:")
    for idx, row in top_alpha.iterrows():
        print(f"  - {row['product_name']} ({row['sub_category']}): return_1y={row['return_1y']:.2f}%")
    
    return top_alpha


# =============================================================================
# Stage B: 分位数计算与过滤
# =============================================================================

def calculate_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    按一级策略（sub_category）分组计算分位数
    
    分位数定义：rank(x) / N，值越高表示在组内排名越靠前
    对于波动率：分位数越高表示波动越大（风险越高）
    """
    result = df.copy()
    
    # 按策略分组计算分位数
    # 收益率和夏普：越高越好，使用默认升序 rank
    # 波动率：越低越好，但我们计算时仍用升序 rank（高分位数=高波动）
    
    result['pct_return_3y'] = df.groupby('sub_category')['return_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    result['pct_sharpe_3y'] = df.groupby('sub_category')['sharpe_ratio_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    result['pct_volatility_3y'] = df.groupby('sub_category')['volatility_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    return result


def apply_filter_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    应用剔除规则（联合 OR 逻辑）
    
    剔除条件：
    1. 收益率分位数 <= 10%
    2. 夏普比率分位数 <= 10%
    3. (波动率分位数 >= 90%) AND (收益率分位数 < 50%)
    
    Returns:
        filtered_pool: 通过过滤的产品
        excluded_df: 被剔除的产品
    """
    # 剔除条件
    cond1 = df['pct_return_3y'] <= FILTER_BOTTOM_PERCENTILE
    cond2 = df['pct_sharpe_3y'] <= FILTER_BOTTOM_PERCENTILE
    cond3 = (df['pct_volatility_3y'] >= FILTER_TOP_VOL_PERCENTILE) & \
            (df['pct_return_3y'] < FILTER_RETURN_MEDIAN)
    
    # 满足任一条件则剔除
    exclude_mask = cond1 | cond2 | cond3
    
    filtered_pool = df[~exclude_mask].copy()
    excluded_df = df[exclude_mask].copy()
    
    print(f"\n[Stage B] 过滤结果:")
    print(f"  - 输入产品数: {len(df)}")
    print(f"  - 通过过滤: {len(filtered_pool)}")
    print(f"  - 被剔除: {len(excluded_df)}")
    print(f"    - 收益率最差10%: {cond1.sum()}")
    print(f"    - 夏普最差10%: {cond2.sum()}")
    print(f"    - 高波动且收益偏弱: {cond3.sum()}")
    
    return filtered_pool, excluded_df


# =============================================================================
# Stage C: 分桶与优选
# =============================================================================

def assign_buckets(df: pd.DataFrame, num_buckets: int = NUM_BUCKETS) -> pd.DataFrame:
    """
    C1: 按 return_1y 降序排序，轮询分配到各桶
    
    分配方式：
    - 排名1 -> Bucket 1
    - 排名2 -> Bucket 2
    - ...
    - 排名5 -> Bucket 5
    - 排名6 -> Bucket 1
    - ...
    """
    result = df.sort_values('return_1y', ascending=False).copy()
    result = result.reset_index(drop=True)
    
    # 轮询分配 bucket_id (1-5)
    result['bucket_id'] = (result.index % num_buckets) + 1
    
    print(f"\n[Stage C1] 轮询分桶:")
    for bucket_id in range(1, num_buckets + 1):
        bucket_count = (result['bucket_id'] == bucket_id).sum()
        print(f"  - Bucket {bucket_id}: {bucket_count} 产品")
    
    return result


def bucket_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    C2: 桶内多维优选
    
    保留满足任一条件的产品（OR 逻辑）：
    - 桶内 return_3y Top 20%
    - 桶内 sharpe_ratio_3y Top 20%
    - 桶内 volatility_3y 最低 20%（即分位数 <= 0.20）
    """
    result = df.copy()
    
    # 计算桶内分位数
    result['bucket_pct_return'] = df.groupby('bucket_id')['return_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    result['bucket_pct_sharpe'] = df.groupby('bucket_id')['sharpe_ratio_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    result['bucket_pct_volatility'] = df.groupby('bucket_id')['volatility_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    # 保留条件（OR 逻辑）
    keep_return = result['bucket_pct_return'] >= BUCKET_TOP_PERCENTILE
    keep_sharpe = result['bucket_pct_sharpe'] >= BUCKET_TOP_PERCENTILE
    keep_low_vol = result['bucket_pct_volatility'] <= (1 - BUCKET_TOP_PERCENTILE)  # 最低 20%
    
    keep_mask = keep_return | keep_sharpe | keep_low_vol
    
    selected = result[keep_mask].copy()
    
    print(f"\n[Stage C2] 桶内优选:")
    print(f"  - 优选前: {len(df)} 产品")
    print(f"  - 优选后: {len(selected)} 产品")
    
    return selected


def ensure_strategy_coverage(
    selected_df: pd.DataFrame,
    full_pool: pd.DataFrame,
    all_strategies: List[str]
) -> pd.DataFrame:
    """
    C3: 确保16种一级策略都有代表
    
    如果某策略在选中集合中缺失，从完整池中按 return_3y 排名补充
    """
    result = selected_df.copy()
    covered_strategies = set(result['sub_category'].unique())
    missing_strategies = set(all_strategies) - covered_strategies
    
    if missing_strategies:
        print(f"\n[Stage C3] 策略覆盖补充:")
        print(f"  - 缺失策略: {missing_strategies}")
        
        for strategy in missing_strategies:
            # 从完整池中找该策略的产品，按 return_3y 排序取第一个
            strategy_products = full_pool[full_pool['sub_category'] == strategy]
            if len(strategy_products) > 0:
                best_product = strategy_products.sort_values('return_3y', ascending=False).iloc[0:1]
                # 分配到产品数最少的桶
                bucket_counts = result['bucket_id'].value_counts()
                min_bucket = bucket_counts.idxmin()
                best_product = best_product.copy()
                best_product['bucket_id'] = min_bucket
                result = pd.concat([result, best_product], ignore_index=True)
                print(f"    - {strategy} -> Bucket {min_bucket}: {best_product.iloc[0]['product_name']}")
    else:
        print(f"\n[Stage C3] 所有16种策略已覆盖，无需补充")
    
    return result


def inject_top_alpha(
    buckets_df: pd.DataFrame,
    top_alpha_df: pd.DataFrame,
    num_buckets: int = NUM_BUCKETS
) -> pd.DataFrame:
    """
    C4: 将强 Alpha 产品注入每个桶（去重）
    """
    result = buckets_df.copy()
    
    # 标记已存在的 Top Alpha
    result['is_top_alpha'] = result['product_code'].isin(top_alpha_df['product_code'])
    
    injected_count = 0
    
    for bucket_id in range(1, num_buckets + 1):
        bucket_products = set(result[result['bucket_id'] == bucket_id]['product_code'])
        
        for _, alpha_row in top_alpha_df.iterrows():
            if alpha_row['product_code'] not in bucket_products:
                # 添加到该桶
                new_row = alpha_row.copy()
                new_row['bucket_id'] = bucket_id
                new_row['is_top_alpha'] = True
                result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                bucket_products.add(alpha_row['product_code'])
                injected_count += 1
    
    print(f"\n[Stage C4] Alpha 注入:")
    print(f"  - 注入 {injected_count} 条记录（Top Alpha 产品复制到各桶）")
    
    return result


# =============================================================================
# 输出函数
# =============================================================================

def save_outputs(
    top_alpha: pd.DataFrame,
    filtered_pool: pd.DataFrame,
    final_buckets: pd.DataFrame,
    output_dir: str
) -> None:
    """
    保存所有输出文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    raw_format_dir = os.path.join(output_dir, 'raw_format')
    os.makedirs(raw_format_dir, exist_ok=True)
    
    # 保存中间结果
    top_alpha.to_csv(os.path.join(output_dir, 'top_return_set.csv'), index=False)
    filtered_pool.to_csv(os.path.join(output_dir, 'filtered_pool.csv'), index=False)
    
    print(f"\n[输出] 中间结果已保存:")
    print(f"  - {output_dir}/top_return_set.csv ({len(top_alpha)} 条)")
    print(f"  - {output_dir}/filtered_pool.csv ({len(filtered_pool)} 条)")
    
    # 保存各桶
    print(f"\n[输出] 桶文件:")
    for bucket_id in range(1, NUM_BUCKETS + 1):
        bucket_data = final_buckets[final_buckets['bucket_id'] == bucket_id].copy()
        
        # 完整格式（含元信息）
        bucket_data.to_csv(
            os.path.join(output_dir, f'bucket_{bucket_id}.csv'),
            index=False
        )
        
        # 原始格式（只保留 raw_products.csv 的列）
        raw_columns_available = [c for c in RAW_COLUMNS if c in bucket_data.columns]
        bucket_raw = bucket_data[raw_columns_available].copy()
        bucket_raw.to_csv(
            os.path.join(raw_format_dir, f'bucket_{bucket_id}_raw.csv'),
            index=False
        )
        
        strategies = bucket_data['sub_category'].nunique()
        alpha_count = bucket_data['is_top_alpha'].sum() if 'is_top_alpha' in bucket_data.columns else 0
        print(f"  - Bucket {bucket_id}: {len(bucket_data)} 产品, {strategies} 策略, {alpha_count} Top Alpha")


# =============================================================================
# 主流程
# =============================================================================

def run_bucket_filter(input_file: str, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    执行完整的分桶流程
    """
    print("=" * 60)
    print("产品池筛选与分桶流程")
    print("=" * 60)
    
    # 1. 加载与清洗数据
    cleaned_df, removed_df = load_and_clean_data(input_file)
    
    # 获取所有策略类型
    all_strategies = cleaned_df['sub_category'].unique().tolist()
    print(f"\n[策略覆盖] 共 {len(all_strategies)} 种一级策略:")
    for s in sorted(all_strategies):
        count = (cleaned_df['sub_category'] == s).sum()
        print(f"  - {s}: {count} 产品")
    
    # 2. Stage A: 识别 Top Alpha
    top_alpha = identify_top_alpha(cleaned_df)
    
    # 3. Stage B: 分位数计算与过滤
    df_with_pct = calculate_percentiles(cleaned_df)
    filtered_pool, excluded = apply_filter_rules(df_with_pct)
    
    # 4. Stage C1: 轮询分桶
    bucketed = assign_buckets(filtered_pool)
    
    # 5. Stage C2: 桶内优选
    selected = bucket_selection(bucketed)
    
    # 6. Stage C3: 策略覆盖
    with_coverage = ensure_strategy_coverage(selected, filtered_pool, all_strategies)
    
    # 7. Stage C4: Alpha 注入
    final_buckets = inject_top_alpha(with_coverage, top_alpha)
    
    # 8. 保存输出
    save_outputs(top_alpha, filtered_pool, final_buckets, output_dir)
    
    print("\n" + "=" * 60)
    print("分桶流程完成!")
    print("=" * 60)
    
    return {
        'top_alpha': top_alpha,
        'filtered_pool': filtered_pool,
        'final_buckets': final_buckets,
        'excluded': excluded
    }


def print_bucket_summary(final_buckets: pd.DataFrame) -> None:
    """
    打印桶分布汇总
    """
    print("\n" + "=" * 60)
    print("桶分布汇总")
    print("=" * 60)
    
    for bucket_id in range(1, NUM_BUCKETS + 1):
        bucket = final_buckets[final_buckets['bucket_id'] == bucket_id]
        
        print(f"\n--- Bucket {bucket_id} ---")
        print(f"产品数: {len(bucket)}")
        
        # 统计策略分布
        strategy_dist = bucket['sub_category'].value_counts()
        print(f"策略覆盖: {len(strategy_dist)} 种")
        
        # 收益统计
        if 'return_1y' in bucket.columns:
            print(f"return_1y: mean={bucket['return_1y'].mean():.2f}%, "
                  f"min={bucket['return_1y'].min():.2f}%, "
                  f"max={bucket['return_1y'].max():.2f}%")
        
        # Top Alpha 统计
        if 'is_top_alpha' in bucket.columns:
            alpha_count = bucket['is_top_alpha'].sum()
            print(f"Top Alpha: {alpha_count} 产品")


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'raw_products.csv')
    output_dir = os.path.join(script_dir, 'outputs')
    
    # 执行分桶
    results = run_bucket_filter(input_file, output_dir)
    
    # 打印汇总
    print_bucket_summary(results['final_buckets'])

