#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_pools.py
==============
产品池过滤与导出脚本

功能：
1. 从 raw_products.csv 读取产品数据
2. 按"4 策略族 z-score 模板"进行策略内评分与过滤
3. 输出每个一级策略单独 CSV + 全部策略合并 CSV

兼容性：Python 3.9+, numpy 1.26.4, pandas 2.2.3
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# 配置区
# =============================================================================

# 策略字段名（若 raw_products.csv 中策略列名不同，在此修改）
STRATEGY_COL = "sub_category"
ASSET_CLASS_COL = "asset_class"

# 输入输出路径
INPUT_CSV = "raw_products.csv"
OUTPUT_DIR = "outputs"
OUTPUT_BY_STRATEGY_DIR = os.path.join(OUTPUT_DIR, "by_strategy")
OUTPUT_ALL_CSV = os.path.join(OUTPUT_DIR, "all_strategies_pool.csv")

# =============================================================================
# 指标口径优先级配置
# =============================================================================

# 收益指标优先级：3y > 5y > 1y
RETURN_COLS_PRIORITY = ["return_3y", "return_5y", "return_1y"]

# 波动率指标优先级：1y > 3y > 5y
VOLATILITY_COLS_PRIORITY = ["volatility_1y", "volatility_3y", "volatility_5y"]

# 最大回撤指标优先级：2y > 1y > 3y
DRAWDOWN_COLS_PRIORITY = ["max_drawdown_2y", "max_drawdown_1y", "max_drawdown_3y"]

# 夏普比率指标优先级：3y > 1y > 5y
SHARPE_COLS_PRIORITY = ["sharpe_ratio_3y", "sharpe_ratio_1y", "sharpe_ratio_5y"]

# =============================================================================
# 4 策略族权重配置
# =============================================================================

# 策略族权重：(w_ret, w_sharpe, w_vol, w_dd)
FAMILY_WEIGHTS = {
    "A": {"ret": 0.35, "sharpe": 0.35, "vol": 0.15, "dd": 0.15},  # 权益进攻
    "B": {"ret": 0.20, "sharpe": 0.40, "vol": 0.15, "dd": 0.25},  # 平衡增强/固收+
    "C": {"ret": 0.10, "sharpe": 0.45, "vol": 0.15, "dd": 0.30},  # 纯防守
    "D": {"ret": 0.25, "sharpe": 0.35, "vol": 0.20, "dd": 0.20},  # 分散/另类
}

# =============================================================================
# 16 一级策略 → 4 策略族映射
# =============================================================================

STRATEGY_FAMILY_MAP = {
    # A - 权益进攻（Equity Attack）
    "股票型": "A",
    "股票增强型": "A",
    "行业主题型": "A",
    "量化多因子": "A",
    
    # B - 平衡增强/固收+（Balanced）
    "固收增强型": "B",
    "偏债混合型": "B",
    "二级债基": "B",
    "可转债型": "B",
    
    # C - 纯防守（Defensive FI/Cash）
    "纯债型": "C",
    "货币基金": "C",
    "现金类-其他": "C",
    "短债型": "C",
    
    # D - 分散/另类（Diversifier）
    "商品型": "D",
    "CTA策略": "D",
    "多策略对冲": "D",
    "市场中性": "D",
}

# =============================================================================
# 过滤闸门配置（分位数阈值）
# =============================================================================

FILTER_CONFIG = {
    "volatility_quantile": 0.90,   # 剔除波动最差 10%
    "drawdown_quantile": 0.90,     # 剔除回撤最差 10%
    "sharpe_quantile": 0.20,       # 剔除夏普最差 20%
}

# 最大允许缺失的核心指标数量（超过则剔除）
MAX_MISSING_METRICS = 1

# raw_products.csv 原始列顺序
RAW_OUTPUT_COLS = [
    "product_name",
    "product_code", 
    "currency",
    "asset_class",
    "sub_category",
    "risk_level",
    "return_1y",
    "return_3y",
    "return_5y",
    "volatility_1y",
    "volatility_3y",
    "volatility_5y",
    "max_drawdown_1y",
    "max_drawdown_2y",
    "max_drawdown_3y",
    "sharpe_ratio_1y",
    "sharpe_ratio_3y",
    "sharpe_ratio_5y",
]

# =============================================================================
# 日志配置
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 工具函数
# =============================================================================

def coalesce_columns(df: pd.DataFrame, col_priority: List[str]) -> pd.Series:
    """
    按优先级合并多列，取第一个非空值。
    
    Args:
        df: 数据框
        col_priority: 列名优先级列表（第一个优先级最高）
        
    Returns:
        合并后的序列
    """
    result = pd.Series(index=df.index, dtype=float)
    result[:] = np.nan
    
    # 按优先级顺序填充：高优先级列先填，低优先级列补缺
    for col in col_priority:
        if col in df.columns:
            result = result.fillna(df[col])
    
    return result


def safe_zscore(series: pd.Series) -> pd.Series:
    """
    计算 z-score，处理 std=0 的情况。
    
    Args:
        series: 数值序列
        
    Returns:
        z-score 序列（std=0 时返回 0）
    """
    mean_val = series.mean()
    std_val = series.std()
    
    if pd.isna(std_val) or std_val == 0:
        return pd.Series(0.0, index=series.index)
    
    return (series - mean_val) / std_val


def get_strategy_family(
    strategy: str,
    asset_class: str,
    strategy_map: Dict[str, str]
) -> str:
    """
    获取策略所属的策略族。
    
    Args:
        strategy: 一级策略名称（sub_category）
        asset_class: 资产大类
        strategy_map: 策略 -> 策略族映射字典
        
    Returns:
        策略族代码 ("A"/"B"/"C"/"D")
    """
    # 1. 首先查映射表
    if strategy in strategy_map:
        return strategy_map[strategy]
    
    # 2. Fallback 规则
    strategy_lower = str(strategy).lower() if pd.notna(strategy) else ""
    asset_class_str = str(asset_class) if pd.notna(asset_class) else ""
    
    # 股票类 -> A
    if asset_class_str == "股票类":
        logger.warning(f"策略 '{strategy}' 未在映射表中，通过 asset_class='股票类' 判定为 A")
        return "A"
    
    # 固收+ 或包含 "固收" -> B
    if asset_class_str == "固收+" or "固收" in strategy_lower:
        logger.warning(f"策略 '{strategy}' 未在映射表中，通过 固收 关键词判定为 B")
        return "B"
    
    # 现金类 或包含 "货币"/"短债"/"纯债" -> C
    if asset_class_str == "现金类":
        logger.warning(f"策略 '{strategy}' 未在映射表中，通过 asset_class='现金类' 判定为 C")
        return "C"
    
    if any(kw in strategy_lower for kw in ["货币", "短债", "纯债"]):
        logger.warning(f"策略 '{strategy}' 未在映射表中，通过关键词判定为 C")
        return "C"
    
    # 包含 "商品"/"cta"/"对冲"/"另类" -> D
    if any(kw in strategy_lower for kw in ["商品", "cta", "对冲", "另类"]):
        logger.warning(f"策略 '{strategy}' 未在映射表中，通过关键词判定为 D")
        return "D"
    
    # 3. 默认 B
    logger.warning(f"策略 '{strategy}' 无法判定族类，默认使用 B")
    return "B"


def calculate_score(
    z_return: float,
    z_sharpe: float,
    z_vol: float,
    z_dd: float,
    family: str
) -> float:
    """
    计算综合评分。
    
    公式：score = w_ret*z_return + w_sharpe*z_sharpe - w_vol*z_vol - w_dd*z_dd
    
    Args:
        z_return: 收益 z-score
        z_sharpe: 夏普 z-score
        z_vol: 波动率 z-score
        z_dd: 回撤幅度 z-score
        family: 策略族代码
        
    Returns:
        综合评分
    """
    weights = FAMILY_WEIGHTS.get(family, FAMILY_WEIGHTS["B"])
    
    score = (
        weights["ret"] * z_return +
        weights["sharpe"] * z_sharpe -
        weights["vol"] * z_vol -
        weights["dd"] * z_dd
    )
    
    return score


# =============================================================================
# 核心处理函数
# =============================================================================

def load_and_prepare_data(input_path: str) -> pd.DataFrame:
    """
    加载并预处理数据。
    
    Args:
        input_path: 输入 CSV 文件路径
        
    Returns:
        预处理后的 DataFrame
    """
    logger.info(f"读取输入文件: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"读取行数: {len(df)}")
    
    # 打印原始列名
    logger.info(f"原始列名: {list(df.columns)}")
    
    # 检查必要列是否存在
    required_cols = [STRATEGY_COL, ASSET_CLASS_COL, "product_name", "product_code"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")
    
    # 移除空行
    df = df.dropna(subset=["product_code"])
    logger.info(f"移除空行后: {len(df)} 行")
    
    return df


def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取并标准化核心指标。
    
    Args:
        df: 原始数据框
        
    Returns:
        添加了标准化指标列的数据框
    """
    df = df.copy()
    
    # 1. 按优先级合并各指标
    logger.info("按优先级合并指标...")
    
    df["_raw_return"] = coalesce_columns(df, RETURN_COLS_PRIORITY)
    df["_raw_volatility"] = coalesce_columns(df, VOLATILITY_COLS_PRIORITY)
    df["_raw_drawdown"] = coalesce_columns(df, DRAWDOWN_COLS_PRIORITY)
    df["_raw_sharpe"] = coalesce_columns(df, SHARPE_COLS_PRIORITY)
    
    # 2. 直接使用原始值（假设已是小数制）
    df["metric_return"] = df["_raw_return"]
    df["metric_volatility"] = df["_raw_volatility"]
    df["metric_drawdown"] = df["_raw_drawdown"]
    df["metric_sharpe"] = df["_raw_sharpe"]
    
    # 3. 回撤幅度转为正数（dd_mag = abs(drawdown)）
    # 原数据中回撤可能是负数（如 -0.20 表示下跌 20%）
    df["metric_drawdown"] = df["metric_drawdown"].abs()
    
    # 4. 统计缺失情况
    metric_cols = ["metric_return", "metric_volatility", "metric_drawdown", "metric_sharpe"]
    df["_missing_count"] = df[metric_cols].isna().sum(axis=1)
    
    missing_stats = df["_missing_count"].value_counts().sort_index()
    logger.info(f"各行缺失指标数分布:\n{missing_stats}")
    
    # 5. 剔除缺失过多的行
    before_filter = len(df)
    df = df[df["_missing_count"] <= MAX_MISSING_METRICS]
    after_filter = len(df)
    logger.info(f"剔除缺失 >{MAX_MISSING_METRICS} 项的行: {before_filter} -> {after_filter}")
    
    # 清理临时列
    df = df.drop(columns=["_raw_return", "_raw_volatility", "_raw_drawdown", "_raw_sharpe", "_missing_count"])
    
    return df


def assign_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个产品分配策略族。
    
    Args:
        df: 数据框
        
    Returns:
        添加了 family 列的数据框
    """
    df = df.copy()
    
    df["family"] = df.apply(
        lambda row: get_strategy_family(
            row[STRATEGY_COL],
            row[ASSET_CLASS_COL],
            STRATEGY_FAMILY_MAP
        ),
        axis=1
    )
    
    family_counts = df["family"].value_counts().sort_index()
    logger.info(f"策略族分布:\n{family_counts}")
    
    return df


def calculate_zscore_and_score_by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    按策略分组计算 z-score 和综合评分。
    
    Args:
        df: 包含标准化指标的数据框
        
    Returns:
        添加了 z-score 和 score 列的数据框
    """
    df = df.copy()
    
    # 初始化 z-score 和 score 列
    df["z_return"] = np.nan
    df["z_sharpe"] = np.nan
    df["z_vol"] = np.nan
    df["z_dd"] = np.nan
    df["score"] = np.nan
    
    strategies = df[STRATEGY_COL].unique()
    logger.info(f"共 {len(strategies)} 个一级策略")
    
    for strategy in strategies:
        mask = df[STRATEGY_COL] == strategy
        strategy_df = df.loc[mask].copy()
        
        if len(strategy_df) < 2:
            logger.warning(f"策略 '{strategy}' 产品数 <2，跳过 z-score 计算")
            continue
        
        # 策略内计算 z-score
        df.loc[mask, "z_return"] = safe_zscore(strategy_df["metric_return"])
        df.loc[mask, "z_sharpe"] = safe_zscore(strategy_df["metric_sharpe"])
        df.loc[mask, "z_vol"] = safe_zscore(strategy_df["metric_volatility"])
        df.loc[mask, "z_dd"] = safe_zscore(strategy_df["metric_drawdown"])
    
    # 计算综合评分
    df["score"] = df.apply(
        lambda row: calculate_score(
            row["z_return"] if pd.notna(row["z_return"]) else 0,
            row["z_sharpe"] if pd.notna(row["z_sharpe"]) else 0,
            row["z_vol"] if pd.notna(row["z_vol"]) else 0,
            row["z_dd"] if pd.notna(row["z_dd"]) else 0,
            row["family"]
        ),
        axis=1
    )
    
    return df


def filter_by_strategy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    """
    按策略分组进行过滤。
    
    过滤规则（策略内）：
    - 剔除波动最差 10%：volatility > quantile(0.90)
    - 剔除回撤最差 10%：dd_mag > quantile(0.90)
    - 剔除夏普最差 20%：sharpe < quantile(0.20)
    
    Args:
        df: 完整数据框
        
    Returns:
        (过滤后的数据框, 各策略过滤前后数量字典)
    """
    vol_q = FILTER_CONFIG["volatility_quantile"]
    dd_q = FILTER_CONFIG["drawdown_quantile"]
    sharpe_q = FILTER_CONFIG["sharpe_quantile"]
    
    filtered_dfs = []
    size_stats = {}  # type: Dict[str, Tuple[int, int]]
    
    strategies = df[STRATEGY_COL].unique()
    
    for strategy in strategies:
        strategy_df = df[df[STRATEGY_COL] == strategy].copy()
        size_before = len(strategy_df)
        
        if size_before < 3:
            # 产品太少，不做过滤
            logger.warning(f"策略 '{strategy}' 产品数 <3，不做过滤")
            filtered_dfs.append(strategy_df)
            size_stats[strategy] = (size_before, size_before)
            continue
        
        # 计算分位数阈值
        vol_threshold = strategy_df["metric_volatility"].quantile(vol_q)
        dd_threshold = strategy_df["metric_drawdown"].quantile(dd_q)
        sharpe_threshold = strategy_df["metric_sharpe"].quantile(sharpe_q)
        
        # 应用过滤条件
        keep_mask = (
            (strategy_df["metric_volatility"] <= vol_threshold) &
            (strategy_df["metric_drawdown"] <= dd_threshold) &
            (strategy_df["metric_sharpe"] >= sharpe_threshold)
        )
        
        # 处理 NaN 值：如果指标为 NaN，保留该行（避免误杀）
        keep_mask = keep_mask | strategy_df["metric_volatility"].isna()
        keep_mask = keep_mask | strategy_df["metric_drawdown"].isna()
        keep_mask = keep_mask | strategy_df["metric_sharpe"].isna()
        
        strategy_filtered = strategy_df[keep_mask]
        size_after = len(strategy_filtered)
        
        logger.info(
            f"策略 '{strategy}': {size_before} -> {size_after} "
            f"(剔除 {size_before - size_after} 个)"
        )
        
        filtered_dfs.append(strategy_filtered)
        size_stats[strategy] = (size_before, size_after)
    
    result_df = pd.concat(filtered_dfs, ignore_index=True)
    
    return result_df, size_stats


def prepare_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备输出列，保持列顺序一致，并对浮点数进行四舍五入。
    
    Args:
        df: 完整数据框
        
    Returns:
        筛选并排序列后的数据框
    """
    output_cols = [
        "product_name",
        "product_code",
        "currency",
        "asset_class",
        STRATEGY_COL,
        "risk_level",
        "metric_return",
        "metric_volatility",
        "metric_drawdown",
        "metric_sharpe",
        "z_return",
        "z_sharpe",
        "z_vol",
        "z_dd",
        "score",
        "family",
    ]
    
    # 保留存在的列
    existing_cols = [col for col in output_cols if col in df.columns]
    result = df[existing_cols].copy()
    
    # 对浮点数列进行四舍五入，避免浮点精度问题
    float_cols_4dp = ["metric_return", "metric_volatility", "metric_drawdown", "metric_sharpe"]
    float_cols_6dp = ["z_return", "z_sharpe", "z_vol", "z_dd", "score"]
    
    for col in float_cols_4dp:
        if col in result.columns:
            result[col] = result[col].round(4)
    
    for col in float_cols_6dp:
        if col in result.columns:
            result[col] = result[col].round(6)
    
    return result


def prepare_raw_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备原始格式输出列，列顺序和值与 raw_products.csv 一致。
    
    Args:
        df: 完整数据框
        
    Returns:
        只包含原始列的数据框
    """
    # 保留存在的原始列
    existing_cols = [col for col in RAW_OUTPUT_COLS if col in df.columns]
    return df[existing_cols].copy()


def save_outputs(
    df: pd.DataFrame,
    size_stats: Dict[str, Tuple[int, int]]
) -> None:
    """
    保存输出文件。
    
    为每个输出生成两份 CSV：
    1. 带评分的版本（含 metric_*, z_*, score, family）
    2. 原始格式版本（列顺序和值与 raw_products.csv 一致）
    
    Args:
        df: 过滤后的数据框
        size_stats: 各策略过滤前后数量
    """
    # 创建输出目录
    os.makedirs(OUTPUT_BY_STRATEGY_DIR, exist_ok=True)
    output_raw_dir = os.path.join(OUTPUT_BY_STRATEGY_DIR, "raw")
    os.makedirs(output_raw_dir, exist_ok=True)
    logger.info(f"创建输出目录: {OUTPUT_BY_STRATEGY_DIR}")
    logger.info(f"创建原始格式输出目录: {output_raw_dir}")
    
    # 1. 按策略分别保存
    strategies = df[STRATEGY_COL].unique()
    
    for strategy in strategies:
        strategy_df = df[df[STRATEGY_COL] == strategy].copy()
        
        # 按 score 降序排列（先排序，保持两份输出顺序一致）
        strategy_df = strategy_df.sort_values("score", ascending=False)
        
        # 安全的文件名
        safe_name = strategy.replace("/", "_").replace("\\", "_")
        
        # 1a. 保存带评分的版本
        scored_df = prepare_output_columns(strategy_df)
        output_path = os.path.join(OUTPUT_BY_STRATEGY_DIR, f"{safe_name}.csv")
        scored_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"保存策略文件: {output_path} ({len(scored_df)} 行)")
        
        # 1b. 保存原始格式版本
        raw_df = prepare_raw_output_columns(strategy_df)
        raw_output_path = os.path.join(output_raw_dir, f"{safe_name}.csv")
        raw_df.to_csv(raw_output_path, index=False, encoding="utf-8-sig")
        logger.info(f"保存原始格式: {raw_output_path} ({len(raw_df)} 行)")
    
    # 2. 保存全策略合并文件
    # 按策略分组，再按 score 降序（先排序）
    df_sorted = df.sort_values(
        [STRATEGY_COL, "score"],
        ascending=[True, False]
    )
    
    # 2a. 带评分的版本
    all_df = prepare_output_columns(df_sorted)
    all_df["strategy_size_before"] = all_df[STRATEGY_COL].map(
        lambda x: size_stats.get(x, (0, 0))[0]
    )
    all_df["strategy_size_after"] = all_df[STRATEGY_COL].map(
        lambda x: size_stats.get(x, (0, 0))[1]
    )
    all_df.to_csv(OUTPUT_ALL_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"保存合并文件: {OUTPUT_ALL_CSV} ({len(all_df)} 行)")
    
    # 2b. 原始格式版本
    all_raw_df = prepare_raw_output_columns(df_sorted)
    raw_all_csv = OUTPUT_ALL_CSV.replace(".csv", "_raw.csv")
    all_raw_df.to_csv(raw_all_csv, index=False, encoding="utf-8-sig")
    logger.info(f"保存原始格式合并文件: {raw_all_csv} ({len(all_raw_df)} 行)")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数入口"""
    logger.info("=" * 60)
    logger.info("产品池过滤与导出脚本 启动")
    logger.info("=" * 60)
    
    # 获取脚本所在目录，作为工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_CSV)
    
    # 设置全局输出路径
    global OUTPUT_DIR, OUTPUT_BY_STRATEGY_DIR, OUTPUT_ALL_CSV
    OUTPUT_DIR = os.path.join(script_dir, "outputs")
    OUTPUT_BY_STRATEGY_DIR = os.path.join(OUTPUT_DIR, "by_strategy")
    OUTPUT_ALL_CSV = os.path.join(OUTPUT_DIR, "all_strategies_pool.csv")
    
    # Step 1: 加载数据
    df = load_and_prepare_data(input_path)
    
    # Step 2: 提取并标准化指标
    df = extract_metrics(df)
    
    # Step 3: 分配策略族
    df = assign_family(df)
    
    # Step 4: 策略内计算 z-score 和综合评分
    df = calculate_zscore_and_score_by_strategy(df)
    
    # Step 5: 策略内过滤
    df_filtered, size_stats = filter_by_strategy(df)
    
    # Step 6: 保存输出
    save_outputs(df_filtered, size_stats)
    
    # 打印统计
    logger.info("=" * 60)
    logger.info("处理完成！")
    logger.info(f"最终总行数: {len(df_filtered)}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

