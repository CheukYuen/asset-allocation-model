#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_core_satellite.py
=======================
Core-Satellite 产品池生成脚本

功能：
1. 读取 all_strategies_pool.csv（已按 score 排序）
2. 按 4 大策略族（A/B/C/D）分组
3. 每个族的前 10% 标记为 Core，其余 90% 标记为 Satellite
4. 输出带 Core/Satellite 标签的 CSV

兼容性：Python 3.9+, numpy 1.26.4, pandas 2.2.3
"""

import os
import logging
import math
from typing import Dict, List

import pandas as pd

# =============================================================================
# 配置区
# =============================================================================

# 输入文件（已按 score 排序的产品池）
INPUT_SCORED_CSV = "outputs/all_strategies_pool.csv"
INPUT_RAW_CSV = "outputs/all_strategies_pool_raw.csv"

# 输出目录
OUTPUT_DIR = "outputs/core_satellite"

# Core 比例（前 10%）
CORE_RATIO = 0.10

# 策略列名
STRATEGY_COL = "sub_category"

# 16 一级策略 → 4 策略族映射
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

# 策略族名称
FAMILY_NAMES = {
    "A": "权益进攻",
    "B": "平衡增强",
    "C": "纯防守",
    "D": "分散另类",
}

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
# 核心函数
# =============================================================================

def get_family(strategy: str) -> str:
    """获取策略所属的策略族"""
    return STRATEGY_FAMILY_MAP.get(strategy, "B")


def assign_core_satellite(df: pd.DataFrame, core_ratio: float = CORE_RATIO) -> pd.DataFrame:
    """
    按策略族分组，为每个产品分配 Core/Satellite 标签。
    
    Args:
        df: 包含 family 和 score 列的数据框（已按 score 排序）
        core_ratio: Core 占比（默认 10%）
        
    Returns:
        添加了 pool_type 列的数据框
    """
    df = df.copy()
    df["pool_type"] = "Satellite"  # 默认为 Satellite
    
    families = df["family"].unique()
    
    for family in sorted(families):
        family_mask = df["family"] == family
        family_df = df[family_mask]
        family_size = len(family_df)
        
        # 计算 Core 数量（至少 1 个）
        core_count = max(1, math.ceil(family_size * core_ratio))
        
        # 获取该族的行索引（已按 score 排序，前面的是高分）
        family_indices = family_df.index.tolist()
        core_indices = family_indices[:core_count]
        
        # 标记 Core
        df.loc[core_indices, "pool_type"] = "Core"
        
        logger.info(
            f"策略族 {family} ({FAMILY_NAMES.get(family, family)}): "
            f"共 {family_size} 个产品, Core {core_count} 个 ({core_count/family_size*100:.1f}%), "
            f"Satellite {family_size - core_count} 个"
        )
    
    return df


def main() -> None:
    """主函数入口"""
    logger.info("=" * 60)
    logger.info("Core-Satellite 产品池生成脚本 启动")
    logger.info("=" * 60)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scored_path = os.path.join(script_dir, INPUT_SCORED_CSV)
    raw_path = os.path.join(script_dir, INPUT_RAW_CSV)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: 读取带评分的数据
    logger.info(f"读取评分数据: {scored_path}")
    df_scored = pd.read_csv(scored_path)
    logger.info(f"读取行数: {len(df_scored)}")
    
    # Step 2: 读取原始格式数据
    logger.info(f"读取原始数据: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    
    # Step 3: 确保 family 列存在（如果没有则重新计算）
    if "family" not in df_scored.columns:
        df_scored["family"] = df_scored[STRATEGY_COL].map(get_family)
    
    # Step 4: 按 family 分组，按 score 降序排列
    df_scored = df_scored.sort_values(
        ["family", "score"],
        ascending=[True, False]
    ).reset_index(drop=True)
    
    # 同步排序原始数据（按 product_code 对齐）
    df_raw["_sort_order"] = df_raw["product_code"].map(
        {code: idx for idx, code in enumerate(df_scored["product_code"])}
    )
    df_raw = df_raw.sort_values("_sort_order").drop(columns=["_sort_order"]).reset_index(drop=True)
    
    # Step 5: 分配 Core/Satellite 标签
    logger.info(f"Core 比例: {CORE_RATIO*100:.0f}%")
    df_scored = assign_core_satellite(df_scored, CORE_RATIO)
    
    # Step 6: 保存输出
    
    # 6a. 按策略族分别保存
    for family in sorted(df_scored["family"].unique()):
        family_mask = df_scored["family"] == family
        family_scored = df_scored[family_mask].copy()
        family_raw = df_raw[family_mask].copy()
        
        # 添加 pool_type 到原始数据
        family_raw["pool_type"] = family_scored["pool_type"].values
        
        # 带评分版本
        scored_path_out = os.path.join(output_dir, f"family_{family}.csv")
        family_scored.to_csv(scored_path_out, index=False, encoding="utf-8-sig")
        logger.info(f"保存策略族文件: {scored_path_out}")
        
        # 原始格式版本
        raw_cols = RAW_OUTPUT_COLS + ["pool_type"]
        existing_cols = [c for c in raw_cols if c in family_raw.columns]
        family_raw_out = family_raw[existing_cols]
        raw_path_out = os.path.join(output_dir, f"family_{family}_raw.csv")
        family_raw_out.to_csv(raw_path_out, index=False, encoding="utf-8-sig")
        logger.info(f"保存原始格式: {raw_path_out}")
    
    # 6b. 保存全部数据（合并文件）
    
    # 带评分版本
    all_scored_path = os.path.join(output_dir, "all_core_satellite.csv")
    df_scored.to_csv(all_scored_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存合并文件: {all_scored_path}")
    
    # 原始格式版本
    df_raw["pool_type"] = df_scored["pool_type"].values
    df_raw["family"] = df_scored["family"].values
    raw_cols_all = RAW_OUTPUT_COLS + ["family", "pool_type"]
    existing_cols_all = [c for c in raw_cols_all if c in df_raw.columns]
    df_raw_out = df_raw[existing_cols_all]
    all_raw_path = os.path.join(output_dir, "all_core_satellite_raw.csv")
    df_raw_out.to_csv(all_raw_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存原始格式合并文件: {all_raw_path}")
    
    # 6c. 分别保存 Core 和 Satellite 文件
    core_mask = df_scored["pool_type"] == "Core"
    satellite_mask = df_scored["pool_type"] == "Satellite"
    
    # Core 文件
    df_core_scored = df_scored[core_mask]
    df_core_raw = df_raw_out[core_mask]
    
    core_scored_path = os.path.join(output_dir, "all_core.csv")
    df_core_scored.to_csv(core_scored_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存 Core 文件: {core_scored_path} ({len(df_core_scored)} 行)")
    
    core_raw_path = os.path.join(output_dir, "all_core_raw.csv")
    df_core_raw.to_csv(core_raw_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存 Core 原始格式: {core_raw_path} ({len(df_core_raw)} 行)")
    
    # Satellite 文件
    df_satellite_scored = df_scored[satellite_mask]
    df_satellite_raw = df_raw_out[satellite_mask]
    
    satellite_scored_path = os.path.join(output_dir, "all_satellite.csv")
    df_satellite_scored.to_csv(satellite_scored_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存 Satellite 文件: {satellite_scored_path} ({len(df_satellite_scored)} 行)")
    
    satellite_raw_path = os.path.join(output_dir, "all_satellite_raw.csv")
    df_satellite_raw.to_csv(satellite_raw_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存 Satellite 原始格式: {satellite_raw_path} ({len(df_satellite_raw)} 行)")
    
    # 打印统计
    logger.info("=" * 60)
    logger.info("处理完成！")
    
    core_count = (df_scored["pool_type"] == "Core").sum()
    satellite_count = (df_scored["pool_type"] == "Satellite").sum()
    logger.info(f"Core 产品: {core_count} 个")
    logger.info(f"Satellite 产品: {satellite_count} 个")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

