"""
taa_signal_engine.py

TAA (Tactical Asset Allocation) weight-adjustment engine.
TAA(战术资产配置)权重调整引擎。

This module computes monthly portfolio weights by applying macro quadrant-based
tilts to a baseline SAA (Strategic Asset Allocation) allocation.
本模块通过对基准 SAA(战略资产配置)应用基于宏观象限的倾斜调整,
计算每月的投资组合权重。

Key concepts:
核心概念：
    - 16 strategies mapped to 4 asset classes (Cash, Bond, Equity, Alternative)
    - 16 个策略映射到 4 个资产大类（现金、债券、权益、另类）
    - 4 macro quadrants (Recovery, Overheat, Stagflation, Recession)
    - 4 个宏观象限（复苏、过热、滞涨、衰退）
    - Asset-class tilts are distributed to strategies proportionally by SAA weight
    - 资产大类的倾斜按 SAA 权重比例分配到各策略
    - Final weights are normalized to sum to 1 with no negatives
    - 最终权重归一化为和等于 1 且无负值

This file is standalone and does not import other project modules.
本文件独立运行，不导入项目其他模块。
"""

from typing import Dict, List
import numpy as np
import pandas as pd


# ==============================================================================
# Module-level constants
# 模块级常量
# ==============================================================================

# Fixed order of 16 strategy names (must be used for all DataFrames)
# 固定顺序的 16 个策略名称（所有 DataFrame 必须使用此顺序）
STRATEGY_NAMES: List[str] = [
    "Cash",
    "DepositFixedIncome",
    "PureBond",
    "NonStandardFixedIncome",
    "FixedIncomePlus",
    "OverseasBond",
    "BalancedFund",
    "EquityA",
    "EquityOverseas",
    "OverseasBalanced",
    "Commodity",
    "HedgeFund",
    "RealEstate",
    "PrivateEquity",
    "OverseasAlternative",
    "StructuredProduct",
]

# Mapping from strategy name to asset class
# 策略名称到资产大类的映射
STRATEGY_TO_ASSET_CLASS: Dict[str, str] = {
    "Cash": "Cash",
    "DepositFixedIncome": "Bond",
    "PureBond": "Bond",
    "NonStandardFixedIncome": "Bond",
    "FixedIncomePlus": "Bond",
    "OverseasBond": "Bond",
    "BalancedFund": "Equity",
    "EquityA": "Equity",
    "EquityOverseas": "Equity",
    "OverseasBalanced": "Equity",
    "Commodity": "Alternative",
    "HedgeFund": "Alternative",
    "RealEstate": "Alternative",
    "PrivateEquity": "Alternative",
    "OverseasAlternative": "Alternative",
    "StructuredProduct": "Alternative",
}

# Asset-class tilt matrix: quadrant -> asset_class -> tilt value
# These tilts represent how much to increase/decrease each asset class
# based on the current macro quadrant
# 资产大类倾斜矩阵：象限 -> 资产大类 -> 倾斜值
# 这些倾斜值表示根据当前宏观象限，各资产大类应增加/减少的权重
DELTA_ASSET: Dict[str, Dict[str, float]] = {
    "Recovery": {
        "Equity": 0.05,
        "Bond": -0.03,
        "Alternative": 0.02,
        "Cash": -0.04,
    },
    "Overheat": {
        "Equity": -0.02,
        "Bond": -0.02,
        "Alternative": 0.04,
        "Cash": 0.00,
    },
    "Stagflation": {
        "Equity": -0.04,
        "Bond": 0.00,
        "Alternative": 0.03,
        "Cash": -0.01,
    },
    "Recession": {
        "Equity": -0.05,
        "Bond": 0.05,
        "Alternative": -0.02,
        "Cash": 0.04,
    },
}

# List of valid quadrant names
# 有效象限名称列表
QUADRANT_NAMES: List[str] = ["Recovery", "Overheat", "Stagflation", "Recession"]

# List of asset class names
# 资产大类名称列表
ASSET_CLASS_NAMES: List[str] = ["Cash", "Bond", "Equity", "Alternative"]


# ==============================================================================
# Function implementations
# 函数实现
# ==============================================================================


def get_strategy_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per strategy and columns:
        - 'strategy'    : strategy name (string)
        - 'asset_class' : asset class name ("Cash", "Bond", "Equity", "Alternative")

    The order of rows matches the fixed strategy list (STRATEGY_NAMES).

    Returns:
        pd.DataFrame: Metadata DataFrame with shape (16, 2).

    ---
    返回一个 DataFrame, 每行对应一个策略，包含以下列：
        - 'strategy'    : 策略名称（字符串）
        - 'asset_class' : 资产大类名称("Cash", "Bond", "Equity", "Alternative")

    行顺序与固定策略列表(STRATEGY_NAMES)一致。

    返回值:
        pd.DataFrame: 形状为 (16, 2) 的元数据 DataFrame。
    """
    data = {
        "strategy": STRATEGY_NAMES,
        "asset_class": [STRATEGY_TO_ASSET_CLASS[s] for s in STRATEGY_NAMES],
    }
    return pd.DataFrame(data)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D weight vector:
        1) Set negative values to 0
        2) If the sum of weights > 0, divide by the sum so that weights sum to 1
        3) If the sum is 0 (all weights <= 0), return a uniform allocation

    Parameters:
        w : np.ndarray of shape (16,) - raw weight vector

    Returns:
        np.ndarray of shape (16,) with non-negative entries summing to 1.

    Example:
        >>> w = np.array([0.3, -0.1, 0.2, 0.0])
        >>> normalize_weights(w)
        array([0.6, 0.0, 0.4, 0.0])

    ---
    归一化一维权重向量：
        1) 将负值设为 0
        2) 如果权重之和 > 0，则除以总和使权重之和为 1
        3) 如果总和为 0（所有权重 <= 0），则返回均匀分配

    参数:
        w : 形状为 (16,) 的 np.ndarray - 原始权重向量

    返回值:
        形状为 (16,) 的 np.ndarray，所有元素非负且和为 1。

    示例:
        >>> w = np.array([0.3, -0.1, 0.2, 0.0])
        >>> normalize_weights(w)
        array([0.6, 0.0, 0.4, 0.0])
    """
    # Step 1: Clip negative values to 0
    # 第一步：将负值截断为 0
    w_clipped = np.maximum(w, 0.0)

    # Step 2: Compute sum
    # 第二步：计算总和
    total = w_clipped.sum()

    # Step 3: Normalize or return uniform
    # 第三步：归一化，或返回均匀分配
    if total > 0:
        return w_clipped / total
    else:
        # All weights were <= 0, return uniform allocation
        # 所有权重都 <= 0，返回均匀分配
        n = len(w)
        return np.ones(n) / n


def compute_raw_strategy_tilts(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame,
    delta_asset: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Compute unadjusted strategy-level tilts Δw^(0) for each month.

    Math (for each asset class AC and month t):
        Let Δw_asset,t(AC) be the tilt for asset class AC in month t,
        determined by the quadrant in that month.

        Let SAA_AC_sum = sum_{j in AC} w_SAA[j]

        Then for each strategy i in asset class AC:
            Δw_strategy^(0)_{i,t}
                = Δw_asset,t(AC) * w_SAA[i] / SAA_AC_sum

    Arguments:
        w_saa     : np.ndarray of shape (16,), SAA baseline weights (sum to 1).
                    The order must match the fixed strategy list.
        quadrants : pd.Series of length T, index = time index,
                    each value in {"Recovery", "Overheat", "Stagflation", "Recession"}.
        metadata  : DataFrame from get_strategy_metadata(), with at least:
                        'strategy', 'asset_class'
        delta_asset : dict mapping quadrant -> {asset_class -> tilt_value}

    Returns:
        delta_w_raw: pd.DataFrame of shape (T, 16)
                     index aligned with quadrants.index,
                     columns are strategy names in the fixed order.

    Edge cases:
        - If SAA_AC_sum == 0 for an asset class in this particular client,
          then all strategies in that asset class receive 0 tilt for that class.

    ---
    计算每月未调整的策略级倾斜量 Δw^(0)。

    数学公式（对于每个资产大类 AC 和月份 t）：
        设 Δw_asset,t(AC) 为月份 t 中资产大类 AC 的倾斜量，
        由该月的象限决定。

        设 SAA_AC_sum = sum_{j in AC} w_SAA[j]（该资产大类的 SAA 权重之和）

        则对于资产大类 AC 中的每个策略 i：
            Δw_strategy^(0)_{i,t}
                = Δw_asset,t(AC) * w_SAA[i] / SAA_AC_sum

    参数:
        w_saa     : 形状为 (16,) 的 np.ndarray，SAA 基准权重（和为 1）。
                    顺序必须与固定策略列表一致。
        quadrants : 长度为 T 的 pd.Series，索引为时间索引，
                    每个值属于 {"Recovery", "Overheat", "Stagflation", "Recession"}。
        metadata  : 来自 get_strategy_metadata() 的 DataFrame，至少包含：
                        'strategy', 'asset_class'
        delta_asset : 映射 象限 -> {资产大类 -> 倾斜值} 的字典

    返回值:
        delta_w_raw: 形状为 (T, 16) 的 pd.DataFrame
                     索引与 quadrants.index 对齐，
                     列为固定顺序的策略名称。

    边界情况:
        - 如果某资产大类的 SAA_AC_sum == 0（客户在该大类无配置），
          则该大类中所有策略的倾斜量为 0。
    """
    # Number of time periods
    # 时间周期数
    T = len(quadrants)
    n_strategies = len(STRATEGY_NAMES)

    # Initialize output array
    # 初始化输出数组
    delta_w_raw = np.zeros((T, n_strategies))

    # Pre-compute: for each asset class, which strategy indices belong to it
    # and what is the sum of SAA weights for that asset class
    # 预计算：每个资产大类包含哪些策略索引，以及该大类的 SAA 权重之和
    asset_class_indices: Dict[str, List[int]] = {ac: [] for ac in ASSET_CLASS_NAMES}
    asset_class_saa_sum: Dict[str, float] = {ac: 0.0 for ac in ASSET_CLASS_NAMES}

    for i, strategy in enumerate(STRATEGY_NAMES):
        ac = STRATEGY_TO_ASSET_CLASS[strategy]
        asset_class_indices[ac].append(i)
        asset_class_saa_sum[ac] += w_saa[i]

    # For each time period, compute strategy tilts
    # 对于每个时间周期，计算策略倾斜量
    for t, (time_idx, quadrant) in enumerate(quadrants.items()):
        # Get the asset-class tilts for this quadrant
        # 获取该象限的资产大类倾斜量
        ac_tilts = delta_asset[quadrant]

        # Distribute each asset-class tilt to its strategies
        # 将每个资产大类的倾斜量分配到其策略
        for ac in ASSET_CLASS_NAMES:
            ac_tilt = ac_tilts[ac]
            ac_sum = asset_class_saa_sum[ac]
            indices = asset_class_indices[ac]

            if ac_sum > 0:
                # Proportional allocation: Δw_i = Δw_AC * w_saa[i] / sum(w_saa in AC)
                # 比例分配：Δw_i = Δw_AC * w_saa[i] / sum(w_saa in AC)
                for i in indices:
                    delta_w_raw[t, i] = ac_tilt * w_saa[i] / ac_sum
            else:
                # Edge case: no SAA weight in this asset class
                # All strategies in this AC get 0 tilt
                # 边界情况：该资产大类无 SAA 权重
                # 该大类中所有策略的倾斜量为 0
                for i in indices:
                    delta_w_raw[t, i] = 0.0

    # Create DataFrame with proper index and columns
    # 创建具有正确索引和列名的 DataFrame
    delta_w_df = pd.DataFrame(
        delta_w_raw,
        index=quadrants.index,
        columns=STRATEGY_NAMES,
    )

    return delta_w_df


def compute_final_weights_over_time(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame,
    delta_asset: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Compute final monthly weights w_final,t for all strategies.

    Steps:
        1) Call compute_raw_strategy_tilts(...) to get Δw^(0)_{i,t}.
        2) For each month t:
               w_temp,t = w_saa + Δw^(0)_{t, :}
           (vector addition, same order of strategies)
        3) For each month t:
               w_final,t = normalize_weights(w_temp,t)
           so that:
               - no negative weights
               - each row sums to 1

    Arguments:
        w_saa     : np.ndarray of shape (16,), baseline SAA weights.
        quadrants : pd.Series indexed by time, specifying the macro quadrant per month.
        metadata  : DataFrame with 'strategy' and 'asset_class'.
        delta_asset : dict with quadrant -> asset_class -> tilt_value.

    Returns:
        final_weights: pd.DataFrame of shape (T, 16)
                       index = quadrants.index,
                       columns = strategy names in fixed order.

    ---
    计算所有策略的最终月度权重 w_final,t。

    步骤:
        1) 调用 compute_raw_strategy_tilts(...) 获取 Δw^(0)_{i,t}。
        2) 对于每个月份 t：
               w_temp,t = w_saa + Δw^(0)_{t, :}
           （向量加法，策略顺序相同）
        3) 对于每个月份 t：
               w_final,t = normalize_weights(w_temp,t)
           使得：
               - 无负权重
               - 每行之和为 1

    参数:
        w_saa     : 形状为 (16,) 的 np.ndarray，SAA 基准权重。
        quadrants : 按时间索引的 pd.Series，指定每月的宏观象限。
        metadata  : 包含 'strategy' 和 'asset_class' 的 DataFrame。
        delta_asset : 映射 象限 -> 资产大类 -> 倾斜值 的字典。

    返回值:
        final_weights: 形状为 (T, 16) 的 pd.DataFrame
                       索引 = quadrants.index，
                       列 = 固定顺序的策略名称。
    """
    # Step 1: Compute raw strategy tilts
    # 第一步：计算原始策略倾斜量
    delta_w_raw = compute_raw_strategy_tilts(w_saa, quadrants, metadata, delta_asset)

    # Step 2 & 3: Add tilts to SAA and normalize each row
    # We work with numpy arrays for efficiency, then convert back to DataFrame
    # 第二步和第三步：将倾斜量加到 SAA 上，并对每行归一化
    # 为提高效率使用 numpy 数组操作，然后转换回 DataFrame

    T = len(quadrants)
    final_weights_array = np.zeros((T, len(STRATEGY_NAMES)))

    for t in range(T):
        # w_temp = w_saa + Δw^(0)_{t, :}
        w_temp = w_saa + delta_w_raw.iloc[t].values

        # w_final = normalize(w_temp)
        # w_final = 归一化(w_temp)
        final_weights_array[t, :] = normalize_weights(w_temp)

    # Create DataFrame with proper index and columns
    # 创建具有正确索引和列名的 DataFrame
    final_weights_df = pd.DataFrame(
        final_weights_array,
        index=quadrants.index,
        columns=STRATEGY_NAMES,
    )

    return final_weights_df


# ==============================================================================
# Main demo block
# 主程序演示模块
# ==============================================================================

if __name__ == "__main__":
    # 1) Set random seed for reproducibility
    # 1) 设置随机种子以保证可复现性
    np.random.seed(42)

    # 2) Construct a synthetic SAA vector w_saa of length 16
    #    Draw random positive numbers and normalize to sum to 1
    # 2) 构造长度为 16 的合成 SAA 向量
    #    生成随机正数并归一化使其和为 1
    raw_weights = np.random.rand(16)
    w_saa = raw_weights / raw_weights.sum()

    print("=" * 60)
    print("TAA Signal Engine Demo")
    print("TAA 信号引擎演示")
    print("=" * 60)

    print("\n1. SAA Baseline Weights (synthetic):")
    print("1. SAA 基准权重（合成数据）:")
    print("-" * 40)
    for i, (name, weight) in enumerate(zip(STRATEGY_NAMES, w_saa)):
        print(f"   {name:25s}: {weight:.4f}")
    print(f"   {'Total':25s}: {w_saa.sum():.4f}")

    # 3) Construct a synthetic quadrants Series for 120 months
    #    Repeat each quadrant for 30 months (2.5 years each)
    #    Order: Recovery -> Overheat -> Stagflation -> Recession (one full cycle = 10 years)
    # 3) 构造 120 个月的合成象限序列
    #    每个象限重复 30 个月（各 2.5 年）
    #    顺序：复苏 -> 过热 -> 滞涨 -> 衰退（一个完整周期 = 10 年）
    months_per_quadrant = 30
    quadrant_sequence = []
    for q in QUADRANT_NAMES:
        quadrant_sequence.extend([q] * months_per_quadrant)

    # Create a monthly DatetimeIndex starting from 2015-01-01
    # 创建从 2015-01-01 开始的月度时间索引
    date_index = pd.date_range(start="2015-01-01", periods=120, freq="MS")
    quadrants = pd.Series(quadrant_sequence, index=date_index, name="quadrant")

    print("\n2. Macro Quadrant Sequence (first 12 months):")
    print("2. 宏观象限序列（前 12 个月）:")
    print("-" * 40)
    print(quadrants.head(12).to_string())

    # 4) Get strategy metadata
    # 4) 获取策略元数据
    metadata = get_strategy_metadata()

    print("\n3. Strategy Metadata (asset class mapping):")
    print("3. 策略元数据（资产大类映射）:")
    print("-" * 40)
    print(metadata.to_string(index=False))

    # 5) Use the delta_asset dict defined at module level
    # 5) 使用模块级定义的 delta_asset 字典
    print("\n4. Asset-Class Tilt Matrix (DELTA_ASSET):")
    print("4. 资产大类倾斜矩阵 (DELTA_ASSET):")
    print("-" * 40)
    for quadrant, tilts in DELTA_ASSET.items():
        tilt_str = ", ".join([f"{ac}: {v:+.2f}" for ac, v in tilts.items()])
        print(f"   {quadrant:12s}: {tilt_str}")

    # 6) Call compute_final_weights_over_time
    # 6) 调用 compute_final_weights_over_time 计算最终权重
    final_weights = compute_final_weights_over_time(
        w_saa=w_saa,
        quadrants=quadrants,
        metadata=metadata,
        delta_asset=DELTA_ASSET,
    )

    print("\n5. Final Weights (first 5 rows, rounded to 4 decimals):")
    print("5. 最终权重（前 5 行，保留 4 位小数）:")
    print("-" * 40)
    print(final_weights.head().round(4).to_string())

    # 7) Quick check: each row sums to ~1.0
    # 7) 快速检查：每行之和应约等于 1.0
    row_sums = final_weights.sum(axis=1)
    print("\n6. Row Sum Verification (should all be ~1.0):")
    print("6. 行和验证（应全部约等于 1.0）:")
    print("-" * 40)
    print(f"   Min row sum / 最小行和: {row_sums.min():.6f}")
    print(f"   Max row sum / 最大行和: {row_sums.max():.6f}")
    print(f"   Mean row sum / 平均行和: {row_sums.mean():.6f}")

    # 8) Group-by asset class: show average weight by asset class for each quadrant
    #    This demonstrates how TAA shifts weights at the asset-class level
    # 8) 按资产大类分组：显示每个象限下各资产大类的平均权重
    #    这展示了 TAA 如何在资产大类层面调整权重
    print("\n7. Average Weights by Asset Class (grouped by quadrant):")
    print("7. 按资产大类的平均权重（按象限分组）:")
    print("-" * 40)

    # Add quadrant column to final_weights for grouping
    # 为 final_weights 添加象限列以便分组
    final_weights_with_quadrant = final_weights.copy()
    final_weights_with_quadrant["quadrant"] = quadrants.values

    # Compute asset-class weights for each row
    # 计算每行的资产大类权重
    asset_class_weights = pd.DataFrame(index=final_weights.index)
    for ac in ASSET_CLASS_NAMES:
        # Get strategies belonging to this asset class
        # 获取属于该资产大类的策略
        ac_strategies = [s for s in STRATEGY_NAMES if STRATEGY_TO_ASSET_CLASS[s] == ac]
        asset_class_weights[ac] = final_weights[ac_strategies].sum(axis=1)

    asset_class_weights["quadrant"] = quadrants.values

    # Group by quadrant and compute mean
    # 按象限分组并计算平均值
    avg_by_quadrant = asset_class_weights.groupby("quadrant").mean()

    # Also compute SAA baseline asset-class weights for comparison
    # 同时计算 SAA 基准的资产大类权重以作比较
    saa_ac_weights = {}
    for ac in ASSET_CLASS_NAMES:
        ac_strategies = [s for s in STRATEGY_NAMES if STRATEGY_TO_ASSET_CLASS[s] == ac]
        ac_indices = [STRATEGY_NAMES.index(s) for s in ac_strategies]
        saa_ac_weights[ac] = sum(w_saa[i] for i in ac_indices)

    print("\n   SAA Baseline (for comparison):")
    print("   SAA 基准（用于对比）:")
    saa_str = ", ".join([f"{ac}: {v:.4f}" for ac, v in saa_ac_weights.items()])
    print(f"   {saa_str}")

    print("\n   TAA Average by Quadrant:")
    print("   TAA 按象限的平均权重:")
    print(avg_by_quadrant.round(4).to_string())

    print("\n" + "=" * 60)
    print("Demo Complete! / 演示完成！")
    print("=" * 60)
