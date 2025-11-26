"""
taa_signal_engine.py — TAA Weight-Adjustment Engine

This module implements the Tactical Asset Allocation (TAA) weight-adjustment engine:
    1. Maps 16 strategies to 5 higher-level asset classes （将 16 个子策略映射到 5 个高级资产类别）
    2. Applies macro-quadrant-based tilts to asset classes （根据宏观象限对资产类别应用倾斜调整）
    3. Distributes tilts to strategies proportional to SAA weights （按 SAA 权重比例将倾斜分配到各子策略）
    4. Applies strategy-specific sensitivity coefficients (beta) （应用策略特定的敏感度系数（beta））
    5. Normalizes to produce final monthly weights （归一化生成最终的月度权重）

Purpose:
    - Learn vectorized operations with NumPy and pandas
    - Practice group-by logic by asset class
    - Understand TAA mathematical formulation

Usage:
    python taa_signal_engine.py

Dependencies:
    - numpy
    - pandas
    - Python 3.9+ (compatible with 3.11)

Author: TAA Learning Project
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from typing import Dict, List

# =============================================================================
# Constants: Strategy Names (16 strategies)
# 常量：策略名称（16 个策略）
# =============================================================================

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

# =============================================================================
# Constants: Asset Classes (5 higher-level categories)
# 常量：资产类别（5 个高级类别）
# =============================================================================

ASSET_CLASSES: List[str] = [
    "Cash",
    "Bond",
    "Equity",
    "Commodity",
    "Alternative",
]

# =============================================================================
# Constants: Strategy-to-Asset-Class Mapping
# 常量：策略到资产类别的映射
# =============================================================================

# Each of the 16 strategies maps to one of the 5 asset classes

STRATEGY_TO_ASSET: Dict[str, str] = {
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
    "Commodity": "Commodity",
    "HedgeFund": "Alternative",
    "RealEstate": "Alternative",
    "PrivateEquity": "Alternative",
    "OverseasAlternative": "Alternative",
    "StructuredProduct": "Alternative",
}

# =============================================================================
# Constants: Macro Quadrants (4 economic regimes)
# 常量：宏观象限（4 个经济周期）
# =============================================================================

QUADRANTS: List[str] = [
    "Recovery",
    "Overheat",
    "Stagflation",
    "Recession",
]

# =============================================================================
# Constants: Asset-Class Tilts by Quadrant (Δw_asset)
# =============================================================================

# For each quadrant, define how much to tilt each asset class
# Positive values = increase allocation, Negative values = decrease allocation
# These are tilts (Δw), not absolute weights
DELTA_ASSET: Dict[str, Dict[str, float]] = {
    "Recovery": {
        "Equity": 0.05,
        "Bond": -0.03,
        "Commodity": 0.02,
        "Alternative": 0.02,
        "Cash": -0.04,
    },
    "Overheat": {
        "Equity": -0.02,
        "Bond": -0.03,
        "Commodity": 0.05,
        "Alternative": 0.03,
        "Cash": 0.00,
    },
    "Stagflation": {
        "Equity": -0.04,
        "Bond": 0.00,
        "Commodity": 0.04,
        "Alternative": 0.02,
        "Cash": -0.02,
    },
    "Recession": {
        "Equity": -0.05,
        "Bond": 0.05,
        "Commodity": -0.02,
        "Alternative": -0.02,
        "Cash": 0.04,
    },
}

# =============================================================================
# Constants: Sensitivity Coefficients (Beta) for Each Strategy
# =============================================================================

# Beta represents how sensitive each strategy is to TAA tilts
# Lower beta (0.3-0.5): Cash-like, short-term instruments
# Medium beta (0.7-0.9): Core bonds
# Higher beta (1.0-1.3): Equities & alternatives

BETA_VALUES: np.ndarray = np.array([
    0.3,   # Cash: low sensitivity
    0.7,   # DepositFixedIncome: medium-low
    0.8,   # PureBond: medium
    0.8,   # NonStandardFixedIncome: medium
    0.9,   # FixedIncomePlus: medium-high
    0.9,   # OverseasBond: medium-high
    1.0,   # BalancedFund: high
    1.1,   # EquityA: high
    1.1,   # EquityOverseas: high
    1.0,   # OverseasBalanced: high
    1.2,   # Commodity: very high
    1.1,   # HedgeFund: high
    1.1,   # RealEstate: high
    1.2,   # PrivateEquity: very high
    1.2,   # OverseasAlternative: very high
    1.0,   # StructuredProduct: high
], dtype=np.float64)


# =============================================================================
# Function 1: Get Strategy Metadata
# =============================================================================

def get_strategy_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame containing metadata for all 16 strategies.

    The DataFrame has one row per strategy and the following columns:
        - 'strategy'    : Strategy name (string)
        - 'asset_class' : Higher-level asset class (string)
        - 'beta'        : Sensitivity coefficient (float)

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (16, 3) with strategy metadata.

    Example
    -------
    >>> metadata = get_strategy_metadata()
    >>> print(metadata.head(3))
               strategy asset_class  beta
    0              Cash        Cash   0.3
    1  DepositFixedIncome        Bond   0.7
    2          PureBond        Bond   0.8
    """
    # Build the metadata DataFrame from our constants
    # We iterate over STRATEGY_NAMES to maintain consistent ordering
    metadata = pd.DataFrame({
        "strategy": STRATEGY_NAMES,
        "asset_class": [STRATEGY_TO_ASSET[s] for s in STRATEGY_NAMES],
        "beta": BETA_VALUES,
    })

    return metadata


# =============================================================================
# Function 2: Normalize Weights
# =============================================================================

def normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D weight vector to ensure valid portfolio weights.

    The normalization process:
        1. Clip negative values to 0 (no short positions allowed)
        2. If sum > 0, divide by sum so weights sum to 1
        3. If all values are <= 0 (sum == 0), fall back to uniform allocation

    Parameters
    ----------
    w : np.ndarray
        A 1D array of raw weights (may contain negatives, may not sum to 1).

    Returns
    -------
    np.ndarray
        A 1D array of normalized weights:
        - All values >= 0
        - Sum equals 1.0
        - Same shape as input

    Example
    -------
    >>> w = np.array([0.5, -0.1, 0.3, 0.2])
    >>> normalize_weights(w)
    array([0.5, 0.0, 0.3, 0.2])  # Sum = 1.0, negative clipped

    >>> w = np.array([-0.1, -0.2, -0.3])
    >>> normalize_weights(w)
    array([0.333..., 0.333..., 0.333...])  # Uniform fallback
    """
    # Step 1: Clip negative values to 0
    # np.maximum(w, 0) returns element-wise max of w and 0
    # This is equivalent to: w[w < 0] = 0, but creates a new array
    w_clipped = np.maximum(w, 0.0)

    # Step 2: Calculate the sum of clipped weights
    total = w_clipped.sum()

    # Step 3: Normalize or fall back to uniform
    if total > 0:
        # Normal case: divide by sum to get weights summing to 1
        w_normalized = w_clipped / total
    else:
        # Edge case: all weights were <= 0
        # Fall back to uniform allocation (equal weight to each asset)
        n = len(w)
        w_normalized = np.ones(n, dtype=np.float64) / n

    return w_normalized


# =============================================================================
# Function 3: Compute Raw Strategy Tilts
# =============================================================================

def compute_raw_strategy_tilts(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute unadjusted (raw) tilts for each strategy at each time period.

    For each month t and each strategy i, the raw tilt is calculated by
    distributing the asset-class tilt proportionally within the asset class:

        Δw_strategy^(0)_{i,t} = Δw_asset,t(AC) * w_saa[i] / SAA_AC_sum

    Where:
        - Δw_asset,t(AC) is the tilt for asset class AC at time t
        - SAA_AC_sum is the sum of SAA weights for all strategies in AC
        - w_saa[i] is the SAA weight of strategy i

    Parameters
    ----------
    w_saa : np.ndarray
        SAA weight vector of shape (16,). Must sum to 1.
    quadrants : pd.Series
        Series of length T with quadrant labels.
        Values must be in {"Recovery", "Overheat", "Stagflation", "Recession"}.
    metadata : pd.DataFrame
        Strategy metadata from get_strategy_metadata().
        Must contain 'strategy' and 'asset_class' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, 16) containing raw tilts.
        - Index: aligned with quadrants.index
        - Columns: strategy names

    Example
    -------
    >>> w_saa = np.array([0.1, 0.1, ...])  # 16 weights
    >>> quadrants = pd.Series(["Recovery", "Overheat", ...])
    >>> metadata = get_strategy_metadata()
    >>> delta_w_raw = compute_raw_strategy_tilts(w_saa, quadrants, metadata)
    >>> print(delta_w_raw.shape)
    (T, 16)
    """
    # Get the number of time periods and strategies
    T = len(quadrants)
    n_strategies = len(STRATEGY_NAMES)

    # -------------------------------------------------------------------------
    # Step 1: Compute SAA sum for each asset class
    # -------------------------------------------------------------------------
    # Create a mapping from asset class to the sum of SAA weights in that class
    # This is used to proportionally distribute tilts within each asset class

    # First, create a Series mapping strategy index to asset class
    asset_classes_series = metadata["asset_class"]

    # Compute sum of SAA weights per asset class
    # We'll use pandas groupby for clarity
    saa_series = pd.Series(w_saa, index=STRATEGY_NAMES)
    saa_by_asset_class = saa_series.groupby(
        metadata.set_index("strategy")["asset_class"]
    ).sum()

    # -------------------------------------------------------------------------
    # Step 2: For each strategy, compute the proportion within its asset class
    # -------------------------------------------------------------------------
    # proportion[i] = w_saa[i] / SAA_sum_of_asset_class[i]
    # Handle edge case where SAA_sum = 0 (no allocation to that asset class)

    proportions = np.zeros(n_strategies, dtype=np.float64)

    for i, strategy in enumerate(STRATEGY_NAMES):
        asset_class = STRATEGY_TO_ASSET[strategy]
        ac_sum = saa_by_asset_class[asset_class]

        if ac_sum > 0:
            proportions[i] = w_saa[i] / ac_sum
        else:
            # If no SAA allocation to this asset class, proportion is 0
            proportions[i] = 0.0

    # -------------------------------------------------------------------------
    # Step 3: Build the raw tilts matrix (T x 16)
    # -------------------------------------------------------------------------
    # For each time period t:
    #   - Get the quadrant for that period
    #   - Get the asset-class tilt vector for that quadrant
    #   - Distribute to strategies using proportions

    # Initialize output array
    delta_w_raw = np.zeros((T, n_strategies), dtype=np.float64)

    for t in range(T):
        quadrant = quadrants.iloc[t]

        # Get the asset-class tilts for this quadrant
        ac_tilts = DELTA_ASSET[quadrant]

        # For each strategy, compute raw tilt
        for i, strategy in enumerate(STRATEGY_NAMES):
            asset_class = STRATEGY_TO_ASSET[strategy]
            # Raw tilt = asset class tilt * proportion within asset class
            delta_w_raw[t, i] = ac_tilts[asset_class] * proportions[i]

    # Convert to DataFrame with proper index and columns
    delta_w_raw_df = pd.DataFrame(
        data=delta_w_raw,
        index=quadrants.index,
        columns=STRATEGY_NAMES
    )

    return delta_w_raw_df


# =============================================================================
# Function 4: Apply Beta Adjustment
# =============================================================================

def apply_beta_adjustment(
    delta_w_raw: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply sensitivity coefficients (beta) to the raw strategy tilts.

    The beta adjustment scales each strategy's tilt by its sensitivity:

        Δw_strategy_{i,t} = beta_i * Δw_strategy^(0)_{i,t}

    Strategies with higher beta are more responsive to TAA signals.

    Parameters
    ----------
    delta_w_raw : pd.DataFrame
        DataFrame of shape (T, 16) with raw tilts from compute_raw_strategy_tilts.
    metadata : pd.DataFrame
        Strategy metadata from get_strategy_metadata().
        Must contain 'strategy' and 'beta' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, 16) with beta-adjusted tilts.
        - Index: same as delta_w_raw
        - Columns: strategy names

    Example
    -------
    >>> delta_w_raw = compute_raw_strategy_tilts(...)
    >>> metadata = get_strategy_metadata()
    >>> delta_w_beta = apply_beta_adjustment(delta_w_raw, metadata)
    """
    # Extract beta values as a 1D array aligned with column order
    # Ensure beta is in the same order as the DataFrame columns
    beta_values = metadata.set_index("strategy").loc[
        delta_w_raw.columns, "beta"
    ].values

    # -------------------------------------------------------------------------
    # Vectorized multiplication using NumPy broadcasting
    # -------------------------------------------------------------------------
    # delta_w_raw is (T, 16) and beta_values is (16,)
    # NumPy broadcasts (16,) to match (T, 16) and multiplies element-wise

    delta_w_beta_values = delta_w_raw.values * beta_values

    # Wrap back into a DataFrame with the same index and columns
    delta_w_beta = pd.DataFrame(
        data=delta_w_beta_values,
        index=delta_w_raw.index,
        columns=delta_w_raw.columns
    )

    return delta_w_beta


# =============================================================================
# Function 5: Compute Final Weights Over Time
# =============================================================================

def compute_final_weights_over_time(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the final TAA portfolio weights for each time period.

    This is the main orchestration function that:
        1. Computes raw tilts (Δw_strategy^(0)) from quadrant signals
        2. Applies beta adjustment to get final tilts (Δw_strategy)
        3. Adds tilts to SAA weights: w_raw = w_saa + Δw_strategy
        4. Normalizes each period's weights (clip negatives, sum to 1)

    Parameters
    ----------
    w_saa : np.ndarray
        SAA weight vector of shape (16,). Should sum to 1.
    quadrants : pd.Series
        Series of length T with quadrant labels.
    metadata : pd.DataFrame
        Strategy metadata from get_strategy_metadata().

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, 16) with final normalized weights.
        - Index: aligned with quadrants.index
        - Columns: strategy names
        - Each row sums to 1.0
        - All values >= 0

    Example
    -------
    >>> w_saa = generate_saa_weights()
    >>> quadrants = generate_quadrant_series(120)
    >>> metadata = get_strategy_metadata()
    >>> final_weights = compute_final_weights_over_time(w_saa, quadrants, metadata)
    >>> print(final_weights.sum(axis=1).head())  # All ~1.0
    """
    # Step 1: Compute raw tilts
    delta_w_raw = compute_raw_strategy_tilts(w_saa, quadrants, metadata)

    # Step 2: Apply beta adjustment
    delta_w_beta = apply_beta_adjustment(delta_w_raw, metadata)

    # Step 3: Add tilts to SAA weights
    # Broadcasting: w_saa is (16,), delta_w_beta is (T, 16)
    # Result: w_raw is (T, 16)
    w_raw = w_saa + delta_w_beta.values

    # Step 4: Normalize each row (each time period)
    # We need to apply normalize_weights to each row
    T = w_raw.shape[0]
    w_final = np.zeros_like(w_raw)

    for t in range(T):
        w_final[t, :] = normalize_weights(w_raw[t, :])

    # Wrap into DataFrame
    final_weights_df = pd.DataFrame(
        data=w_final,
        index=quadrants.index,
        columns=STRATEGY_NAMES
    )

    return final_weights_df


# =============================================================================
# Main Demo Block
# =============================================================================

if __name__ == "__main__":
    """
    Demo script to showcase the TAA signal engine.

    Run this file directly to see example outputs:
        python taa_signal_engine.py
    """
    print("=" * 70)
    print("TAA Signal Engine - Demo")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Set random seed for reproducibility
    # -------------------------------------------------------------------------
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Step 2: Generate synthetic SAA weight vector
    # -------------------------------------------------------------------------
    # Create random positive weights and normalize to sum to 1
    raw_weights = np.random.rand(16)
    w_saa = raw_weights / raw_weights.sum()

    print("\n" + "-" * 70)
    print("1. Synthetic SAA Weights")
    print("-" * 70)
    weights_display = pd.Series(w_saa, index=STRATEGY_NAMES, name="Weight")
    print(weights_display.to_string())
    print(f"\nTotal: {w_saa.sum():.6f}")

    # -------------------------------------------------------------------------
    # Step 3: Generate synthetic quadrant series (120 months)
    # -------------------------------------------------------------------------
    # Each quadrant persists for 30 months (30 * 4 = 120)
    n_months = 120
    months_per_quadrant = n_months // 4

    quadrant_path = np.repeat(QUADRANTS, months_per_quadrant)
    quadrants = pd.Series(quadrant_path, index=range(n_months), name="Quadrant")

    print("\n" + "-" * 70)
    print("2. Quadrant Path")
    print("-" * 70)
    print(f"Total months: {len(quadrants)}")
    print(f"\nQuadrant distribution:")
    print(quadrants.value_counts().sort_index())
    print(f"\nFirst 10 months: {quadrants.head(10).tolist()}")

    # -------------------------------------------------------------------------
    # Step 4: Get strategy metadata
    # -------------------------------------------------------------------------
    metadata = get_strategy_metadata()

    print("\n" + "-" * 70)
    print("3. Strategy Metadata")
    print("-" * 70)
    print(metadata.to_string())

    # -------------------------------------------------------------------------
    # Step 5: Compute final weights over time
    # -------------------------------------------------------------------------
    final_weights = compute_final_weights_over_time(w_saa, quadrants, metadata)

    print("\n" + "-" * 70)
    print("4. Final TAA Weights")
    print("-" * 70)
    print(f"Shape: {final_weights.shape} (months × strategies)")

    print("\n--- First 5 months ---")
    print(final_weights.head().to_string())

    print("\n--- Last 5 months ---")
    print(final_weights.tail().to_string())

    # -------------------------------------------------------------------------
    # Step 6: Verify that each row sums to 1.0
    # -------------------------------------------------------------------------
    row_sums = final_weights.sum(axis=1)

    print("\n" + "-" * 70)
    print("5. Row Sum Verification")
    print("-" * 70)
    print(f"Min row sum: {row_sums.min():.10f}")
    print(f"Max row sum: {row_sums.max():.10f}")
    print(f"Mean row sum: {row_sums.mean():.10f}")
    print(f"All rows sum to ~1.0: {np.allclose(row_sums, 1.0)}")

    # -------------------------------------------------------------------------
    # Step 7: Show average tilt per asset class (optional inspection)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. Average Weight by Asset Class")
    print("-" * 70)

    # Compute mean weight per strategy, then group by asset class
    mean_weights = final_weights.mean()

    # Create a mapping from strategy to asset class
    strategy_to_ac = pd.Series(
        [STRATEGY_TO_ASSET[s] for s in STRATEGY_NAMES],
        index=STRATEGY_NAMES
    )

    # Group by asset class and sum
    mean_by_ac = mean_weights.groupby(strategy_to_ac).sum()
    print("Average TAA allocation by asset class:")
    print(mean_by_ac.sort_values(ascending=False).to_string())

    # Compare to SAA allocation by asset class
    saa_series = pd.Series(w_saa, index=STRATEGY_NAMES)
    saa_by_ac = saa_series.groupby(strategy_to_ac).sum()
    print("\nOriginal SAA allocation by asset class:")
    print(saa_by_ac.sort_values(ascending=False).to_string())

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nYou can import these functions in other modules:")
    print("  from core.taa_signal_engine import compute_final_weights_over_time")
    print("  from core.taa_signal_engine import get_strategy_metadata")
    print("=" * 70)

