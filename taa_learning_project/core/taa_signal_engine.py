"""
taa_signal_engine.py

TAA (Tactical Asset Allocation) weight-adjustment engine.

This module computes monthly portfolio weights by applying macro quadrant-based
tilts to a baseline SAA (Strategic Asset Allocation) allocation.

Key concepts:
    - 16 strategies mapped to 4 asset classes (Cash, Bond, Equity, Alternative)
    - 4 macro quadrants (Recovery, Overheat, Stagflation, Recession)
    - Asset-class tilts are distributed to strategies proportionally by SAA weight
    - Final weights are normalized to sum to 1 with no negatives

This file is standalone and does not import other project modules.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


# ==============================================================================
# Module-level constants
# ==============================================================================

# Fixed order of 16 strategy names (must be used for all DataFrames)
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
QUADRANT_NAMES: List[str] = ["Recovery", "Overheat", "Stagflation", "Recession"]

# List of asset class names
ASSET_CLASS_NAMES: List[str] = ["Cash", "Bond", "Equity", "Alternative"]


# ==============================================================================
# Function implementations
# ==============================================================================


def get_strategy_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per strategy and columns:
        - 'strategy'    : strategy name (string)
        - 'asset_class' : asset class name ("Cash", "Bond", "Equity", "Alternative")

    The order of rows matches the fixed strategy list (STRATEGY_NAMES).

    Returns:
        pd.DataFrame: Metadata DataFrame with shape (16, 2).
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
    """
    # Step 1: Clip negative values to 0
    w_clipped = np.maximum(w, 0.0)

    # Step 2: Compute sum
    total = w_clipped.sum()

    # Step 3: Normalize or return uniform
    if total > 0:
        return w_clipped / total
    else:
        # All weights were <= 0, return uniform allocation
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
    """
    # Number of time periods
    T = len(quadrants)
    n_strategies = len(STRATEGY_NAMES)

    # Initialize output array
    delta_w_raw = np.zeros((T, n_strategies))

    # Pre-compute: for each asset class, which strategy indices belong to it
    # and what is the sum of SAA weights for that asset class
    asset_class_indices: Dict[str, List[int]] = {ac: [] for ac in ASSET_CLASS_NAMES}
    asset_class_saa_sum: Dict[str, float] = {ac: 0.0 for ac in ASSET_CLASS_NAMES}

    for i, strategy in enumerate(STRATEGY_NAMES):
        ac = STRATEGY_TO_ASSET_CLASS[strategy]
        asset_class_indices[ac].append(i)
        asset_class_saa_sum[ac] += w_saa[i]

    # For each time period, compute strategy tilts
    for t, (time_idx, quadrant) in enumerate(quadrants.items()):
        # Get the asset-class tilts for this quadrant
        ac_tilts = delta_asset[quadrant]

        # Distribute each asset-class tilt to its strategies
        for ac in ASSET_CLASS_NAMES:
            ac_tilt = ac_tilts[ac]
            ac_sum = asset_class_saa_sum[ac]
            indices = asset_class_indices[ac]

            if ac_sum > 0:
                # Proportional allocation: Δw_i = Δw_AC * w_saa[i] / sum(w_saa in AC)
                for i in indices:
                    delta_w_raw[t, i] = ac_tilt * w_saa[i] / ac_sum
            else:
                # Edge case: no SAA weight in this asset class
                # All strategies in this AC get 0 tilt
                for i in indices:
                    delta_w_raw[t, i] = 0.0

    # Create DataFrame with proper index and columns
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
    """
    # Step 1: Compute raw strategy tilts
    delta_w_raw = compute_raw_strategy_tilts(w_saa, quadrants, metadata, delta_asset)

    # Step 2 & 3: Add tilts to SAA and normalize each row
    # We work with numpy arrays for efficiency, then convert back to DataFrame

    T = len(quadrants)
    final_weights_array = np.zeros((T, len(STRATEGY_NAMES)))

    for t in range(T):
        # w_temp = w_saa + Δw^(0)_{t, :}
        w_temp = w_saa + delta_w_raw.iloc[t].values

        # w_final = normalize(w_temp)
        final_weights_array[t, :] = normalize_weights(w_temp)

    # Create DataFrame with proper index and columns
    final_weights_df = pd.DataFrame(
        final_weights_array,
        index=quadrants.index,
        columns=STRATEGY_NAMES,
    )

    return final_weights_df


# ==============================================================================
# Main demo block
# ==============================================================================

if __name__ == "__main__":
    # 1) Set random seed for reproducibility
    np.random.seed(42)

    # 2) Construct a synthetic SAA vector w_saa of length 16
    #    Draw random positive numbers and normalize to sum to 1
    raw_weights = np.random.rand(16)
    w_saa = raw_weights / raw_weights.sum()

    print("=" * 60)
    print("TAA Signal Engine Demo")
    print("=" * 60)

    print("\n1. SAA Baseline Weights (synthetic):")
    print("-" * 40)
    for i, (name, weight) in enumerate(zip(STRATEGY_NAMES, w_saa)):
        print(f"   {name:25s}: {weight:.4f}")
    print(f"   {'Total':25s}: {w_saa.sum():.4f}")

    # 3) Construct a synthetic quadrants Series for 120 months
    #    Repeat each quadrant for 30 months (2.5 years each)
    #    Order: Recovery -> Overheat -> Stagflation -> Recession (one full cycle = 10 years)
    months_per_quadrant = 30
    quadrant_sequence = []
    for q in QUADRANT_NAMES:
        quadrant_sequence.extend([q] * months_per_quadrant)

    # Create a monthly DatetimeIndex starting from 2015-01-01
    date_index = pd.date_range(start="2015-01-01", periods=120, freq="MS")
    quadrants = pd.Series(quadrant_sequence, index=date_index, name="quadrant")

    print("\n2. Macro Quadrant Sequence (first 12 months):")
    print("-" * 40)
    print(quadrants.head(12).to_string())

    # 4) Get strategy metadata
    metadata = get_strategy_metadata()

    print("\n3. Strategy Metadata (asset class mapping):")
    print("-" * 40)
    print(metadata.to_string(index=False))

    # 5) Use the delta_asset dict defined at module level
    print("\n4. Asset-Class Tilt Matrix (DELTA_ASSET):")
    print("-" * 40)
    for quadrant, tilts in DELTA_ASSET.items():
        tilt_str = ", ".join([f"{ac}: {v:+.2f}" for ac, v in tilts.items()])
        print(f"   {quadrant:12s}: {tilt_str}")

    # 6) Call compute_final_weights_over_time
    final_weights = compute_final_weights_over_time(
        w_saa=w_saa,
        quadrants=quadrants,
        metadata=metadata,
        delta_asset=DELTA_ASSET,
    )

    print("\n5. Final Weights (first 5 rows, rounded to 4 decimals):")
    print("-" * 40)
    print(final_weights.head().round(4).to_string())

    # 7) Quick check: each row sums to ~1.0
    row_sums = final_weights.sum(axis=1)
    print("\n6. Row Sum Verification (should all be ~1.0):")
    print("-" * 40)
    print(f"   Min row sum: {row_sums.min():.6f}")
    print(f"   Max row sum: {row_sums.max():.6f}")
    print(f"   Mean row sum: {row_sums.mean():.6f}")

    # 8) Group-by asset class: show average weight by asset class for each quadrant
    #    This demonstrates how TAA shifts weights at the asset-class level
    print("\n7. Average Weights by Asset Class (grouped by quadrant):")
    print("-" * 40)

    # Add quadrant column to final_weights for grouping
    final_weights_with_quadrant = final_weights.copy()
    final_weights_with_quadrant["quadrant"] = quadrants.values

    # Compute asset-class weights for each row
    asset_class_weights = pd.DataFrame(index=final_weights.index)
    for ac in ASSET_CLASS_NAMES:
        # Get strategies belonging to this asset class
        ac_strategies = [s for s in STRATEGY_NAMES if STRATEGY_TO_ASSET_CLASS[s] == ac]
        asset_class_weights[ac] = final_weights[ac_strategies].sum(axis=1)

    asset_class_weights["quadrant"] = quadrants.values

    # Group by quadrant and compute mean
    avg_by_quadrant = asset_class_weights.groupby("quadrant").mean()

    # Also compute SAA baseline asset-class weights for comparison
    saa_ac_weights = {}
    for ac in ASSET_CLASS_NAMES:
        ac_strategies = [s for s in STRATEGY_NAMES if STRATEGY_TO_ASSET_CLASS[s] == ac]
        ac_indices = [STRATEGY_NAMES.index(s) for s in ac_strategies]
        saa_ac_weights[ac] = sum(w_saa[i] for i in ac_indices)

    print("\n   SAA Baseline (for comparison):")
    saa_str = ", ".join([f"{ac}: {v:.4f}" for ac, v in saa_ac_weights.items()])
    print(f"   {saa_str}")

    print("\n   TAA Average by Quadrant:")
    print(avg_by_quadrant.round(4).to_string())

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

