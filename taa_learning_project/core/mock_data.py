"""
mock_data.py — Synthetic TAA Dataset Generator

This module generates synthetic inputs for a Tactical Asset Allocation (TAA) project:
    1. SAA (Strategic Asset Allocation) weight vector — 16 strategies
    2. Monthly returns DataFrame — (T x 16) synthetic returns
    3. Macro quadrant path — (T x 1) economic regime labels

Purpose:
    - Learn NumPy vector operations
    - Learn pandas DataFrame/Series construction
    - Practice reproducible random data generation

Usage:
    python mock_data.py

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
from typing import List, Tuple

# =============================================================================
# Constants
# =============================================================================

# 16 strategy names used across the TAA project
# These represent different asset classes and investment strategies
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

# 4 macro-economic quadrants representing different market regimes
QUADRANTS: List[str] = [
    "Recovery",      # Economic expansion, low inflation
    "Overheat",      # Economic expansion, high inflation
    "Stagflation",   # Economic contraction, high inflation
    "Recession",     # Economic contraction, low inflation
]


# =============================================================================
# Function 1: Generate SAA Weights
# =============================================================================

def generate_saa_weights(n_strategies: int = 16) -> np.ndarray:
    """
    Generate a random Strategic Asset Allocation (SAA) weight vector.

    The weights are:
        - Non-negative (all >= 0)
        - Sum to 1.0 (fully invested portfolio)
        - Randomly generated but reproducible with seed

    Parameters
    ----------
    n_strategies : int, optional
        Number of strategies/assets in the portfolio.
        Default is 16 (matching STRATEGY_NAMES).

    Returns
    -------
    np.ndarray
        A 1D array of shape (n_strategies,) containing portfolio weights.
        Each weight is between 0 and 1, and all weights sum to 1.

    Example
    -------
    >>> weights = generate_saa_weights()
    >>> print(weights.shape)
    (16,)
    >>> print(weights.sum())
    1.0
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Step 1: Generate random positive numbers using uniform distribution
    # np.random.rand() returns values in [0, 1)
    raw_weights = np.random.rand(n_strategies)

    # Step 2: Normalize to sum to 1
    # This is a common technique: divide each weight by the total sum
    normalized_weights = raw_weights / raw_weights.sum()

    return normalized_weights


# =============================================================================
# Function 2: Generate Monthly Returns
# =============================================================================

def generate_monthly_returns(
    n_months: int = 120,
    n_strategies: int = 16
) -> pd.DataFrame:
    """
    Generate synthetic monthly returns for all strategies.

    Each strategy has its own expected return (mean) and risk (volatility).
    Returns are generated using Gaussian (normal) distribution:
        return = mean + volatility * random_noise

    Parameters
    ----------
    n_months : int, optional
        Number of months to simulate. Default is 120 (10 years).
    n_strategies : int, optional
        Number of strategies. Default is 16.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_months, n_strategies).
        - Columns: strategy names from STRATEGY_NAMES
        - Values: decimal returns (e.g., 0.01 means +1%)
        - Index: integers from 0 to n_months-1

    Example
    -------
    >>> returns_df = generate_monthly_returns(n_months=60)
    >>> print(returns_df.shape)
    (60, 16)
    >>> print(returns_df.columns.tolist()[:3])
    ['Cash', 'DepositFixedIncome', 'PureBond']
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Define expected monthly return (mean) for each strategy
    # Lower risk assets have lower expected returns
    # Higher risk assets have higher expected returns
    # -------------------------------------------------------------------------
    means = np.array([
        0.002,   # Cash: ~2.4% annual
        0.003,   # DepositFixedIncome: ~3.6% annual
        0.003,   # PureBond: ~3.6% annual
        0.004,   # NonStandardFixedIncome: ~4.8% annual
        0.004,   # FixedIncomePlus: ~4.8% annual
        0.003,   # OverseasBond: ~3.6% annual
        0.005,   # BalancedFund: ~6% annual
        0.007,   # EquityA: ~8.4% annual
        0.006,   # EquityOverseas: ~7.2% annual
        0.005,   # OverseasBalanced: ~6% annual
        0.004,   # Commodity: ~4.8% annual
        0.005,   # HedgeFund: ~6% annual
        0.005,   # RealEstate: ~6% annual
        0.008,   # PrivateEquity: ~9.6% annual
        0.005,   # OverseasAlternative: ~6% annual
        0.004,   # StructuredProduct: ~4.8% annual
    ])

    # -------------------------------------------------------------------------
    # Define monthly volatility (standard deviation) for each strategy
    # Cash is very stable, equities are volatile
    # -------------------------------------------------------------------------
    vols = np.array([
        0.005,   # Cash: very low volatility
        0.010,   # DepositFixedIncome: low volatility
        0.015,   # PureBond: low-moderate volatility
        0.020,   # NonStandardFixedIncome: moderate volatility
        0.018,   # FixedIncomePlus: moderate volatility
        0.020,   # OverseasBond: moderate (includes FX risk)
        0.025,   # BalancedFund: moderate volatility
        0.045,   # EquityA: high volatility
        0.050,   # EquityOverseas: high volatility
        0.035,   # OverseasBalanced: moderate-high volatility
        0.040,   # Commodity: high volatility
        0.025,   # HedgeFund: moderate volatility
        0.030,   # RealEstate: moderate-high volatility
        0.055,   # PrivateEquity: very high volatility
        0.035,   # OverseasAlternative: moderate-high volatility
        0.025,   # StructuredProduct: moderate volatility
    ])

    # -------------------------------------------------------------------------
    # Generate random returns using the formula:
    #   returns = mean + vol * standard_normal_noise
    #
    # np.random.randn(n_months, n_strategies) generates a matrix of
    # standard normal random numbers (mean=0, std=1)
    # -------------------------------------------------------------------------
    random_noise = np.random.randn(n_months, n_strategies)

    # Broadcasting: means and vols are (16,), random_noise is (T, 16)
    # NumPy automatically broadcasts (16,) to match (T, 16)
    returns_matrix = means + vols * random_noise

    # -------------------------------------------------------------------------
    # Convert NumPy array to pandas DataFrame
    # - columns: strategy names
    # - index: month numbers (0, 1, 2, ...)
    # -------------------------------------------------------------------------
    returns_df = pd.DataFrame(
        data=returns_matrix,
        columns=STRATEGY_NAMES,
        index=range(n_months)
    )

    return returns_df


# =============================================================================
# Function 3: Generate Quadrant Path
# =============================================================================

def generate_quadrant_path(n_months: int = 120) -> pd.Series:
    """
    Generate a path of macro-economic quadrants over time.

    Uses a block structure where each quadrant persists for multiple months.
    This simulates realistic economic regimes that don't change randomly
    each month but instead persist for extended periods.

    The 4 quadrants cycle in order:
        Recovery → Overheat → Stagflation → Recession → Recovery → ...

    Parameters
    ----------
    n_months : int, optional
        Number of months to simulate. Default is 120.

    Returns
    -------
    pd.Series
        A Series of shape (n_months,) containing quadrant labels.
        - Values: one of ["Recovery", "Overheat", "Stagflation", "Recession"]
        - Index: integers from 0 to n_months-1

    Example
    -------
    >>> quadrants = generate_quadrant_path(n_months=120)
    >>> print(quadrants.value_counts())
    Recovery       30
    Overheat       30
    Stagflation    30
    Recession      30
    dtype: int64
    """
    # Calculate how many months each quadrant should last
    # With 4 quadrants and 120 months, each quadrant lasts 30 months
    n_quadrants = len(QUADRANTS)
    months_per_quadrant = n_months // n_quadrants

    # -------------------------------------------------------------------------
    # Use np.repeat() to create blocks of repeated values
    #
    # np.repeat(arr, repeats) repeats each element of arr 'repeats' times
    # Example: np.repeat(['A', 'B'], 3) → ['A', 'A', 'A', 'B', 'B', 'B']
    # -------------------------------------------------------------------------
    quadrant_blocks = np.repeat(QUADRANTS, months_per_quadrant)

    # Handle remainder if n_months is not divisible by 4
    # For example, if n_months=122, we need 2 extra months
    remainder = n_months - len(quadrant_blocks)
    if remainder > 0:
        # Add extra months from the beginning of the cycle
        extra_months = np.array(QUADRANTS[:remainder])
        quadrant_blocks = np.concatenate([quadrant_blocks, extra_months])

    # -------------------------------------------------------------------------
    # Convert to pandas Series
    # Series is like a labeled 1D array - perfect for time series data
    # -------------------------------------------------------------------------
    quadrant_series = pd.Series(
        data=quadrant_blocks,
        index=range(n_months),
        name="Quadrant"
    )

    return quadrant_series


# =============================================================================
# Function 4: Create Mock Dataset (Main Entry Point)
# =============================================================================

def create_mock_dataset(
    n_months: int = 120
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Create a complete mock dataset for TAA backtesting.

    This is the main entry point that packages all synthetic data together.
    Other modules should import and call this function to get test data.

    Parameters
    ----------
    n_months : int, optional
        Number of months to simulate. Default is 120 (10 years).

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, pd.Series]
        A tuple containing:
        - w_saa: np.ndarray of shape (16,) — SAA weights
        - returns_df: pd.DataFrame of shape (n_months, 16) — monthly returns
        - quadrants: pd.Series of shape (n_months,) — macro quadrants

    Example
    -------
    >>> w_saa, returns_df, quadrants = create_mock_dataset(n_months=60)
    >>> print(f"SAA weights: {w_saa.shape}")
    SAA weights: (16,)
    >>> print(f"Returns: {returns_df.shape}")
    Returns: (60, 16)
    >>> print(f"Quadrants: {quadrants.shape}")
    Quadrants: (60,)
    """
    # Generate all three components
    w_saa = generate_saa_weights(n_strategies=16)
    returns_df = generate_monthly_returns(n_months=n_months, n_strategies=16)
    quadrants = generate_quadrant_path(n_months=n_months)

    return w_saa, returns_df, quadrants


# =============================================================================
# Main Demo Block
# =============================================================================

if __name__ == "__main__":
    """
    Demo script to showcase the mock data generation.

    Run this file directly to see example outputs:
        python mock_data.py
    """
    print("=" * 70)
    print("TAA Mock Data Generator - Demo")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Generate the complete dataset
    # -------------------------------------------------------------------------
    w_saa, returns_df, quadrants = create_mock_dataset(n_months=120)

    # -------------------------------------------------------------------------
    # Display SAA Weights
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. Strategic Asset Allocation (SAA) Weights")
    print("-" * 70)

    # Create a Series for better display (shows strategy names alongside weights)
    weights_display = pd.Series(w_saa, index=STRATEGY_NAMES, name="Weight")
    print(weights_display.to_string())
    print(f"\nTotal weight: {w_saa.sum():.6f}")
    print(f"Shape: {w_saa.shape}")

    # -------------------------------------------------------------------------
    # Display Returns DataFrame
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. Monthly Returns DataFrame")
    print("-" * 70)

    print(f"\nShape: {returns_df.shape} (months × strategies)")
    print(f"Columns: {returns_df.columns.tolist()}")

    print("\n--- First 5 months (head) ---")
    print(returns_df.head().to_string())

    print("\n--- Last 5 months (tail) ---")
    print(returns_df.tail().to_string())

    # Show summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Mean monthly return (all assets): {returns_df.values.mean():.4f}")
    print(f"Std monthly return (all assets):  {returns_df.values.std():.4f}")

    # -------------------------------------------------------------------------
    # Display Quadrant Distribution
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. Macro Quadrant Path")
    print("-" * 70)

    print(f"\nShape: {quadrants.shape}")
    print(f"\n--- Quadrant Distribution ---")
    print(quadrants.value_counts().sort_index())

    print(f"\n--- First 10 months ---")
    print(quadrants.head(10).to_string())

    print(f"\n--- Last 10 months ---")
    print(quadrants.tail(10).to_string())

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nYou can import these functions in other modules:")
    print("  from core.mock_data import create_mock_dataset")
    print("  w_saa, returns_df, quadrants = create_mock_dataset()")
    print("=" * 70)

