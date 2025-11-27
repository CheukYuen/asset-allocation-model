"""
run_mock_data.py — Utility script for TAA mock data

This script:
    - Loads synthetic TAA data from CSV files in ../data/
    - Falls back to generating data via core.mock_data if CSVs are missing
    - Prints human-readable summaries for learning NumPy/pandas

Usage:
    python scripts/run_mock_data.py

Dependencies:
    - numpy
    - pandas
    - Python 3.9+ (compatible with 3.11)

Author: TAA Learning Project
"""

# =============================================================================
# Imports
# =============================================================================
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Function 1: Get Paths
# =============================================================================

def get_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Resolve all relevant paths relative to this script.

    Returns
    -------
    Tuple[Path, Path, Path, Path]
        - script_path  : absolute Path of this script
        - project_root : parent of the scripts/ directory
        - data_dir     : project_root / "data"
        - core_dir     : project_root / "core"

    Example
    -------
    >>> script_path, project_root, data_dir, core_dir = get_paths()
    >>> print(data_dir.name)
    data
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_dir = project_root / "data"
    core_dir = project_root / "core"

    return script_path, project_root, data_dir, core_dir


# =============================================================================
# Function 2: Ensure Core on sys.path
# =============================================================================

def ensure_core_on_syspath(core_dir: Path) -> None:
    """
    Add the core directory to sys.path if not already present.

    This allows importing modules from core/ directly, e.g.:
        from mock_data import create_mock_dataset

    Parameters
    ----------
    core_dir : Path
        Absolute path to the core/ directory.

    Returns
    -------
    None
    """
    core_dir_str = str(core_dir)
    if core_dir_str not in sys.path:
        sys.path.insert(0, core_dir_str)


# =============================================================================
# Function 3: Load Dataset from CSV
# =============================================================================

def load_dataset_from_csv(
    data_dir: Path
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Load the three mock data CSVs from the data directory.

    Expected CSV files:
        - mock_saa_weights.csv : index=strategy names, column="weight"
        - mock_returns.csv     : index=dates (DatetimeIndex), columns=strategies
        - mock_quadrants.csv   : index=dates (DatetimeIndex), column="quadrant"

    Parameters
    ----------
    data_dir : Path
        Path to the data/ directory containing the CSV files.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, pd.Series]
        - w_saa      : SAA weights as numpy array, shape (16,)
        - returns_df : Monthly returns DataFrame, shape (T, 16)
        - quadrants  : Macro quadrant Series, shape (T,)

    Raises
    ------
    FileNotFoundError
        If any of the required CSV files is missing.
    """
    weights_path = data_dir / "mock_saa_weights.csv"
    returns_path = data_dir / "mock_returns.csv"
    quadrants_path = data_dir / "mock_quadrants.csv"

    # Check all files exist before loading
    missing_files = []
    for path in [weights_path, returns_path, quadrants_path]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            f"Missing CSV files: {', '.join(missing_files)}"
        )

    # Load SAA weights
    # index_col=0 uses the first column (strategy names) as index
    weights_df = pd.read_csv(weights_path, index_col=0)
    w_saa = weights_df["weight"].values  # Extract as numpy array

    # Load returns with date parsing
    # index_col=0 uses the first column (dates) as index
    # parse_dates=True converts the index to DatetimeIndex
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # Load quadrants with date parsing
    quadrants_df = pd.read_csv(quadrants_path, index_col=0, parse_dates=True)
    quadrants = quadrants_df["quadrant"]  # Extract as Series

    return w_saa, returns_df, quadrants


# =============================================================================
# Function 4: Generate and Save Dataset
# =============================================================================

def generate_and_save_dataset(
    data_dir: Path
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Generate mock data using core.mock_data and save to CSV files.

    This function:
        1. Imports create_mock_dataset from mock_data
        2. Generates 120 months of synthetic data
        3. Converts integer indices to DatetimeIndex for CSVs
        4. Saves to data_dir/ in the specified format

    Parameters
    ----------
    data_dir : Path
        Path to the data/ directory where CSVs will be saved.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, pd.Series]
        - w_saa      : SAA weights as numpy array, shape (16,)
        - returns_df : Monthly returns DataFrame, shape (120, 16)
        - quadrants  : Macro quadrant Series, shape (120,)
    """
    # Import from mock_data (core_dir should already be on sys.path)
    from mock_data import create_mock_dataset, STRATEGY_NAMES

    # Generate the dataset
    n_months = 120
    w_saa, returns_df, quadrants = create_mock_dataset(n_months=n_months)

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Save SAA weights
    # Format: index=strategy names, column="weight"
    # -------------------------------------------------------------------------
    weights_series = pd.Series(w_saa, index=STRATEGY_NAMES, name="weight")
    weights_series.to_csv(data_dir / "mock_saa_weights.csv")

    # -------------------------------------------------------------------------
    # Create DatetimeIndex for returns and quadrants
    # Using month-end dates starting from 2015-01-31
    # freq="ME" is the pandas 2.x way for month-end (replaces deprecated "M")
    # -------------------------------------------------------------------------
    date_index = pd.date_range(
        start="2015-01-31",
        periods=n_months,
        freq="ME"
    )

    # -------------------------------------------------------------------------
    # Save returns with date index
    # -------------------------------------------------------------------------
    returns_df_dated = returns_df.copy()
    returns_df_dated.index = date_index
    returns_df_dated.to_csv(data_dir / "mock_returns.csv")

    # Update the returns_df to have date index (for consistency)
    returns_df.index = date_index

    # -------------------------------------------------------------------------
    # Save quadrants with date index
    # -------------------------------------------------------------------------
    quadrants_dated = quadrants.copy()
    quadrants_dated.index = date_index
    quadrants_dated.name = "quadrant"
    quadrants_dated.to_csv(data_dir / "mock_quadrants.csv")

    # Update quadrants to have date index
    quadrants.index = date_index

    return w_saa, returns_df, quadrants


# =============================================================================
# Function 5: Load or Create Dataset
# =============================================================================

def load_or_create_dataset(
    data_dir: Path
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series, str]:
    """
    Try to load data from CSV; if missing, generate and save.

    Parameters
    ----------
    data_dir : Path
        Path to the data/ directory.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, pd.Series, str]
        - w_saa      : SAA weights as numpy array, shape (16,)
        - returns_df : Monthly returns DataFrame
        - quadrants  : Macro quadrant Series
        - mode       : Either "loaded_from_csv" or "generated_and_saved"
    """
    try:
        w_saa, returns_df, quadrants = load_dataset_from_csv(data_dir)
        mode = "loaded_from_csv"
    except FileNotFoundError:
        w_saa, returns_df, quadrants = generate_and_save_dataset(data_dir)
        mode = "generated_and_saved"

    return w_saa, returns_df, quadrants, mode


# =============================================================================
# Function 6: Print Summary
# =============================================================================

def print_summary(
    w_saa: np.ndarray,
    returns_df: pd.DataFrame,
    quadrants: pd.Series
) -> None:
    """
    Pretty-print a summary of the mock dataset.

    Displays:
        1. SAA weights table (strategy + weight)
        2. First 5 rows of returns (rounded to 4 decimals)
        3. Basic statistics (mean and std per strategy)
        4. Quadrant distribution

    Parameters
    ----------
    w_saa : np.ndarray
        SAA weights, shape (16,).
    returns_df : pd.DataFrame
        Monthly returns, shape (T, 16).
    quadrants : pd.Series
        Macro quadrants, shape (T,).

    Returns
    -------
    None
    """
    # -------------------------------------------------------------------------
    # Section 1: SAA Weights
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== 1. SAA Weights ｜战略配置权重 ===")
    print("=" * 70)

    # Create a formatted table
    strategy_names = returns_df.columns.tolist()
    weights_display = pd.DataFrame({
        "Strategy": strategy_names,
        "Weight": w_saa,
        "Weight (%)": w_saa * 100
    })
    weights_display["Weight (%)"] = weights_display["Weight (%)"].round(2)

    print(weights_display.to_string(index=False))
    print(f"\nTotal weight: {w_saa.sum():.6f}")

    # -------------------------------------------------------------------------
    # Section 2: Sample of Monthly Returns
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== 2. Sample of Monthly Returns ｜月度收益示例（前 5 行） ===")
    print("=" * 70)

    print(f"\nShape: {returns_df.shape} (months × strategies)")
    print("\n--- First 5 rows (rounded to 4 decimals) ---")
    print(returns_df.head().round(4).to_string())

    # -------------------------------------------------------------------------
    # Section 3: Basic Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== 3. Return Statistics ｜收益率统计 ===")
    print("=" * 70)

    # Calculate mean and std for each strategy (monthly)
    means = returns_df.mean()
    stds = returns_df.std()

    stats_df = pd.DataFrame({
        "Strategy": strategy_names,
        "Mean (Monthly)": means.values,
        "Std (Monthly)": stds.values,
        "Mean (Annual)": (means.values * 12),
        "Std (Annual)": (stds.values * np.sqrt(12))
    })

    # Round for display
    for col in ["Mean (Monthly)", "Std (Monthly)", "Mean (Annual)", "Std (Annual)"]:
        stats_df[col] = stats_df[col].round(4)

    print(stats_df.to_string(index=False))

    # Overall portfolio summary
    print(f"\nOverall mean monthly return: {returns_df.values.mean():.4f}")
    print(f"Overall std monthly return:  {returns_df.values.std():.4f}")

    # -------------------------------------------------------------------------
    # Section 4: Quadrant Distribution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== 4. Quadrant Distribution ｜宏观象限分布 ===")
    print("=" * 70)

    print(f"\nTotal periods: {len(quadrants)}")
    print("\n--- Distribution ---")
    print(quadrants.value_counts().sort_index().to_string())

    # Show percentage distribution
    print("\n--- Percentage ---")
    pct_dist = (quadrants.value_counts(normalize=True) * 100).round(1)
    print(pct_dist.sort_index().to_string())


# =============================================================================
# Main Entrypoint
# =============================================================================

if __name__ == "__main__":
    # Step 1: Resolve paths
    script_path, project_root, data_dir, core_dir = get_paths()

    # Step 2: Add core to sys.path for imports
    ensure_core_on_syspath(core_dir)

    # Step 3: Load or create dataset
    w_saa, returns_df, quadrants, mode = load_or_create_dataset(data_dir)

    # Step 4: Print mode and path info
    print("=" * 70)
    print("TAA Mock Data Utility")
    print("=" * 70)
    print(f"Data source mode: {mode}")
    print(f"Data directory  : {data_dir}")

    # Step 5: Print detailed summary
    print_summary(w_saa, returns_df, quadrants)

    print("\n" + "=" * 70)
    print("Script completed successfully!")
    print("=" * 70)

