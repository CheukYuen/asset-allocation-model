"""
run_backtest.py — TAA Backtest Runner Script

This script:
    1. Loads pre-generated mock data from ../data/ (CSV files)
    2. Computes TAA-adjusted final weights using core/taa_signal_engine.py
    3. Runs backtest metrics using core/backtest_engine.py
    4. Prints a human-readable backtest report (bilingual)

Usage:
    python scripts/run_backtest.py

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
from typing import Tuple, Dict, Any

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
        from taa_signal_engine import get_strategy_metadata

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
# Function 4: Run Backtest
# =============================================================================

def run_backtest(
    w_saa: np.ndarray,
    returns_df: pd.DataFrame,
    quadrants: pd.Series
) -> Dict[str, Any]:
    """
    Compute TAA-adjusted weights and run the backtest.

    This function:
        1. Gets strategy metadata via get_strategy_metadata()
        2. Computes final weights with compute_final_weights_over_time(...)
        3. Calls aggregate_backtest_metrics(...) to get all performance metrics

    Parameters
    ----------
    w_saa : np.ndarray
        SAA baseline weights, shape (16,).
    returns_df : pd.DataFrame
        Monthly returns for each strategy, shape (T, 16).
    quadrants : pd.Series
        Macro quadrant for each month, shape (T,).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "portfolio_returns": pd.Series of monthly returns
        - "cumulative_returns": pd.Series of cumulative returns
        - "annualized_return": float
        - "annualized_volatility": float
        - "sharpe_ratio": float
        - "max_drawdown": float (negative)
        - "calmar_ratio": float
        - "win_rate": float
    """
    # Import from core modules (core_dir should already be on sys.path)
    from taa_signal_engine import (
        get_strategy_metadata,
        DELTA_ASSET,
        compute_final_weights_over_time,
    )
    from backtest_engine import aggregate_backtest_metrics

    # Step 1: Get strategy metadata
    metadata = get_strategy_metadata()

    # Step 2: Compute TAA-adjusted weights over time
    weights_df = compute_final_weights_over_time(
        w_saa=w_saa,
        quadrants=quadrants,
        metadata=metadata,
        delta_asset=DELTA_ASSET,
    )

    # Step 3: Run backtest metrics
    results = aggregate_backtest_metrics(
        returns_df=returns_df,
        weights_df=weights_df,
    )

    return results


# =============================================================================
# Function 5: Print Backtest Report
# =============================================================================

def print_backtest_report(results: Dict[str, Any]) -> None:
    """
    Print a human-readable backtest summary report.

    Displays:
        - Data period (start date, end date, number of months)
        - Annualized return
        - Annualized volatility
        - Sharpe ratio
        - Maximum drawdown (MDD)
        - Calmar ratio
        - Win rate
        - Sample of portfolio returns (first 5 rows)
        - Sample of cumulative curve (first 5 rows)

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary from aggregate_backtest_metrics containing all metrics.

    Returns
    -------
    None
    """
    # Extract metrics from results
    portfolio_returns = results["portfolio_returns"]
    cumulative_returns = results["cumulative_returns"]
    ann_return = results["annualized_return"]
    ann_vol = results["annualized_volatility"]
    sharpe = results["sharpe_ratio"]
    mdd = results["max_drawdown"]
    calmar = results["calmar_ratio"]
    win = results["win_rate"]

    # Infer data period
    start_date = portfolio_returns.index[0]
    end_date = portfolio_returns.index[-1]
    T = len(portfolio_returns)

    # Format dates for display
    if hasattr(start_date, 'strftime'):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        start_str = str(start_date)
        end_str = str(end_date)

    # Print the report
    print("")
    print("=" * 60)
    print("=== Backtest Summary ｜回测摘要 ===")
    print("=" * 60)
    print(f"Data period          : {start_str} → {end_str} (T = {T} months)")
    print("")
    print(f"Annualized Return    : {ann_return:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio         : {sharpe:.2f}")
    print(f"Max Drawdown (MDD)   : {mdd:.2%}")
    print(f"Calmar Ratio         : {calmar:.2f}")
    print(f"Win Rate             : {win:.2%}")
    print("")
    print("=" * 60)
    print("=== Sample of Portfolio Return ｜组合月度收益示例（前 5 行） ===")
    print("=" * 60)
    print(portfolio_returns.head().round(4).to_string())
    print("")
    print("=" * 60)
    print("=== Sample of Cumulative Curve ｜累计净值示例（前 5 行） ===")
    print("=" * 60)
    print(cumulative_returns.head().round(4).to_string())
    print("")


# =============================================================================
# Main Entrypoint
# =============================================================================

if __name__ == "__main__":
    # Step 1: Resolve paths
    script_path, project_root, data_dir, core_dir = get_paths()

    # Step 2: Add core to sys.path for imports
    ensure_core_on_syspath(core_dir)

    # Step 3: Try to load data from CSV files
    try:
        w_saa, returns_df, quadrants = load_dataset_from_csv(data_dir)
    except FileNotFoundError:
        print("Data files not found in ../data/.")
        print("Please run:  python scripts/run_mock_data.py  first.")
        raise SystemExit(1)

    # Step 4: Run backtest
    results = run_backtest(w_saa, returns_df, quadrants)

    # Step 5: Print report
    print_backtest_report(results)

