"""
backtest_engine.py — Performance Analytics Engine for TAA

This module provides backtesting and performance analytics for portfolio strategies.
It computes key metrics including:
    - Portfolio monthly returns (weight × return)
    - Cumulative return curve
    - Annualized return (geometric)
    - Annualized volatility
    - Sharpe ratio
    - Maximum drawdown (MDD)
    - Calmar ratio
    - Win rate

This module works with:
    - core/mock_data.py (data generation)
    - core/taa_signal_engine.py (TAA weight computation)

Usage:
    python core/backtest_engine.py

Dependencies:
    - numpy
    - pandas
    - Python 3.9+ (compatible with 3.11)

Author: TAA Learning Project

backtest_engine.py — TAA 绩效分析引擎

本模块提供组合策略的回测与绩效分析功能。
计算的核心指标包括:
    - 组合月度收益 (权重 × 收益)
    - 累计收益曲线
    - 年化收益率 (几何)
    - 年化波动率
    - 夏普比率
    - 最大回撤 (MDD)
    - 卡玛比率
    - 胜率

配合模块:
    - core/mock_data.py (数据生成)
    - core/taa_signal_engine.py (TAA 权重计算)

运行方式:
    python core/backtest_engine.py

依赖:
    - numpy
    - pandas
    - Python 3.9+ (兼容 3.11)
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from typing import Dict, Any


# =============================================================================
# Function 1: Compute Portfolio Returns
# =============================================================================

def compute_portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio monthly returns as weighted sum of strategy returns.

    Formula:
        r_{p,t} = sum_i( w_{i,t} * r_{i,t} )

    This is the core calculation that combines strategy returns with their
    corresponding weights to get the overall portfolio return for each period.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Monthly returns for each strategy.
        Shape: (T, 16) where T is number of months.
        Index: time periods (DatetimeIndex or integer).
        Columns: strategy names.
        Values: decimal returns (e.g., 0.01 = +1%).

    weights_df : pd.DataFrame
        Portfolio weights for each strategy at each time period.
        Shape: (T, 16), same as returns_df.
        Index: must align with returns_df.index.
        Columns: must align with returns_df.columns.
        Values: weights that sum to 1 per row.

    Returns
    -------
    pd.Series
        Portfolio returns for each time period.
        Index: same as returns_df.index.
        Name: "portfolio_return".

    Example
    -------
    >>> # If weights are [0.6, 0.4] and returns are [0.02, 0.05]
    >>> # Portfolio return = 0.6 * 0.02 + 0.4 * 0.05 = 0.032

    计算组合的月度收益率(各策略收益的加权和)。

    公式:
        r_{p,t} = sum_i( w_{i,t} * r_{i,t} )

    这是核心计算: 将各策略收益率与对应权重相乘后求和,得到每期的组合收益。

    参数:
        returns_df: 各策略月度收益率, 形状 (T, 16), T 为月数
        weights_df: 各策略权重, 形状 (T, 16), 每行权重之和为 1

    返回:
        pd.Series: 每期的组合收益率
    """
    # Element-wise multiplication: (T, 16) * (T, 16) = (T, 16)
    # Each cell is w_{i,t} * r_{i,t}
    weighted_returns = returns_df * weights_df

    # Sum across columns (axis=1) to get portfolio return per period
    # This gives us sum_i( w_{i,t} * r_{i,t} ) for each t
    portfolio_returns = weighted_returns.sum(axis=1)

    # Name the series for clarity
    portfolio_returns.name = "portfolio_return"

    return portfolio_returns


# =============================================================================
# Function 2: Compute Cumulative Return
# =============================================================================

def compute_cumulative_return(
    portfolio_returns: pd.Series
) -> pd.Series:
    """
    Compute the cumulative return curve (wealth index).

    Formula:
        C_0 = 1.0  (initial investment)
        C_t = prod_{k=1..t}(1 + r_{p,k})

    This shows how $1 invested at the start would grow over time.
    A value of 1.5 means the portfolio has gained 50%.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.
        Index: time periods.
        Values: decimal returns.

    Returns
    -------
    pd.Series
        Cumulative return curve (wealth index).
        Index: same as portfolio_returns.index.
        Name: "cumulative_return".
        Starts at (1 + first_return) and compounds from there.

    Example
    -------
    >>> returns = pd.Series([0.01, 0.02, -0.01])
    >>> cumulative = compute_cumulative_return(returns)
    >>> # cumulative[0] = 1.01
    >>> # cumulative[1] = 1.01 * 1.02 = 1.0302
    >>> # cumulative[2] = 1.0302 * 0.99 = 1.019898

    计算累计收益曲线(财富指数)。

    公式:
        C_0 = 1.0 (初始投资)
        C_t = prod_{k=1..t}(1 + r_{p,k})

    展示初始投入 1 元如何随时间增长。值为 1.5 表示组合盈利 50%。

    参数:
        portfolio_returns: 组合月度收益率序列

    返回:
        pd.Series: 累计收益曲线, 从 (1 + 首月收益) 开始复利累积
    """
    # Add 1 to each return to get growth factors
    # e.g., 0.02 return becomes 1.02 growth factor
    growth_factors = 1 + portfolio_returns

    # Cumulative product gives the compounded wealth index
    # cumprod() computes: [a, a*b, a*b*c, ...]
    cumulative_returns = growth_factors.cumprod()

    # Name the series
    cumulative_returns.name = "cumulative_return"

    return cumulative_returns


# =============================================================================
# Function 3: Annualized Return
# =============================================================================

def annualized_return(
    portfolio_returns: pd.Series
) -> float:
    """
    Compute annualized return using geometric compounding.

    Formula:
        R_total = cumulative_return[-1] - 1
        Annualized = (1 + R_total)^(12/T) - 1

    This converts the total return over T months into an equivalent
    annual rate, accounting for compounding.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.

    Returns
    -------
    float
        Annualized return as a decimal (e.g., 0.08 = 8% per year).

    Example
    -------
    >>> # If total return over 24 months is 20%:
    >>> # Annualized = (1.20)^(12/24) - 1 = 1.0954 - 1 = 9.54%

    使用几何复利计算年化收益率。

    公式:
        R_total = cumulative_return[-1] - 1
        年化收益 = (1 + R_total)^(12/T) - 1

    将 T 个月的总收益转换为等效年化收益率(考虑复利)。

    参数:
        portfolio_returns: 组合月度收益率序列

    返回:
        float: 年化收益率(小数形式, 如 0.08 表示 8%)
    """
    # Number of periods (months)
    T = len(portfolio_returns)

    if T == 0:
        return np.nan

    # Compute cumulative return
    cumulative = compute_cumulative_return(portfolio_returns)

    # Total return: final cumulative value - 1
    # If cumulative[-1] = 1.20, then R_total = 0.20 (20% gain)
    R_total = cumulative.iloc[-1] - 1

    # Annualize: (1 + R_total)^(12/T) - 1
    # This raises the growth factor to the power of (months in year / actual months)
    ann_return = (1 + R_total) ** (12.0 / T) - 1

    return float(ann_return)


# =============================================================================
# Function 4: Annualized Volatility
# =============================================================================

def annualized_volatility(
    portfolio_returns: pd.Series
) -> float:
    """
    Compute annualized volatility (standard deviation).

    Formula:
        Annualized Vol = std(r_monthly) * sqrt(12)

    Volatility scales with the square root of time (under certain assumptions).
    Monthly vol is multiplied by sqrt(12) to get annual vol.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.

    Returns
    -------
    float
        Annualized volatility as a decimal (e.g., 0.15 = 15% annual vol).

    Example
    -------
    >>> # If monthly std = 0.03 (3%):
    >>> # Annual vol = 0.03 * sqrt(12) = 0.03 * 3.464 = 10.39%

    计算年化波动率(标准差)。

    公式:
        年化波动率 = std(月度收益) * sqrt(12)

    波动率按时间的平方根缩放。月度波动率乘以 sqrt(12) 得到年化波动率。

    参数:
        portfolio_returns: 组合月度收益率序列

    返回:
        float: 年化波动率(小数形式, 如 0.15 表示 15%)
    """
    # Calculate monthly standard deviation
    # ddof=1 uses sample standard deviation (N-1 denominator)
    monthly_std = portfolio_returns.std(ddof=1)

    # Annualize by multiplying by sqrt(12)
    ann_vol = monthly_std * np.sqrt(12)

    return float(ann_vol)


# =============================================================================
# Function 5: Sharpe Ratio
# =============================================================================

def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Compute the Sharpe ratio (risk-adjusted return).

    Formula:
        Sharpe = (Annualized Return - Risk Free Rate) / Annualized Volatility

    The Sharpe ratio measures excess return per unit of risk.
    Higher is better. A Sharpe > 1 is generally considered good.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.
    risk_free_rate : float, optional
        Annual risk-free rate as decimal. Default is 0.0.

    Returns
    -------
    float
        Sharpe ratio. Returns np.nan if volatility is 0.

    Example
    -------
    >>> # If annualized return = 10%, vol = 15%, rf = 2%:
    >>> # Sharpe = (0.10 - 0.02) / 0.15 = 0.533

    计算夏普比率(风险调整后收益)。

    公式:
        Sharpe = (年化收益 - 无风险利率) / 年化波动率

    夏普比率衡量每单位风险的超额收益。越高越好, 一般认为 Sharpe > 1 较好。

    参数:
        portfolio_returns: 组合月度收益率序列
        risk_free_rate: 年化无风险利率(小数形式), 默认 0.0

    返回:
        float: 夏普比率, 若波动率为 0 则返回 np.nan
    """
    ann_ret = annualized_return(portfolio_returns)
    ann_vol = annualized_volatility(portfolio_returns)

    # Handle edge case: zero volatility
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan

    # Sharpe = excess return / volatility
    sharpe = (ann_ret - risk_free_rate) / ann_vol

    return float(sharpe)


# =============================================================================
# Function 6: Maximum Drawdown
# =============================================================================

def max_drawdown(
    cumulative_returns: pd.Series
) -> float:
    """
    Compute Maximum Drawdown (MDD).

    Formula:
        Running Peak = cummax(C_t)
        Drawdown_t = (C_t / Running Peak) - 1
        MDD = min(Drawdown_t)

    MDD measures the largest peak-to-trough decline.
    It answers: "What's the worst loss from a historical high?"

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative return curve (wealth index).
        Should start at a value > 0 (typically close to 1).

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g., -0.23 = -23% drawdown).

    Example
    -------
    >>> # If wealth goes: 1.0 -> 1.2 -> 0.9 -> 1.1
    >>> # Peak at 1.2, trough at 0.9
    >>> # MDD = (0.9 / 1.2) - 1 = -0.25 = -25%

    计算最大回撤(MDD)。

    公式:
        滚动峰值 = cummax(C_t)
        回撤_t = (C_t / 滚动峰值) - 1
        MDD = min(回撤_t)

    最大回撤衡量从历史高点到谷底的最大跌幅。
    回答: "从历史高点算起, 最糟糕的亏损是多少?"

    参数:
        cumulative_returns: 累计收益曲线(财富指数), 起始值应 > 0

    返回:
        float: 最大回撤(负数, 如 -0.23 表示 -23% 回撤)
    """
    # Calculate running maximum (peak) at each point
    # cummax() returns the cumulative maximum seen so far
    running_peak = cumulative_returns.cummax()

    # Calculate drawdown at each point: how far below the peak
    # If at peak, drawdown = 0
    # If 10% below peak, drawdown = -0.10
    drawdowns = (cumulative_returns / running_peak) - 1

    # Maximum drawdown is the minimum (most negative) drawdown
    mdd = drawdowns.min()

    return float(mdd)


# =============================================================================
# Function 7: Calmar Ratio
# =============================================================================

def calmar_ratio(
    portfolio_returns: pd.Series,
    cumulative_returns: pd.Series
) -> float:
    """
    Compute the Calmar ratio (return vs drawdown risk).

    Formula:
        Calmar = Annualized Return / |MDD|

    The Calmar ratio measures return per unit of drawdown risk.
    Higher is better. It penalizes strategies with large drawdowns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.
    cumulative_returns : pd.Series
        Cumulative return curve.

    Returns
    -------
    float
        Calmar ratio. Returns np.nan if MDD is 0 or positive.

    Example
    -------
    >>> # If annualized return = 12%, MDD = -20%:
    >>> # Calmar = 0.12 / 0.20 = 0.60

    计算卡玛比率(收益与回撤风险之比)。

    公式:
        Calmar = 年化收益 / |MDD|

    卡玛比率衡量每单位回撤风险的收益。越高越好, 惩罚大回撤策略。

    参数:
        portfolio_returns: 组合月度收益率序列
        cumulative_returns: 累计收益曲线

    返回:
        float: 卡玛比率, 若 MDD 为 0 或正数则返回 np.nan
    """
    ann_ret = annualized_return(portfolio_returns)
    mdd = max_drawdown(cumulative_returns)

    # Handle edge cases:
    # - MDD should be negative; if 0 or positive, something is wrong
    if mdd >= 0:
        return np.nan

    # Calmar = annualized return / |MDD|
    # We use abs(mdd) because MDD is negative
    calmar = ann_ret / abs(mdd)

    return float(calmar)


# =============================================================================
# Function 8: Win Rate
# =============================================================================

def win_rate(
    portfolio_returns: pd.Series
) -> float:
    """
    Compute win rate (percentage of positive return months).

    Formula:
        Win Rate = (# of months with r > 0) / T

    This simple metric shows how often the portfolio makes money.
    A win rate > 50% is generally good, but depends on the magnitude
    of wins vs losses.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly portfolio returns.

    Returns
    -------
    float
        Win rate as a decimal (e.g., 0.60 = 60% winning months).

    Example
    -------
    >>> # If 7 out of 12 months are positive:
    >>> # Win rate = 7/12 = 0.583 = 58.3%

    计算胜率(正收益月份占比)。

    公式:
        胜率 = (收益 > 0 的月数) / T

    展示组合盈利的频率。胜率 > 50% 通常较好, 但也取决于盈亏的幅度。

    参数:
        portfolio_returns: 组合月度收益率序列

    返回:
        float: 胜率(小数形式, 如 0.60 表示 60% 的月份盈利)
    """
    T = len(portfolio_returns)

    if T == 0:
        return np.nan

    # Count months with positive returns
    # (portfolio_returns > 0) creates a boolean Series
    # .sum() counts True values (True = 1, False = 0)
    winning_months = (portfolio_returns > 0).sum()

    # Win rate = winning months / total months
    win_rate_value = winning_months / T

    return float(win_rate_value)


# =============================================================================
# Function 9: Aggregate Backtest Metrics
# =============================================================================

def aggregate_backtest_metrics(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    High-level wrapper that computes all backtest metrics.

    This convenience function runs the complete backtest pipeline:
        1. Compute portfolio monthly returns
        2. Compute cumulative return curve
        3. Compute all performance metrics

    Parameters
    ----------
    returns_df : pd.DataFrame
        Monthly returns for each strategy. Shape (T, 16).
    weights_df : pd.DataFrame
        Portfolio weights. Shape (T, 16).

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

    Example
    -------
    >>> metrics = aggregate_backtest_metrics(returns_df, weights_df)
    >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    >>> print(f"MDD: {metrics['max_drawdown']:.2%}")

    高层封装函数, 计算所有回测指标。

    执行完整的回测流程:
        1. 计算组合月度收益
        2. 计算累计收益曲线
        3. 计算所有绩效指标

    参数:
        returns_df: 各策略月度收益率, 形状 (T, 16)
        weights_df: 组合权重, 形状 (T, 16)

    返回:
        Dict[str, Any]: 包含以下内容的字典:
        - "portfolio_returns": 月度收益序列
        - "cumulative_returns": 累计收益序列
        - "annualized_return": 年化收益率
        - "annualized_volatility": 年化波动率
        - "sharpe_ratio": 夏普比率
        - "max_drawdown": 最大回撤(负数)
        - "calmar_ratio": 卡玛比率
        - "win_rate": 胜率
    """
    # Step 1: Compute portfolio returns
    portfolio_returns = compute_portfolio_returns(returns_df, weights_df)

    # Step 2: Compute cumulative returns
    cumulative_returns = compute_cumulative_return(portfolio_returns)

    # Step 3: Compute all metrics
    metrics = {
        "portfolio_returns": portfolio_returns,
        "cumulative_returns": cumulative_returns,
        "annualized_return": annualized_return(portfolio_returns),
        "annualized_volatility": annualized_volatility(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(cumulative_returns),
        "calmar_ratio": calmar_ratio(portfolio_returns, cumulative_returns),
        "win_rate": win_rate(portfolio_returns),
    }

    return metrics


# =============================================================================
# Main Demo Block
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of the backtest engine.

    This demo:
        1. Loads mock data from mock_data.py
        2. Computes TAA-adjusted weights from taa_signal_engine.py
        3. Runs the full backtest
        4. Prints all metrics in human-readable format

    Run with:
        python core/backtest_engine.py
    """
    import sys
    from pathlib import Path

    # -------------------------------------------------------------------------
    # Step 1: Setup sys.path for imports
    # -------------------------------------------------------------------------
    # Get the directory containing this script (core/)
    script_path = Path(__file__).resolve()
    core_dir = script_path.parent
    project_root = core_dir.parent

    # Add core directory to sys.path if not already present
    core_dir_str = str(core_dir)
    if core_dir_str not in sys.path:
        sys.path.insert(0, core_dir_str)

    # -------------------------------------------------------------------------
    # Step 2: Import other core modules
    # -------------------------------------------------------------------------
    from mock_data import create_mock_dataset
    from taa_signal_engine import (
        get_strategy_metadata,
        DELTA_ASSET,
        compute_final_weights_over_time,
    )

    # -------------------------------------------------------------------------
    # Step 3: Generate mock data
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("TAA Backtest Engine Demo")
    print("=" * 70)

    n_months = 120
    w_saa, returns_df, quadrants = create_mock_dataset(n_months=n_months)

    print(f"\nGenerated {n_months} months of mock data.")
    print(f"  - SAA weights shape: {w_saa.shape}")
    print(f"  - Returns shape: {returns_df.shape}")
    print(f"  - Quadrants shape: {quadrants.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Convert returns_df index to DatetimeIndex for consistency
    # -------------------------------------------------------------------------
    # mock_data returns integer index; TAA engine expects any index
    # For demonstration, we'll create a DatetimeIndex
    date_index = pd.date_range(start="2015-01-01", periods=n_months, freq="MS")
    returns_df.index = date_index
    quadrants.index = date_index

    # -------------------------------------------------------------------------
    # Step 5: Compute TAA-adjusted weights
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Computing TAA-adjusted weights...")
    print("-" * 70)

    metadata = get_strategy_metadata()
    weights_df = compute_final_weights_over_time(
        w_saa=w_saa,
        quadrants=quadrants,
        metadata=metadata,
        delta_asset=DELTA_ASSET,
    )

    print(f"  - Weights shape: {weights_df.shape}")
    print(f"  - Row sums (should be 1.0): min={weights_df.sum(axis=1).min():.6f}, "
          f"max={weights_df.sum(axis=1).max():.6f}")

    # -------------------------------------------------------------------------
    # Step 6: Run backtest
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Running backtest...")
    print("-" * 70)

    metrics = aggregate_backtest_metrics(returns_df, weights_df)

    # -------------------------------------------------------------------------
    # Step 7: Print all metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== Backtest Results ===")
    print("=" * 70)

    print("\n--- Performance Metrics ---")
    print(f"  Annualized Return:     {metrics['annualized_return']:>10.2%}")
    print(f"  Annualized Volatility: {metrics['annualized_volatility']:>10.2%}")
    print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Maximum Drawdown:      {metrics['max_drawdown']:>10.2%}")
    print(f"  Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}")
    print(f"  Win Rate:              {metrics['win_rate']:>10.2%}")

    # -------------------------------------------------------------------------
    # Step 8: Print sample of portfolio returns
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Portfolio Returns (first 10 months):")
    print("-" * 70)

    portfolio_returns = metrics["portfolio_returns"]
    for i, (date, ret) in enumerate(portfolio_returns.head(10).items()):
        print(f"  {date.strftime('%Y-%m')}: {ret:>8.4f} ({ret*100:>6.2f}%)")

    # -------------------------------------------------------------------------
    # Step 9: Print sample of cumulative returns
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Cumulative Returns (first 10 months):")
    print("-" * 70)

    cumulative_returns = metrics["cumulative_returns"]
    for i, (date, cum) in enumerate(cumulative_returns.head(10).items()):
        gain = (cum - 1) * 100
        print(f"  {date.strftime('%Y-%m')}: {cum:>8.4f} ({gain:>+6.2f}%)")

    # -------------------------------------------------------------------------
    # Step 10: Print final summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Final Portfolio Value:")
    print("-" * 70)

    final_value = cumulative_returns.iloc[-1]
    total_return = (final_value - 1) * 100
    print(f"  Starting value: $1.00")
    print(f"  Final value:    ${final_value:.4f}")
    print(f"  Total return:   {total_return:+.2f}%")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nYou can import these functions in other modules:")
    print("  from backtest_engine import aggregate_backtest_metrics")
    print("  metrics = aggregate_backtest_metrics(returns_df, weights_df)")
    print("=" * 70)

