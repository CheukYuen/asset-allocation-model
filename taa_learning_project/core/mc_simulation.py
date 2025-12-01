"""
mc_simulation.py — Monte Carlo Simulation Engine for Portfolio Analysis

This module provides Monte Carlo simulation tools for portfolio wealth projection.
It simulates future wealth paths based on historical portfolio returns and
summarizes the distribution of terminal wealth.

Key features:
    - Two simulation methods: "normal" (parametric) and "bootstrap" (non-parametric)
    - Terminal wealth distribution analysis with quantiles
    - Time-series quantile paths for visualization
    - Integration with TAA backtest pipeline

本模块提供组合财富预测的蒙特卡罗模拟工具。
基于历史组合收益模拟未来财富路径，并汇总终端财富分布。

核心功能：
    - 两种模拟方法：正态分布法（参数化）和自助法（非参数化）
    - 终端财富分布分析（含分位数）
    - 时间序列分位数路径（用于可视化）
    - 与 TAA 回测流程集成

Usage:
    python core/mc_simulation.py

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
from typing import Dict, Any, Sequence, List


# =============================================================================
# Function 1: Simulate Wealth Paths
# =============================================================================

def simulate_paths_from_portfolio_returns(
    portfolio_returns: pd.Series,
    n_years: int = 30,
    n_paths: int = 1000,
    method: str = "bootstrap",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate future wealth paths using historical portfolio monthly returns.

    This function generates Monte Carlo simulations of portfolio wealth evolution
    over a specified time horizon. Two methods are supported:
        - "normal": Assumes returns follow a normal distribution
        - "bootstrap": Resamples historical returns with replacement

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical monthly returns of the portfolio (decimal format).
        Example: 0.01 means +1% return for that month.
        历史组合月度收益率（小数格式）。
        例如：0.01 表示该月 +1% 的收益。

    n_years : int, default 30
        Number of future years to simulate.
        模拟的未来年数。

    n_paths : int, default 1000
        Number of Monte Carlo paths (scenarios) to generate.
        生成的蒙特卡罗路径（情景）数量。

    method : {"normal", "bootstrap"}, default "bootstrap"
        Simulation method:
        - "normal": Sample monthly returns from Normal(mu, sigma)
                    where mu = mean(historical), sigma = std(historical).
        - "bootstrap": Sample monthly returns with replacement from
                       the historical Series.
        模拟方法：
        - "normal"：从正态分布 N(mu, sigma) 中抽样月度收益
        - "bootstrap"：从历史序列中有放回抽样

    random_state : int, default 42
        Random seed for reproducibility.
        随机种子，用于可重复性。

    Returns
    -------
    wealth_paths : pd.DataFrame
        Shape: (T + 1, n_paths), where T = n_years * 12.
        - Index: RangeIndex from 0 to T (0 is initial point, wealth = 1.0)
        - Columns: "path_0", "path_1", ..., "path_{n_paths-1}"
        - Values: Wealth levels over time, starting from 1.0 at t=0.

        形状：(T + 1, n_paths)，其中 T = n_years * 12。
        - 索引：从 0 到 T 的整数索引（0 为起始点，财富 = 1.0）
        - 列名："path_0", "path_1", ..., "path_{n_paths-1}"
        - 值：随时间变化的财富水平，从 t=0 时的 1.0 开始

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pd.Series(np.random.randn(60) * 0.03 + 0.005)
    >>> wealth = simulate_paths_from_portfolio_returns(returns, n_years=10, n_paths=100)
    >>> print(wealth.shape)
    (121, 100)
    >>> print(wealth.iloc[0, :3])  # All paths start at 1.0
    path_0    1.0
    path_1    1.0
    path_2    1.0
    Name: 0, dtype: float64

    Notes
    -----
    Wealth evolution formula:
        W_{t+1} = W_t * (1 + r_{t,k})
    where r_{t,k} is the simulated monthly return for path k at time t.

    财富演化公式：
        W_{t+1} = W_t * (1 + r_{t,k})
    其中 r_{t,k} 是路径 k 在时刻 t 的模拟月度收益。
    """
    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    if method not in ("normal", "bootstrap"):
        raise ValueError(
            f"method must be 'normal' or 'bootstrap', got '{method}'"
        )

    if len(portfolio_returns) == 0:
        raise ValueError("portfolio_returns cannot be empty")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    # Initialize random number generator with seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Total number of future months
    T = n_years * 12

    # Convert historical returns to numpy array for efficient operations
    hist_returns = portfolio_returns.values.astype(np.float64)

    # -------------------------------------------------------------------------
    # Generate simulated returns matrix: shape (T, n_paths)
    # -------------------------------------------------------------------------
    if method == "normal":
        # Parametric approach: assume returns ~ Normal(mu, sigma)
        # 参数化方法：假设收益服从正态分布
        mu = float(np.mean(hist_returns))
        sigma = float(np.std(hist_returns, ddof=1))

        # Generate (T x n_paths) matrix of normally distributed returns
        # rng.normal(loc, scale, size) generates random samples
        simulated_returns = rng.normal(loc=mu, scale=sigma, size=(T, n_paths))

    else:  # method == "bootstrap"
        # Non-parametric approach: resample historical returns with replacement
        # 非参数化方法：从历史收益中有放回抽样
        n_hist = len(hist_returns)

        # Generate random indices to sample from historical returns
        # Shape: (T, n_paths) - each element is an index into hist_returns
        random_indices = rng.integers(low=0, high=n_hist, size=(T, n_paths))

        # Use fancy indexing to get the resampled returns
        simulated_returns = hist_returns[random_indices]

    # -------------------------------------------------------------------------
    # Convert returns to wealth paths via cumulative product
    # -------------------------------------------------------------------------
    # Step 1: Convert returns to growth factors (1 + r)
    growth_factors = 1.0 + simulated_returns  # Shape: (T, n_paths)

    # Step 2: Compute cumulative product along time axis (axis=0)
    # This gives wealth at end of each period, assuming W_0 = 1
    cumulative_wealth = np.cumprod(growth_factors, axis=0)  # Shape: (T, n_paths)

    # Step 3: Prepend initial wealth W_0 = 1.0
    # Create a row of ones with shape (1, n_paths)
    initial_wealth = np.ones((1, n_paths), dtype=np.float64)

    # Stack: initial wealth row on top of cumulative wealth
    # Result shape: (T + 1, n_paths)
    wealth_array = np.vstack([initial_wealth, cumulative_wealth])

    # -------------------------------------------------------------------------
    # Create DataFrame with proper column names and index
    # -------------------------------------------------------------------------
    column_names: List[str] = [f"path_{i}" for i in range(n_paths)]

    wealth_paths = pd.DataFrame(
        data=wealth_array,
        index=range(T + 1),  # 0 to T inclusive
        columns=column_names,
    )

    return wealth_paths


# =============================================================================
# Function 2: Summarize Terminal Wealth
# =============================================================================

def summarize_terminal_wealth(
    wealth_paths: pd.DataFrame,
    confidence_levels: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    goal_wealth: float = 1.0,
) -> Dict[str, Any]:
    """
    Summarize the terminal wealth distribution at the final time step.

    This function analyzes the distribution of wealth at the end of all
    simulation paths, providing key statistics and success probabilities.

    Parameters
    ----------
    wealth_paths : pd.DataFrame
        Output from simulate_paths_from_portfolio_returns.
        Shape: (T + 1, n_paths)
        来自 simulate_paths_from_portfolio_returns 的输出。
        形状：(T + 1, n_paths)

    confidence_levels : sequence of float, default (0.05, 0.25, 0.50, 0.75, 0.95)
        Quantiles to compute on the terminal wealth distribution.
        Values should be in (0, 1).
        要计算的终端财富分布分位数。
        值应在 (0, 1) 范围内。

    goal_wealth : float, default 1.0
        A target wealth level to evaluate success probability.
        For example, goal_wealth=2.0 means "double the initial investment".
        目标财富水平，用于评估成功概率。
        例如，goal_wealth=2.0 表示"使初始投资翻倍"。

    Returns
    -------
    summary : Dict[str, Any]
        Dictionary containing:
        - "terminal_wealth": np.ndarray with shape (n_paths,) - terminal wealth values
        - "quantiles": dict mapping q -> wealth_q (e.g., 0.05 -> 1.23)
        - "mean": float - mean terminal wealth
        - "std": float - standard deviation of terminal wealth
        - "min": float - minimum terminal wealth
        - "max": float - maximum terminal wealth
        - "prob_above_goal": float - fraction of paths with W_T >= goal_wealth
        - "prob_below_1": float - fraction of paths with W_T < 1.0 (loss)

        包含以下内容的字典：
        - "terminal_wealth": 形状为 (n_paths,) 的 np.ndarray - 终端财富值
        - "quantiles": 字典，映射 q -> wealth_q（如 0.05 -> 1.23）
        - "mean": float - 终端财富均值
        - "std": float - 终端财富标准差
        - "min": float - 终端财富最小值
        - "max": float - 终端财富最大值
        - "prob_above_goal": float - W_T >= goal_wealth 的路径比例
        - "prob_below_1": float - W_T < 1.0（亏损）的路径比例

    Examples
    --------
    >>> summary = summarize_terminal_wealth(wealth_paths, goal_wealth=2.0)
    >>> print(f"Median terminal wealth: {summary['quantiles'][0.50]:.2f}")
    >>> print(f"Probability of doubling: {summary['prob_above_goal']:.1%}")

    Notes
    -----
    - prob_above_goal answers: "What's the probability of reaching my goal?"
    - prob_below_1 answers: "What's the probability of losing money?"

    - prob_above_goal 回答："达到目标的概率是多少？"
    - prob_below_1 回答："亏损的概率是多少？"
    """
    # -------------------------------------------------------------------------
    # Extract terminal wealth (last row)
    # -------------------------------------------------------------------------
    terminal_wealth_series = wealth_paths.iloc[-1]
    terminal_wealth = terminal_wealth_series.values.astype(np.float64)
    n_paths = len(terminal_wealth)

    # -------------------------------------------------------------------------
    # Compute quantiles
    # -------------------------------------------------------------------------
    quantiles_dict: Dict[float, float] = {}
    for q in confidence_levels:
        quantiles_dict[q] = float(np.quantile(terminal_wealth, q))

    # -------------------------------------------------------------------------
    # Compute basic statistics
    # -------------------------------------------------------------------------
    mean_wealth = float(np.mean(terminal_wealth))
    std_wealth = float(np.std(terminal_wealth, ddof=1))
    min_wealth = float(np.min(terminal_wealth))
    max_wealth = float(np.max(terminal_wealth))

    # -------------------------------------------------------------------------
    # Compute success/failure probabilities
    # -------------------------------------------------------------------------
    # Probability of reaching or exceeding goal wealth
    # 达到或超过目标财富的概率
    count_above_goal = np.sum(terminal_wealth >= goal_wealth)
    prob_above_goal = float(count_above_goal) / n_paths

    # Probability of losing money (ending below initial investment)
    # 亏损概率（终端财富低于初始投资）
    count_below_1 = np.sum(terminal_wealth < 1.0)
    prob_below_1 = float(count_below_1) / n_paths

    # -------------------------------------------------------------------------
    # Build and return summary dictionary
    # -------------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "terminal_wealth": terminal_wealth,
        "quantiles": quantiles_dict,
        "mean": mean_wealth,
        "std": std_wealth,
        "min": min_wealth,
        "max": max_wealth,
        "prob_above_goal": prob_above_goal,
        "prob_below_1": prob_below_1,
    }

    return summary


# =============================================================================
# Function 3: Summarize Quantile Paths
# =============================================================================

def summarize_quantile_paths(
    wealth_paths: pd.DataFrame,
    confidence_levels: Sequence[float] = (0.05, 0.50, 0.95),
) -> pd.DataFrame:
    """
    Compute quantile paths over time.

    For each time step t and each quantile q in confidence_levels,
    compute the q-quantile of wealth over all simulation paths.
    This is useful for visualizing the "fan" of possible outcomes.

    Parameters
    ----------
    wealth_paths : pd.DataFrame
        Output from simulate_paths_from_portfolio_returns.
        Shape: (T + 1, n_paths)
        来自 simulate_paths_from_portfolio_returns 的输出。
        形状：(T + 1, n_paths)

    confidence_levels : sequence of float, default (0.05, 0.50, 0.95)
        Quantiles to compute at each time step.
        Values should be in (0, 1).
        在每个时间步要计算的分位数。
        值应在 (0, 1) 范围内。

    Returns
    -------
    quantile_paths : pd.DataFrame
        Shape: (T + 1, len(confidence_levels))
        - Index: same as wealth_paths.index (0 to T)
        - Columns: e.g. "q_5", "q_50", "q_95" for q=0.05, 0.50, 0.95
        - Values: quantile of wealth at each time step

        形状：(T + 1, len(confidence_levels))
        - 索引：与 wealth_paths.index 相同（0 到 T）
        - 列名：如 "q_5", "q_50", "q_95" 对应 q=0.05, 0.50, 0.95
        - 值：每个时间步的财富分位数

    Examples
    --------
    >>> quantile_df = summarize_quantile_paths(wealth_paths)
    >>> print(quantile_df.columns.tolist())
    ['q_5', 'q_50', 'q_95']
    >>> print(quantile_df.iloc[-1])  # Terminal quantiles
    q_5     1.23
    q_50    3.45
    q_95    8.90
    Name: 360, dtype: float64

    Notes
    -----
    Column naming convention: "q_X" where X = int(q * 100).
    For example, q=0.05 becomes "q_5", q=0.50 becomes "q_50".

    列名命名规则："q_X"，其中 X = int(q * 100)。
    例如，q=0.05 变为 "q_5"，q=0.50 变为 "q_50"。
    """
    # -------------------------------------------------------------------------
    # Convert wealth_paths to numpy array for efficient computation
    # -------------------------------------------------------------------------
    wealth_array = wealth_paths.values.astype(np.float64)
    # Shape: (T + 1, n_paths)

    # -------------------------------------------------------------------------
    # Compute quantiles at each time step
    # -------------------------------------------------------------------------
    # np.quantile with axis=1 computes quantiles across columns (paths)
    # Result shape: (len(confidence_levels), T + 1)
    quantile_values = np.quantile(wealth_array, confidence_levels, axis=1)
    # Transpose to get shape (T + 1, len(confidence_levels))
    quantile_values = quantile_values.T

    # -------------------------------------------------------------------------
    # Create column names: "q_5", "q_25", "q_50", etc.
    # -------------------------------------------------------------------------
    column_names: List[str] = [
        f"q_{int(q * 100)}" for q in confidence_levels
    ]

    # -------------------------------------------------------------------------
    # Build DataFrame
    # -------------------------------------------------------------------------
    quantile_paths = pd.DataFrame(
        data=quantile_values,
        index=wealth_paths.index,
        columns=column_names,
    )

    return quantile_paths


# =============================================================================
# Function 4: Aggregate MC Results
# =============================================================================

def aggregate_mc_results(
    portfolio_returns: pd.Series,
    n_years: int = 30,
    n_paths: int = 1000,
    method: str = "bootstrap",
    goal_wealth: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    High-level wrapper for Monte Carlo simulation and analysis.

    This convenience function runs the complete MC pipeline:
        1) Simulate wealth paths from portfolio_returns
        2) Summarize terminal wealth distribution
        3) Summarize quantile paths over time

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical monthly returns of the portfolio (decimal format).
        历史组合月度收益率（小数格式）。

    n_years : int, default 30
        Number of future years to simulate.
        模拟的未来年数。

    n_paths : int, default 1000
        Number of Monte Carlo paths to generate.
        生成的蒙特卡罗路径数量。

    method : {"normal", "bootstrap"}, default "bootstrap"
        Simulation method.
        模拟方法。

    goal_wealth : float, default 1.0
        Target wealth level for success probability calculation.
        目标财富水平，用于成功概率计算。

    random_state : int, default 42
        Random seed for reproducibility.
        随机种子，用于可重复性。

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing:
        - "wealth_paths": pd.DataFrame from simulate_paths_from_portfolio_returns
        - "terminal_summary": dict from summarize_terminal_wealth
        - "quantile_paths": pd.DataFrame from summarize_quantile_paths

        包含以下内容的字典：
        - "wealth_paths": 来自 simulate_paths_from_portfolio_returns 的 pd.DataFrame
        - "terminal_summary": 来自 summarize_terminal_wealth 的字典
        - "quantile_paths": 来自 summarize_quantile_paths 的 pd.DataFrame

    Examples
    --------
    >>> results = aggregate_mc_results(portfolio_returns, n_years=30, goal_wealth=3.0)
    >>> print(f"Median terminal wealth: {results['terminal_summary']['quantiles'][0.50]:.2f}")
    >>> print(f"P(wealth >= 3x): {results['terminal_summary']['prob_above_goal']:.1%}")

    Notes
    -----
    This function uses a shorter set of quantiles (5%, 50%, 95%) for the
    quantile_paths output, which is typically sufficient for visualization.

    此函数使用较短的分位数集合（5%、50%、95%）生成 quantile_paths 输出，
    这通常足以满足可视化需求。
    """
    # -------------------------------------------------------------------------
    # Step 1: Simulate wealth paths
    # -------------------------------------------------------------------------
    wealth_paths = simulate_paths_from_portfolio_returns(
        portfolio_returns=portfolio_returns,
        n_years=n_years,
        n_paths=n_paths,
        method=method,
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # Step 2: Summarize terminal wealth
    # -------------------------------------------------------------------------
    terminal_summary = summarize_terminal_wealth(
        wealth_paths=wealth_paths,
        confidence_levels=(0.05, 0.25, 0.50, 0.75, 0.95),
        goal_wealth=goal_wealth,
    )

    # -------------------------------------------------------------------------
    # Step 3: Summarize quantile paths
    # -------------------------------------------------------------------------
    quantile_paths = summarize_quantile_paths(
        wealth_paths=wealth_paths,
        confidence_levels=(0.05, 0.50, 0.95),
    )

    # -------------------------------------------------------------------------
    # Build and return results dictionary
    # -------------------------------------------------------------------------
    results: Dict[str, Any] = {
        "wealth_paths": wealth_paths,
        "terminal_summary": terminal_summary,
        "quantile_paths": quantile_paths,
    }

    return results


# =============================================================================
# Main Demo Block
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of Monte Carlo simulation engine.

    This demo integrates with other core modules to show the full pipeline:
        1. Generate mock data (mock_data.py)
        2. Compute TAA-adjusted weights (taa_signal_engine.py)
        3. Compute portfolio returns (backtest_engine.py)
        4. Run Monte Carlo simulation
        5. Print summary statistics

    Run with:
        python core/mc_simulation.py
    """
    import sys
    from pathlib import Path

    # -------------------------------------------------------------------------
    # Step 1: Setup sys.path for imports
    # -------------------------------------------------------------------------
    THIS_FILE = Path(__file__).resolve()
    CORE_DIR = THIS_FILE.parent
    PROJECT_ROOT = CORE_DIR.parent

    # Add core directory to sys.path if not already present
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))

    # -------------------------------------------------------------------------
    # Step 2: Import other core modules
    # -------------------------------------------------------------------------
    from mock_data import create_mock_dataset
    from taa_signal_engine import (
        get_strategy_metadata,
        DELTA_ASSET,
        compute_final_weights_over_time,
    )
    from backtest_engine import compute_portfolio_returns

    # -------------------------------------------------------------------------
    # Step 3: Generate synthetic dataset
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Monte Carlo Simulation Engine Demo")
    print("蒙特卡罗模拟引擎演示")
    print("=" * 70)

    n_months = 120
    w_saa, returns_df, quadrants = create_mock_dataset(n_months=n_months)

    # Convert to DatetimeIndex for consistency
    date_index = pd.date_range(start="2015-01-01", periods=n_months, freq="MS")
    returns_df.index = date_index
    quadrants.index = date_index

    print(f"\nGenerated {n_months} months of mock data.")
    print(f"生成了 {n_months} 个月的模拟数据。")

    # -------------------------------------------------------------------------
    # Step 4: Compute TAA-adjusted weights
    # -------------------------------------------------------------------------
    metadata = get_strategy_metadata()
    weights_df = compute_final_weights_over_time(
        w_saa=w_saa,
        quadrants=quadrants,
        metadata=metadata,
        delta_asset=DELTA_ASSET,
    )

    # -------------------------------------------------------------------------
    # Step 5: Compute portfolio monthly returns
    # -------------------------------------------------------------------------
    portfolio_returns = compute_portfolio_returns(
        returns_df=returns_df,
        weights_df=weights_df,
    )

    print("\n" + "-" * 70)
    print("Historical Portfolio Returns Summary | 历史组合收益摘要")
    print("-" * 70)
    print(f"  Length of historical series: {len(portfolio_returns)} months")
    print(f"  历史序列长度: {len(portfolio_returns)} 个月")
    print(f"  Historical mean (monthly):   {portfolio_returns.mean():.4f} ({portfolio_returns.mean()*100:.2f}%)")
    print(f"  历史均值（月度）:            {portfolio_returns.mean():.4f} ({portfolio_returns.mean()*100:.2f}%)")
    print(f"  Historical std  (monthly):   {portfolio_returns.std(ddof=1):.4f} ({portfolio_returns.std(ddof=1)*100:.2f}%)")
    print(f"  历史标准差（月度）:          {portfolio_returns.std(ddof=1):.4f} ({portfolio_returns.std(ddof=1)*100:.2f}%)")

    # -------------------------------------------------------------------------
    # Step 6: Run Monte Carlo simulation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Running Monte Carlo Simulation | 运行蒙特卡罗模拟")
    print("-" * 70)

    n_years = 30
    n_paths = 1000
    goal_wealth = 3.0  # Target: triple the initial investment

    print(f"  Simulation horizon: {n_years} years ({n_years * 12} months)")
    print(f"  模拟期限: {n_years} 年 ({n_years * 12} 个月)")
    print(f"  Number of paths: {n_paths}")
    print(f"  路径数量: {n_paths}")
    print(f"  Method: bootstrap (resampling with replacement)")
    print(f"  方法: 自助法（有放回抽样）")
    print(f"  Target wealth (goal_wealth): {goal_wealth}x")
    print(f"  目标财富: {goal_wealth} 倍")

    results = aggregate_mc_results(
        portfolio_returns=portfolio_returns,
        n_years=n_years,
        n_paths=n_paths,
        method="bootstrap",
        goal_wealth=goal_wealth,
        random_state=42,
    )

    # -------------------------------------------------------------------------
    # Step 7: Print Monte Carlo summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== Monte Carlo Summary | 蒙特卡罗模拟摘要 ===")
    print("=" * 70)

    terminal_summary = results["terminal_summary"]
    quantile_paths = results["quantile_paths"]

    print(f"\n--- Terminal Wealth Statistics | 终端财富统计 ---")
    print(f"  Mean terminal wealth:   {terminal_summary['mean']:.2f}x")
    print(f"  终端财富均值:           {terminal_summary['mean']:.2f} 倍")
    print(f"  Std terminal wealth:    {terminal_summary['std']:.2f}")
    print(f"  终端财富标准差:         {terminal_summary['std']:.2f}")
    print(f"  Min terminal wealth:    {terminal_summary['min']:.2f}x")
    print(f"  终端财富最小值:         {terminal_summary['min']:.2f} 倍")
    print(f"  Max terminal wealth:    {terminal_summary['max']:.2f}x")
    print(f"  终端财富最大值:         {terminal_summary['max']:.2f} 倍")

    print(f"\n--- Success Probabilities | 成功概率 ---")
    print(f"  P(terminal >= {goal_wealth}x):  {terminal_summary['prob_above_goal']:.1%}")
    print(f"  P(终端财富 >= {goal_wealth}倍): {terminal_summary['prob_above_goal']:.1%}")
    print(f"  P(terminal < 1.0):      {terminal_summary['prob_below_1']:.1%}")
    print(f"  P(亏损):                {terminal_summary['prob_below_1']:.1%}")

    print(f"\n--- Quantiles of Terminal Wealth | 终端财富分位数 ---")
    quantiles = terminal_summary["quantiles"]
    for q, value in quantiles.items():
        pct = int(q * 100)
        print(f"  q_{pct:2d}%:  {value:>8.2f}x")

    print(f"\n--- Quantile Paths (sample) | 分位数路径（样例）---")
    print("  Showing first 5 and last 5 time steps for q_5, q_50, q_95:")
    print("  显示 q_5, q_50, q_95 的前 5 和后 5 个时间步:")
    print()
    print("  First 5 rows (months 0-4):")
    print(quantile_paths.head().to_string(index=True))
    print()
    print("  Last 5 rows (months {}-{}):".format(
        len(quantile_paths) - 5, len(quantile_paths) - 1
    ))
    print(quantile_paths.tail().to_string(index=True))

    # -------------------------------------------------------------------------
    # Step 8: Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo Complete! | 演示完成！")
    print("=" * 70)
    print("\nYou can import these functions in other modules:")
    print("您可以在其他模块中导入这些函数:")
    print("  from mc_simulation import aggregate_mc_results")
    print("  results = aggregate_mc_results(portfolio_returns, n_years=30, goal_wealth=3.0)")
    print("=" * 70)

