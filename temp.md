下面给你 **完整、一次成型、可直接丢给 Cursor 的 backtest_engine.py Prompt**。
它会把整个链路串起来：

```
core/mock_data.py     → 生成数据
core/taa_signal_engine.py  → TAA 最终权重（w_final,t）
core/backtest_engine.py    → 用 w_final,t + returns 回测指标
```

你的目标：

* 练 numpy / pandas
* 模块化
* backtest_engine.py **只做“数学计算 + 回测指标”**
* 不读写 CSV（读 CSV 属于 scripts/）
* 只对传入的 DataFrame 做运算
* 输出：Return series、年化收益、年化波动、Sharpe、最大回撤、Calmar、胜率

你可以直接复制下面整个 Prompt 到 Cursor 生成 `core/backtest_engine.py`。

---

# ✅ 完整 Prompt：`core/backtest_engine.py`

````
You are a senior quantitative Python developer.
Generate a standalone module: `core/backtest_engine.py`.

This module must work with the other two modules:

    core/mock_data.py
    core/taa_signal_engine.py

And it must serve as the “performance analytics engine” of the TAA project.

==================================================
# 1. File goal (backtest_engine.py)

Implement a clean, NumPy/pandas-based backtesting engine that calculates:

- Portfolio monthly returns (weight × return)
- Cumulative return curve
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown (MDD)
- Calmar ratio
- Win rate（上涨月份占比）

This file should NOT:
- Read or write CSV
- Include plotting
- Depend on any external packages outside numpy/pandas

==================================================
# 2. Libraries & constraints

- Python 3.9+, must run on 3.11
- Allowed:
    - numpy
    - pandas
- Forbidden:
    - scipy, sklearn, statsmodels, torch, tensorflow
- Code must be explicit and beginner-friendly
- Use type hints and docstrings
- File must be standalone (no relative imports outside core/)

==================================================
# 3. Inputs to this module

The main function in this file will receive:

1) `returns_df`:  
    - A pandas DataFrame (T × 16)  
    - Index: monthly DatetimeIndex  
    - Columns: the 16 fixed strategies  
    - Values: decimal returns, e.g. 0.01 = +1%

2) `weights_df`:  
    - A pandas DataFrame (T × 16)  
    - Output from `taa_signal_engine.compute_final_weights_over_time()`  
    - Index: same as `returns_df.index`  
    - Values: final weights w_final,t (sum to 1 per row)

The caller (scripts) is responsible for ensuring the shapes and indices match.

==================================================
# 4. Required functions

Implement the following functions.

--------------------------------------------------
4.1 compute_portfolio_returns

```python
def compute_portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame
) -> pd.Series:
    """
    Computes portfolio monthly returns:

        r_{p,t} = sum_i( w_{i,t} * r_{i,t} )

    Requirements:
        - returns_df and weights_df must align by index and columns
        - return a pandas Series indexed by time
        - name the Series "portfolio_return"
    """
````

---

4.2 compute_cumulative_return

```python
def compute_cumulative_return(
    portfolio_returns: pd.Series
) -> pd.Series:
    """
    Computes cumulative return curve:

        C0 = 1.0
        Ct = prod_{k=1..t}(1 + r_{p,k})

    Returns:
        pandas Series with same index as portfolio_returns.
        Name = "cumulative_return".
    """
```

---

4.3 annualized_return

```python
def annualized_return(
    portfolio_returns: pd.Series
) -> float:
    """
    Annualized return (geometric):

        (1 + R_total)^(12 / T) - 1

    where R_total = cumulative_return[-1] - 1.

    Assume monthly frequency.
    """
```

---

4.4 annualized_volatility

```python
def annualized_volatility(
    portfolio_returns: pd.Series
) -> float:
    """
    Annualized volatility:

        std(r_monthly) * sqrt(12)
    """
```

---

4.5 sharpe_ratio

```python
def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Sharpe ratio:

        (annualized_return - risk_free_rate) /
        annualized_volatility

    If volatility is 0, return np.nan.
    """
```

---

4.6 max_drawdown

```python
def max_drawdown(
    cumulative_returns: pd.Series
) -> float:
    """
    Computes Maximum Drawdown (MDD):

        MDD = min( (Ct / peak) - 1 )

    where peak is the running maximum of cumulative_returns.

    Return a negative number (e.g., -0.23).
    """
```

---

4.7 calmar_ratio

```python
def calmar_ratio(
    portfolio_returns: pd.Series,
    cumulative_returns: pd.Series
) -> float:
    """
    Calmar ratio:

        annualized_return / abs(MDD)

    If MDD is 0 or positive (should not happen), return np.nan.
    """
```

---

4.8 win_rate

```python
def win_rate(
    portfolio_returns: pd.Series
) -> float:
    """
    Win rate:

        (# of months with r > 0) / T
    """
```

---

4.9 aggregate_backtest_metrics

Implement a convenience wrapper that runs all metrics:

```python
from typing import Dict, Any

def aggregate_backtest_metrics(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    High-level wrapper:
        1) Compute portfolio monthly returns
        2) Compute cumulative return curve
        3) Compute all metrics:
            - annualized_return
            - annualized_volatility
            - sharpe_ratio
            - max_drawdown
            - calmar_ratio
            - win_rate

    Returns a dictionary with keys:
        "portfolio_returns"
        "cumulative_returns"
        "annualized_return"
        "annualized_volatility"
        "sharpe_ratio"
        "max_drawdown"
        "calmar_ratio"
        "win_rate"
    """
```

==================================================

# 5. Main demo block (if **name** == "**main**")

At the bottom, implement a simple demonstration:

```python
if __name__ == "__main__":
    # 1) Import the mock dataset generator
    #    but ONLY via sys.path insertion, not relative imports.
    #
    #    from core/mock_data import create_mock_dataset
    #
    #    (Follow the same path strategy used in scripts/run_mock_data.py)

    # 2) Use create_mock_dataset(n_months=120)
    #    to generate w_saa, returns_df, quadrants.

    # 3) Import TAA engine:
    #       from core.taa_signal_engine import (
    #           get_strategy_metadata, DELTA_ASSET, compute_final_weights_over_time
    #       )

    # 4) Compute final TAA-adjusted weights:
    #       metadata = get_strategy_metadata()
    #       weights_df = compute_final_weights_over_time(
    #                        w_saa=w_saa,
    #                        quadrants=quadrants,
    #                        metadata=metadata,
    #                        delta_asset=DELTA_ASSET,
    #                    )

    # 5) Run aggregate_backtest_metrics(...)
    # 6) Print all metrics in a human-readable way
    # 7) Print the first few rows of portfolio_returns and cumulative_returns
```

The main block is a standalone demonstration for learning —
the scripts/ directory will implement full production-like usage.

==================================================

# 6. Final output rules

* Return ONLY the Python code for this file.
* No markdown fences like ```python.
* No external explanation.
* The file must run as:

  python core/backtest_engine.py

from anywhere, as long as core/ is on sys.path.

```

---

# 🎉 下一步你可以选择：

### ✔ 我帮你写 `scripts/run_backtest.py` 的 Prompt（读 CSV → 读 TAA → 回测 → 打印报表）

### ✔ 或写 `core/mc_simulation.py` Prompt（蒙特卡罗、方差缩减、路径模拟）

### ✔ 或写 `AGENTS.md`（完整项目规范，让 Cursor 自动遵守所有规则）

你想做哪个？
```
