You are a senior quantitative Python developer.
Generate a standalone module: `core/mc_simulation.py`.

The goal is to implement a simple Monte Carlo engine at the **portfolio level**,
for educational purposes (learning NumPy & pandas), integrated with:

    core/mock_data.py
    core/taa_signal_engine.py
    core/backtest_engine.py

==================================================
# 1. File goal (mc_simulation.py)

Implement tools to:

1. Take a **historical portfolio monthly return series**
2. Simulate many possible **future wealth paths** over N years
   using Monte Carlo methods
3. Summarize the distribution of terminal wealth and key quantiles

This module should:

- Work mainly at the **portfolio level** (not per-asset), i.e. use a single
  Series `portfolio_returns` as input.
- Provide two simulation methods:
    - "normal": assume returns ~ Normal(mean, std)
    - "bootstrap": resample historical monthly returns with replacement
- Be written in clear, beginner-friendly NumPy/pandas code.

The module must NOT:

- Read or write CSV files
- Produce plots
- Depend on any non-standard library beyond NumPy and pandas

==================================================
# 2. Libraries & constraints

- Python 3.9+, must run on 3.11
- Allowed:
    - numpy
    - pandas
    - python stdlib (typing, pathlib, sys, math, etc.)
- Forbidden:
    - scipy, sklearn, statsmodels, torch, tensorflow, etc.

- Use type hints and docstrings.
- Code must be explicit and easy to follow.

==================================================
# 3. Core concepts & assumptions

- Input `portfolio_returns`:
    - pandas Series
    - monthly frequency
    - decimal returns (0.01 = +1%)

- Time horizon:
    - `n_years`: number of future years to simulate (e.g. 30 years)
    - monthly steps: `T = n_years * 12`

- Number of paths:
    - `n_paths`: number of Monte Carlo scenarios (e.g. 1000)

- Starting wealth:
    - assume initial wealth W0 = 1.0 for all simulations

- Wealth evolution:
    - For each path k and month t:
        W_{t+1} = W_t * (1 + r_{t,k})

where r_{t,k} is simulated monthly return for that path and month.

==================================================
# 4. Functions to implement

--------------------------------------------------
4.1 simulate_paths_from_portfolio_returns

```python
from typing import Literal

def simulate_paths_from_portfolio_returns(
    portfolio_returns: pd.Series,
    n_years: int = 30,
    n_paths: int = 1000,
    method: Literal["normal", "bootstrap"] = "bootstrap",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate future wealth paths using historical portfolio monthly returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical monthly returns of the portfolio (decimal format).
    n_years : int, default 30
        Number of future years to simulate.
    n_paths : int, default 1000
        Number of Monte Carlo paths.
    method : {"normal", "bootstrap"}, default "bootstrap"
        - "normal"   : sample monthly returns from Normal(mu, sigma)
                       where mu = mean(hist), sigma = std(hist).
        - "bootstrap": sample monthly returns with replacement from
                       the historical Series.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    wealth_paths : pd.DataFrame
        Shape: (T + 1, n_paths), where T = n_years * 12.
        - Index: a simple integer index or a DatetimeIndex is acceptable.
                 For simplicity, you can use a RangeIndex from 0..T
                 where 0 is the initial point (wealth = 1.0).
        - Columns: "path_0", "path_1", ..., "path_{n_paths-1}"
        - Values: wealth levels over time, starting from 1.0 at t=0.
    """
````

Implementation details:

* Set the NumPy random seed using `np.random.default_rng(random_state)`.
* Let `T = n_years * 12`.
* For method "normal":

  * Estimate `mu = portfolio_returns.mean()`
  * Estimate `sigma = portfolio_returns.std(ddof=1)`
  * Generate a (T × n_paths) matrix of simulated monthly returns from
    N(mu, sigma).
* For method "bootstrap":

  * Convert historical returns to a NumPy array.
  * Use integer random indices to sample with replacement:
    shape (T, n_paths).
* Transform simulated monthly returns into wealth paths by cumulative product:

  * Start from wealth 1.0 at t=0.
  * For each path: W_t = prod_{k=1..t} (1 + r_k), prepend W_0 = 1.0.
* Build a DataFrame with index 0..T and columns "path_i".

---

4.2 summarize_terminal_wealth

```python
from typing import Dict, Any, Sequence

def summarize_terminal_wealth(
    wealth_paths: pd.DataFrame,
    confidence_levels: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    goal_wealth: float = 1.0,
) -> Dict[str, Any]:
    """
    Summarize the terminal wealth distribution at the final time step.

    Parameters
    ----------
    wealth_paths : pd.DataFrame
        Output from simulate_paths_from_portfolio_returns.
        Shape: (T + 1, n_paths)
    confidence_levels : sequence of float, default (0.05, 0.25, 0.50, 0.75, 0.95)
        Quantiles to compute on the terminal wealth distribution.
        Values should be in (0, 1).
    goal_wealth : float, default 1.0
        A simple target wealth level to evaluate success probability.

    Returns
    -------
    summary : Dict[str, Any]
        Keys:
            - "terminal_wealth"   : 1D NumPy array or Series with shape (n_paths,)
            - "quantiles"         : dict mapping q -> wealth_q
            - "mean"              : float
            - "std"               : float
            - "min"               : float
            - "max"               : float
            - "prob_above_goal"   : float  (fraction of paths with W_T >= goal_wealth)
            - "prob_below_1"      : float  (fraction of paths with W_T < 1.0)
    """
```

Implementation details:

* Terminal wealth = last row of `wealth_paths` (e.g. `wealth_paths.iloc[-1]`).
* Convert to a NumPy array for computations.
* Compute quantiles via `np.quantile`.
* Compute simple stats: mean, std, min, max.
* Compute:

  * prob_above_goal: fraction of terminal wealth >= goal_wealth
  * prob_below_1   : fraction of terminal wealth < 1.0

---

4.3 summarize_quantile_paths (optional but useful)

```python
def summarize_quantile_paths(
    wealth_paths: pd.DataFrame,
    confidence_levels: Sequence[float] = (0.05, 0.50, 0.95),
) -> pd.DataFrame:
    """
    Compute quantile paths over time.

    For each time step t and each q in confidence_levels,
    compute the q-quantile of wealth over all paths.

    Returns
    -------
    quantile_paths : pd.DataFrame
        - Index: same as wealth_paths.index
        - Columns: e.g. "q_5", "q_50", "q_95" for q=0.05,0.50,0.95
    """
```

Implementation details:

* Use `np.quantile` along axis=1.
* Column naming: "q_5", "q_25", "q_50", etc. (100 * q as integer).

---

4.4 aggregate_mc_results

```python
def aggregate_mc_results(
    portfolio_returns: pd.Series,
    n_years: int = 30,
    n_paths: int = 1000,
    method: str = "bootstrap",
    goal_wealth: float = 1.0,
) -> Dict[str, Any]:
    """
    High-level wrapper:

        1) Simulate wealth paths from portfolio_returns.
        2) Summarize terminal wealth.
        3) Summarize quantile paths (e.g. 5%, 50%, 95%).

    Returns a dict with keys:
        - "wealth_paths"      : pd.DataFrame
        - "terminal_summary"  : dict
        - "quantile_paths"    : pd.DataFrame
    """
```

Implementation details:

* Call `simulate_paths_from_portfolio_returns(...)`
* Call `summarize_terminal_wealth(...)`
* Call `summarize_quantile_paths(...)` with a shorter set of quantiles
  (e.g. (0.05, 0.50, 0.95)).

==================================================

# 5. Demonstration (**main**): integrate with core modules

At the bottom of the file, add a demo that connects everything together.

We assume the directory layout:

taa_learning_project/
│
├── core/
│   ├── mock_data.py
│   ├── taa_signal_engine.py
│   ├── backtest_engine.py
│   └── mc_simulation.py  ← THIS FILE
└── data/
└── ...

In `if __name__ == "__main__":`, do:

1. Use `pathlib` and `sys.path` to make sure `core/` is on sys.path if needed.
   For example:

```python
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
CORE_DIR = THIS_FILE.parent
PROJECT_ROOT = CORE_DIR.parent

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
```

2. Import the necessary functions from other core modules:

```python
from mock_data import create_mock_dataset
from taa_signal_engine import (
    get_strategy_metadata,
    DELTA_ASSET,
    compute_final_weights_over_time,
)
from backtest_engine import (
    compute_portfolio_returns,
)
```

(Assume backtest_engine.py exposes `compute_portfolio_returns`.)

3. Generate a synthetic dataset:

```python
w_saa, returns_df, quadrants = create_mock_dataset(n_months=120)
```

4. Compute TAA-adjusted weights:

```python
metadata = get_strategy_metadata()
weights_df = compute_final_weights_over_time(
    w_saa=w_saa,
    quadrants=quadrants,
    metadata=metadata,
    delta_asset=DELTA_ASSET,
)
```

5. Compute portfolio monthly returns:

```python
portfolio_returns = compute_portfolio_returns(
    returns_df=returns_df,
    weights_df=weights_df,
)
```

6. Run Monte Carlo simulation:

```python
results = aggregate_mc_results(
    portfolio_returns=portfolio_returns,
    n_years=30,
    n_paths=1000,
    method="bootstrap",
    goal_wealth=3.0,  # for example, target 3x wealth
)
```

7. Print a small, human-readable summary:

* Print basic info about input:

  * length of historical series
  * mean / std of portfolio_returns
* Print terminal wealth summary:

  * quantiles
  * mean / std
  * prob_above_goal
  * prob_below_1
* Optionally print the last few rows of `quantile_paths`.

The headings can be bilingual, e.g.:

```text
=== Monte Carlo Summary ｜蒙特卡罗模拟摘要 ===
Historical monthly returns length: ...
Historical mean (monthly): ...
Historical std  (monthly): ...

Target wealth (goal_wealth): ...
P(terminal ≥ goal_wealth): ...
P(terminal < 1.0)        : ...

Quantiles of terminal wealth:
  q_5%  : ...
  q_25% : ...
  q_50% : ...
  q_75% : ...
  q_95% : ...
```

==================================================

# 6. Final output rules

* Return ONLY the Python code for `core/mc_simulation.py`.
* Do NOT include Markdown fences like ```python.
* Do NOT include explanatory prose outside comments/docstrings.
* The file must run as:

  python core/mc_simulation.py

from the project root, as long as core/ is on sys.path.

