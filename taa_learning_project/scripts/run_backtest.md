You are a senior Python developer.
Generate a single Python script: `scripts/run_backtest.py`.

This script will:

    1) Load pre-generated mock data from ../data (CSV files)
    2) Compute TAA-adjusted final weights using core/taa_signal_engine.py
    3) Run backtest metrics using core/backtest_engine.py
    4) Print a small, human-readable backtest report

It is a COMMAND-LINE script (no plotting, no web server).

==================================================
# 1. Project layout & relative paths

The repo layout (simplified) is:

taa_learning_project/
│
├── data/
│   ├── mock_returns.csv
│   ├── mock_quadrants.csv
│   └── mock_saa_weights.csv
│
├── core/
│   ├── mock_data.py
│   ├── taa_signal_engine.py
│   ├── backtest_engine.py
│   └── ...
│
└── scripts/
    ├── run_mock_data.py
    ├── run_taa_signal.py
    └── run_backtest.py   ← THIS FILE

`run_backtest.py` must **resolve paths relative to itself**, not assume where the project root is.

Use:

```python
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORE_DIR = PROJECT_ROOT / "core"
````

==================================================

# 2. Libraries & constraints

* Python 3.9+, must run on 3.11
* Allowed:

  * numpy
  * pandas
  * python standard library: pathlib, typing, sys, os
* Forbidden:

  * scipy, sklearn, statsmodels, torch, tensorflow, etc.

==================================================

# 3. Behavior requirements

The script should do the following:

1. Resolve project paths (script_root, project_root, data_dir, core_dir).

2. Ensure `core_dir` is on `sys.path` so we can import:

```python
from mock_data import create_mock_dataset  # optional fallback
from taa_signal_engine import (
    get_strategy_metadata,
    DELTA_ASSET,
    compute_final_weights_over_time,
)
from backtest_engine import (
    aggregate_backtest_metrics,
)
```

3. Try to load existing CSV data from `data_dir`:

* `mock_saa_weights.csv`

  * index: strategy names
  * column: "weight"
  * convert to a NumPy array `w_saa` (shape (16,))

* `mock_returns.csv`

  * index: dates (parse as DatetimeIndex)
  * columns: 16 strategy names
  * values: decimal returns

* `mock_quadrants.csv`

  * index: dates (parse as DatetimeIndex)
  * column: "quadrant"
  * convert to a pandas Series `quadrants`

If any of these files is missing, the script should:

* print a short message asking the user to run `python scripts/run_mock_data.py` first
* then exit gracefully (e.g., via `raise SystemExit(1)`)

4. After successfully loading data:

* Call `get_strategy_metadata()` to get the strategy/asset_class mapping.
* Call `compute_final_weights_over_time(...)` to produce `final_weights_df`:

```python
weights_df = compute_final_weights_over_time(
    w_saa=w_saa,
    quadrants=quadrants,
    metadata=metadata,
    delta_asset=DELTA_ASSET,
)
```

* Ensure that:

  * `weights_df.index` aligns with `returns_df.index`
  * each row of `weights_df` sums approximately to 1 (you can assume TAA engine handles that; no need to re-normalize here)

5. Run the backtest:

Use:

```python
results = aggregate_backtest_metrics(
    returns_df=returns_df,
    weights_df=weights_df,
)
```

`results` is expected to be a dictionary with at least:

* "portfolio_returns"    : pd.Series
* "cumulative_returns"   : pd.Series
* "annualized_return"    : float
* "annualized_volatility": float
* "sharpe_ratio"         : float
* "max_drawdown"         : float
* "calmar_ratio"         : float
* "win_rate"             : float

6. Print a mini backtest report:

Print something like:

```text
=== Backtest Summary ｜回测摘要 ===
Data period          : {start_date} → {end_date} (T = {T} months)

Annualized Return    : {annualized_return:.2%}
Annualized Volatility: {annualized_volatility:.2%}
Sharpe Ratio         : {sharpe_ratio:.2f}
Max Drawdown (MDD)   : {max_drawdown:.2%}
Calmar Ratio         : {calmar_ratio:.2f}
Win Rate             : {win_rate:.2%}

=== Sample of Portfolio Return ｜组合月度收益示例（前 5 行） ===
{portfolio_returns.head().round(4)}

=== Sample of Cumulative Curve ｜累计净值示例（前 5 行） ===
{cumulative_returns.head().round(4)}
```

Use bilingual headings (English + Chinese) as shown above to help human inspection.

==================================================

# 4. Functions to implement

To keep the script organized, implement at least these helpers:

1. `get_paths() -> Tuple[Path, Path, Path, Path]`

```python
from typing import Tuple

def get_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Returns:
        script_path  : absolute Path of this script
        project_root : parent directory of scripts/
        data_dir     : project_root / "data"
        core_dir     : project_root / "core"
    """
```

2. `ensure_core_on_syspath(core_dir: Path) -> None`

* If `str(core_dir)` is not in `sys.path`, insert it at position 0.

3. `load_dataset_from_csv(data_dir: Path) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]`

* Loads the three CSVs.
* Parses DatetimeIndex for returns/quadrants.
* Returns `w_saa`, `returns_df`, `quadrants`.
* If any file is missing, raise `FileNotFoundError`.

4. `run_backtest(w_saa, returns_df, quadrants) -> Dict[str, Any]`

* Gets metadata via `get_strategy_metadata()`
* Computes `weights_df` with `compute_final_weights_over_time(...)`
* Calls `aggregate_backtest_metrics(...)`
* Returns the `results` dict.

5. `print_backtest_report(results: Dict[str, Any]) -> None`

* Extracts:

  * portfolio_returns
  * cumulative_returns
  * annualized_return
  * annualized_volatility
  * sharpe_ratio
  * max_drawdown
  * calmar_ratio
  * win_rate
* Infers:

  * start_date, end_date, T
* Prints the human-readable report as described above.

==================================================

# 5. Main entrypoint

At the bottom:

```python
if __name__ == "__main__":
    script_path, project_root, data_dir, core_dir = get_paths()
    ensure_core_on_syspath(core_dir)

    try:
        w_saa, returns_df, quadrants = load_dataset_from_csv(data_dir)
    except FileNotFoundError:
        print("Data files not found in ../data/.")
        print("Please run:  python scripts/run_mock_data.py  first.")
        raise SystemExit(1)

    results = run_backtest(w_saa, returns_df, quadrants)

    print_backtest_report(results)
```

==================================================

# 6. Final output rules

* Return ONLY the final Python code for `scripts/run_backtest.py`.
* Do NOT include Markdown fences like ```python.
* Do NOT include explanatory prose outside comments/docstrings.
* The script must run correctly when executed as:

  python scripts/run_backtest.py

from the project root (where core/, data/, scripts/ live).

