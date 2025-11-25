You are a senior quantitative Python developer.
My goal is to learn Python, NumPy, and pandas by building a Tactical Asset Allocation (TAA) engine step by step.

This file is the **second core module** of the project, after `mock_data.py`.  
It must be fully self-contained and runnable on its own (no project imports required).

==================================================
# 1. File goal (taa_signal_engine.py)

Implement the TAA weight-adjustment engine:

- Input:
  - A 16-dimensional SAA weight vector `w_saa`
  - A time series of macro quadrants (one per month)

- Internal logic:
  - Map each of the 16 strategies to a higher-level asset class
  - For each quadrant, read a pre-defined asset-class tilt vector Δw_asset,t
  - Distribute these tilts down to the 16 strategies proportional to their SAA weights
  - Apply strategy-specific sensitivity coefficients β
  - Normalize to obtain final weights `w_final,t` (one vector per month)

- Output:
  - A `(T × 16)` DataFrame of final monthly weights

This file is focused on:
- Vectorized operations with NumPy and pandas
- Group-by logic by asset class
- Clean, well-documented implementation of the TAA math

==================================================
# 2. Libraries and constraints (must follow)

- Target Python version: **3.9** (must also run on 3.11)
- Allowed libraries:
  - Python standard library
  - `numpy`
  - `pandas`
- Forbidden:
  - `scipy`, `sklearn`, `statsmodels`, `torch`, `tensorflow`, etc.
- The file must be **standalone**:
  - Do NOT import from other project modules
  - Use synthetic data in the `__main__` section for demonstration
- Use `np.random.seed(42)` in the demo to keep results reproducible
- Use type hints where reasonable
- Use clear docstrings + comments to help a beginner understand NumPy/pandas

==================================================
# 3. Strategy list and asset-class mapping (fixed structure)

Use exactly the following 16 strategy names, in this order:

[
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
 "StructuredProduct"
]

Define a mapping from each strategy to a **higher-level asset class**.

Use the following asset classes:

- "Cash"
- "Bond"
- "Equity"
- "Commodity"
- "Alternative"

A reasonable mapping to implement:

- "Cash"                       → "Cash"
- "DepositFixedIncome"         → "Bond"
- "PureBond"                   → "Bond"
- "NonStandardFixedIncome"     → "Bond"
- "FixedIncomePlus"            → "Bond"
- "OverseasBond"               → "Bond"
- "BalancedFund"               → "Equity"
- "EquityA"                    → "Equity"
- "EquityOverseas"             → "Equity"
- "OverseasBalanced"           → "Equity"
- "Commodity"                  → "Commodity"
- "HedgeFund"                  → "Alternative"
- "RealEstate"                 → "Alternative"
- "PrivateEquity"              → "Alternative"
- "OverseasAlternative"        → "Alternative"
- "StructuredProduct"          → "Alternative"

Use a pandas DataFrame or a simple dict/list to store this metadata inside the file.

==================================================
# 4. Quadrants and asset-class tilt matrix (Δw_asset,t)

We assume 4 macro quadrants:

- "Recovery"
- "Overheat"
- "Stagflation"
- "Recession"

Define a **quadrant → asset-class tilt** mapping, for example as a dictionary of dictionaries:

```python
delta_asset = {
    "Recovery":   {"Equity": 0.05,  "Bond": -0.03, "Commodity": 0.02, "Alternative": 0.02, "Cash": -0.04},
    "Overheat":   {"Equity": -0.02, "Bond": -0.03, "Commodity": 0.05, "Alternative": 0.03, "Cash": 0.00},
    "Stagflation":{"Equity": -0.04, "Bond": 0.00,  "Commodity": 0.04, "Alternative": 0.02, "Cash": -0.02},
    "Recession":  {"Equity": -0.05, "Bond": 0.05,  "Commodity": -0.02,"Alternative": -0.02,"Cash": 0.04},
}
````

You may adjust the exact numbers if needed, but:

* All asset classes must appear in each quadrant
* The sum of tilts per quadrant does not have to be 0
* These are **tilts** (Δw), not final weights

==================================================

# 5. Sensitivity coefficients β for each strategy

Define a 16-dimensional vector of β values, representing how sensitive each strategy is to TAA tilts.

Example (you can hard-code it as a NumPy array aligned with the strategy order):

* Cash-like & short-term instruments: lower β (e.g. 0.3–0.5)
* Core bonds: medium β (e.g. 0.7–0.9)
* Equities & alternatives: higher β (e.g. 1.0–1.3)

For example (just an example, you can fine-tune inside the code):

```text
[0.3, 0.7, 0.8, 0.8, 0.9, 0.9,
 1.0, 1.1, 1.1, 1.0,
 1.2,
 1.1, 1.1, 1.2, 1.2, 1.0]
```

Store this as a NumPy array, and ensure the ordering is consistent with the strategy list.

==================================================

# 6. Functions to implement (public API of this file)

The file must implement at least the following functions:

1. `get_strategy_metadata()`

```python
def get_strategy_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per strategy and columns:
        - 'strategy'     : strategy name (string)
        - 'asset_class'  : asset class name (string)
        - 'beta'         : sensitivity coefficient (float)
    """
```

2. `normalize_weights(w: np.ndarray) -> np.ndarray`

```python
def normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D weight vector:
        1) Set negative values to 0
        2) If sum > 0, divide by the sum so that weights sum to 1
        3) If all values are <= 0 (sum == 0), fall back to a uniform allocation
    """
```

3. `compute_raw_strategy_tilts(...)`

```python
def compute_raw_strategy_tilts(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    For each month t and each strategy i, compute the unadjusted tilt Δw_strategy^(0)_{i,t}
    by distributing the asset-class tilt Δw_asset,t within each asset class proportional
    to the SAA weights.

    Arguments:
        w_saa    : np.ndarray of shape (16,), SAA weights (sum to 1)
        quadrants: pd.Series of length T, each value in {"Recovery", "Overheat", "Stagflation", "Recession"}
        metadata : DataFrame from get_strategy_metadata(), including 'asset_class'

    Returns:
        delta_w_raw: pd.DataFrame of shape (T, 16), index aligned with quadrants index,
                     columns are strategy names.
    """
```

The formula inside this function (for each asset class AC at time t):

* Let Δw_asset,t(AC) be the tilt for this asset class
* Let SAA_AC_sum = sum of SAA weights of all strategies in this asset class
* For each strategy i in this asset class:

```text
Δw_strategy^(0)_{i,t} = Δw_asset,t(AC) * w_saa[i] / SAA_AC_sum
```

Handle edge cases:

* If SAA_AC_sum == 0 for an asset class, distribute 0 tilt to its strategies.

4. `apply_beta_adjustment(...)`

```python
def apply_beta_adjustment(
    delta_w_raw: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply β_i to each strategy tilt:
        Δw_strategy_{i,t} = beta_i * Δw_strategy^(0)_{i,t}

    Arguments:
        delta_w_raw: DataFrame of shape (T, 16)
        metadata   : DataFrame with 'strategy' and 'beta'

    Returns:
        delta_w_beta: DataFrame of shape (T, 16) with β-adjusted tilts.
    """
```

5. `compute_final_weights_over_time(...)`

```python
def compute_final_weights_over_time(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    High-level function that:
        1) Computes raw tilts Δw_strategy^(0)
        2) Applies β adjustment
        3) Adds the tilts to w_saa
        4) Normalizes each month's weights so they sum to 1 (and no negatives)

    Returns:
        final_weights: DataFrame of shape (T, 16),
                       index aligned with quadrants,
                       columns are strategy names.
    """
```

==================================================

# 7. Main demo block

At the bottom of the file, include:

```python
if __name__ == "__main__":
    # 1) Set random seed
    # 2) Construct a synthetic SAA weight vector (e.g. using random positive numbers normalized to 1)
    # 3) Construct a synthetic quadrant Series for, say, 120 months
    #    - e.g. by repeating each quadrant for 12 months in some order
    # 4) Call get_strategy_metadata()
    # 5) Call compute_final_weights_over_time(...)
    # 6) Print:
    #    - first few rows of final_weights
    #    - a check that each row sums to ~1.0
    #    - the average tilt per asset class over time (optional, for inspection)
```

The main block should:

* Show how the functions are used together
* Print outputs in a clear, human-readable format
* Demonstrate basic NumPy/pandas usage (e.g. `DataFrame.head()`, `.sum(axis=1)`, etc.)

==================================================

# 8. Final output rules

* Return **only** the final Python code for this single file.
* Do NOT include Markdown fences like ```python.
* Do NOT add extra prose explanations outside comments/docstrings.
* Ensure the file can be run as:

  python taa_signal_engine.py
