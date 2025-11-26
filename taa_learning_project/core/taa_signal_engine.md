You are a senior quantitative Python developer.
My goal is to learn Python, NumPy, and pandas by building a Tactical Asset Allocation (TAA) engine step by step.

This file is the second core module of the project, after `mock_data.py`.  
It must be fully self-contained and runnable on its own (no project imports required).

==================================================
# 1. File goal (taa_signal_engine.py)

Implement the **TAA weight-adjustment engine WITHOUT any beta/sensitivity layer**.

- Inputs:
  - A 16-dimensional SAA weight vector `w_saa` (baseline strategic allocation)
  - A pandas Series of macro quadrants, one per month

- Logic:
  1) Map each of the 16 strategies to one of four asset classes:
        - Cash
        - Bond
        - Equity
        - Alternative
  2) For each quadrant, obtain an asset-class tilt Δw_asset,t
  3) For each asset class, allocate the tilt to its member strategies
     **proportional to their SAA weights**
  4) Add the strategy-level tilts on top of `w_saa`
  5) Row-wise normalize to get final monthly weights (`w_final,t`)

- Output:
  - A `(T × 16)` DataFrame of final weights

This module focuses on:
- vectorized NumPy operations
- pandas DataFrame manipulation
- clean implementation of TAA math

==================================================
# 2. Libraries & constraints

- Python 3.9+ (must run on 3.11 locally)
- Allowed:
    - numpy
    - pandas
    - python standard library
- Forbidden:
    - scipy, sklearn, statsmodels, torch, tensorflow, etc.
- File must run standalone:
    - Do NOT import other project modules
    - Use synthetic data in the `__main__` demo
- Use `np.random.seed(42)` in the demo section
- Use type hints and docstrings
- Keep the code explicit and beginner-friendly

==================================================
# 3. Strategy universe (fixed order)

Use these 16 strategy names in this exact order:

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

These names must be the column order for all `(T × 16)` DataFrames in this module.

==================================================
# 4. Asset-class mapping (4 categories only)

Each strategy belongs to exactly ONE of the four asset classes:

- "Cash"
- "Bond"
- "Equity"
- "Alternative"

Implement the mapping as:

- Cash                       → Cash
- DepositFixedIncome         → Bond
- PureBond                   → Bond
- NonStandardFixedIncome     → Bond
- FixedIncomePlus            → Bond
- OverseasBond               → Bond

- BalancedFund               → Equity
- EquityA                    → Equity
- EquityOverseas             → Equity
- OverseasBalanced           → Equity

- Commodity                  → Alternative
- HedgeFund                  → Alternative
- RealEstate                 → Alternative
- PrivateEquity              → Alternative
- OverseasAlternative        → Alternative
- StructuredProduct          → Alternative

Store this metadata inside the file (e.g. as a pandas DataFrame).

==================================================
# 5. Macro quadrants

We consider four macro quadrants:

- "Recovery"
- "Overheat"
- "Stagflation"
- "Recession"

The quadrants Series is a pandas Series indexed by time (e.g., monthly dates) with values from this set.

==================================================
# 6. Asset-class tilt matrix (Δw_asset,t)

Define a **quadrant → asset-class tilt** mapping for the four asset classes:

- Cash
- Bond
- Equity
- Alternative

Example (you may tweak exact numbers, but keep the structure and signs reasonable):

```python
delta_asset = {
    "Recovery": {
        "Equity":      0.05,
        "Bond":       -0.03,
        "Alternative": 0.02,
        "Cash":       -0.04,
    },
    "Overheat": {
        "Equity":     -0.02,
        "Bond":       -0.02,
        "Alternative": 0.04,
        "Cash":        0.00,
    },
    "Stagflation": {
        "Equity":     -0.04,
        "Bond":        0.00,
        "Alternative": 0.03,
        "Cash":       -0.01,
    },
    "Recession": {
        "Equity":     -0.05,
        "Bond":        0.05,
        "Alternative":-0.02,
        "Cash":        0.04,
    },
}
````

This dict should be defined at module level and used by the engine.

==================================================

# 7. Functions to implement (public API)

### 7.1 get_strategy_metadata()

```python
def get_strategy_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per strategy and columns:
        - 'strategy'    : strategy name (string)
        - 'asset_class' : asset class name ("Cash", "Bond", "Equity", "Alternative")
    The order of rows must match the fixed strategy list.
    """
```

This metadata is the central mapping between strategies and asset classes.

---

### 7.2 normalize_weights(w: np.ndarray) -> np.ndarray

```python
def normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D weight vector:
        1) Set negative values to 0
        2) If the sum of weights > 0, divide by the sum so that weights sum to 1
        3) If the sum is 0 (all weights <= 0), return a uniform allocation

    Parameters:
        w : np.ndarray of shape (16,)

    Returns:
        np.ndarray of shape (16,) with non-negative entries summing to 1.
    """
```

This function will be applied row-wise on tentative portfolio weights.

---

### 7.3 compute_raw_strategy_tilts(...)

This is the **core mapping from asset-class tilts to strategy-level tilts**, using SAA-proportional allocation.

```python
def compute_raw_strategy_tilts(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame,
    delta_asset: dict,
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
```

---

### 7.4 compute_final_weights_over_time(...)

High-level function combining raw tilts and normalization.

```python
def compute_final_weights_over_time(
    w_saa: np.ndarray,
    quadrants: pd.Series,
    metadata: pd.DataFrame,
    delta_asset: dict,
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
```

==================================================

# 8. Main demo block

At the bottom of the file, include:

```python
if __name__ == "__main__":
    # 1) Set np.random.seed(42)
    # 2) Construct a synthetic SAA vector w_saa of length 16:
    #       - draw random positive numbers and normalize to sum to 1
    # 3) Construct a synthetic quadrants Series for, say, 120 months:
    #       - e.g., repeat each quadrant for 12 months in some order
    #       - index can be a DatetimeIndex with monthly frequency
    # 4) Call get_strategy_metadata()
    # 5) Use the delta_asset dict defined at module level
    # 6) Call compute_final_weights_over_time(...)
    # 7) Print:
    #       - first 5 rows of final_weights
    #       - a quick check that each row sums to ~1.0
    #       - optionally, a group-by asset_class average weight
    #         (to show how TAA shifts weights at asset-class level)
```

Make the prints human-readable (e.g. round to 4 decimals).

==================================================

# 9. Final output rules

* Return ONLY the final Python code for this single file.
* Do NOT include Markdown fences like ```python.
* Do NOT add any prose explanations outside comments/docstrings.
* Ensure the file can be run directly as:

  python taa_signal_engine.py
