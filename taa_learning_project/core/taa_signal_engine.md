You are a senior quantitative Python developer.
My goal is to learn Python, NumPy, and pandas by building a Tactical Asset Allocation (TAA) engine step by step.

This file is the second core module of the project, after `mock_data.py`.  
It must be fully self-contained and runnable on its own (no project imports required).

==================================================
# 1. File goal (taa_signal_engine.py)

Implement the TAA weight-adjustment engine:

- Inputs:
  - A 16-dimensional SAA weight vector `w_saa`
  - A pandas Series of macro quadrants, one per month

- Logic:
  - Map each of the 16 strategies to one of **four** asset classes:
        - Cash
        - Bond
        - Equity
        - Alternative
  - For each quadrant, obtain an asset-class tilt Δw_asset,t
  - Allocate tilts to individual strategies proportional to SAA weights
  - Apply sensitivity coefficients β per strategy
  - Normalize to obtain final monthly weights (`w_final,t`)

- Output:
  - A `(T × 16)` DataFrame of final weights

This module is focused on:
- vectorized operations
- group-based tilt allocation
- clean implementation of TAA math
- readable NumPy/pandas code

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
    - No import from project modules
    - Use synthetic data in main demo
- Use `np.random.seed(42)` in demo
- Use type hints and docstrings
- Code must be beginner-friendly

==================================================
# 3. 16 strategies (fixed)

Use these strategy names in **this exact order**:

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

==================================================
# 4. Asset-class mapping (MUST follow 4 大类)

Map each strategy to one of the **four** asset classes:

- Cash
- Bond
- Equity
- Alternative

Mapping to implement:

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

Store this metadata inside the file (e.g., pandas DataFrame).

==================================================
# 5. Macro-quadrant definition

Quadrants (PRD-consistent):

- "Recovery"
- "Overheat"
- "Stagflation"
- "Recession"

==================================================
# 6. Asset-class tilt matrix (Δw_asset,t)

Because we now have **4 asset classes**, define tilts ONLY for:

- Cash
- Bond
- Equity
- Alternative

Example (you may adjust numbers):

delta_asset = {
    "Recovery": {
        "Equity":      0.05,
        "Bond":       -0.03,
        "Alternative": 0.02,
        "Cash":       -0.04
    },
    "Overheat": {
        "Equity":     -0.02,
        "Bond":       -0.02,
        "Alternative": 0.04,
        "Cash":        0.00
    },
    "Stagflation": {
        "Equity":     -0.04,
        "Bond":        0.00,
        "Alternative": 0.03,
        "Cash":       -0.01
    },
    "Recession": {
        "Equity":     -0.05,
        "Bond":        0.05,
        "Alternative":-0.02,
        "Cash":        0.04
    }
}

No Commodity tilt — Commodity is treated under Alternative.

==================================================
# 7. Sensitivity coefficients β

Define a NumPy array of length 16.
General rule:

- Cash strategies: low beta (0.3–0.5)
- Bonds: medium beta (0.7–0.9)
- Equity: high beta (1.0–1.2)
- Alternative: medium-high (0.9–1.2)

Example:

[0.3, 0.7, 0.8, 0.8, 0.9, 0.9,
 1.0, 1.1, 1.1, 1.0,
 1.0,   # Commodity (Alternative)
 1.0, 1.1, 1.2, 1.1, 1.0]

==================================================
# 8. Required functions (public API)

1) get_strategy_metadata()

Returns DataFrame with:
- strategy
- asset_class
- beta

2) normalize_weights(w: np.ndarray) -> np.ndarray

Rules:
- Set negatives → 0
- If sum>0: normalize to 1
- If sum==0: return uniform allocation

3) compute_raw_strategy_tilts(...)

Formula per asset-class AC:
Δw_strategy^(0)_{i,t} =
    Δw_asset,t(AC) * w_saa[i] / sum_{j∈AC}(w_saa[j])

Return DataFrame (T × 16).

4) apply_beta_adjustment(...)

Multiply each strategy’s tilt by its β.

5) compute_final_weights_over_time(...)

Steps:
1. raw tilts
2. beta-adjusted tilts
3. tentative weights = w_saa + Δw_strategy
4. normalize per row

Return DataFrame (T × 16).

==================================================
# 9. Main demo block

In `if __name__ == "__main__":`:

- Set seed
- Create synthetic SAA (sum to 1)
- Create synthetic quadrants for 120 months
- Load metadata
- Compute final weights
- Print:
  - first few rows
  - row sums
  - simple sanity checks

==================================================
# 10. Final output rules

- Return ONLY final Python code
- No ``` fences
- No extra prose outside comments/docstrings
- Must run as:

  python taa_signal_engine.py
