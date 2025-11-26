You are a senior quantitative Python developer.
My goal is to learn Python, NumPy, and pandas by building a TAA (Tactical Asset Allocation) project.

This file is the FIRST core module of the project: `core/mock_data.py`.
It must be fully self-contained and runnable on its own (no project imports required).

==================================================
# 1. File goal

Implement a standalone synthetic data generator that outputs:

- A 16-dim SAA weight vector `w_saa`:
    * 4 asset classes
    * equal-weight (1/K) **inside** each asset class
    * mimics the “420 templates” structure
- Monthly returns for the 16 strategies
- A quadrant path (“Recovery”, “Overheat”, “Stagflation”, “Recession”)

Additionally, when run as a script, the file must save:

- ../data/mock_saa_weights.csv
- ../data/mock_returns.csv
- ../data/mock_quadrants.csv

The path resolution **must rely only on relative path**:
core/mock_data.py  →  ../data/

Therefore, use:
    data_dir = Path(__file__).resolve().parent.parent / "data"

==================================================
# 2. Constraints

- Python 3.9+, must run on Python 3.11
- Allowed:
    - numpy
    - pandas
    - python standard library (pathlib, os, typing)
- Forbidden:
    - scipy, sklearn, statsmodels, torch, tensorflow
- Use `np.random.seed(42)` at module import
- Code must be explicit and beginner-friendly
- Use type hints and docstrings
- Export functions for other modules

==================================================
# 3. Strategy universe (fixed order for the whole project)

```

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

````

The return matrix and SAA all must use this fixed ordering.

==================================================
# 4. Asset-class structure (exact mapping)

4 asset classes:
- "Cash"
- "Bond"
- "Equity"
- "Alternative"

Mapping:

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

Store this mapping inside the file as a dict or DataFrame.

==================================================
# 5. SAA generation rule (must mimic 420 templates)

Implement equal weight **inside each asset class**.

Steps:

1) Define asset-class total weights:

```python
asset_class_weights = {
    "Cash": 0.10,
    "Bond": 0.40,
    "Equity": 0.35,
    "Alternative": 0.15,
}
````

2. Count K_AC = number of strategies in each asset class.

3. For strategies in AC:

```
w_saa[i] = asset_class_weights[AC] / K_AC
```

4. Return a NumPy array of shape (16,), summing to 1.

Implement:

```python
def generate_saa_weights() -> np.ndarray:
    """
    Returns a 16-dim SAA vector with:
        - 4 asset classes
        - 1/K equal-split within each class
        - asset-class weights defined in a dict
    """
```

==================================================

# 6. Monthly returns (synthetic)

Implement:

```python
def generate_monthly_returns(
    n_months: int = 120,
    n_strategies: int = 16
) -> pd.DataFrame:
    """
    Generates monthly returns (decimal) for the 16 strategies.
    index    = monthly DatetimeIndex ("2015-01-01", freq="MS")
    columns  = strategy names in fixed order
    values   = Gaussian random returns with realistic means & vols
    """
```

Suggested:

* Cash: low mean, low vol
* Bond: medium mean, low vol
* Equity: higher mean, higher vol
* Alternative: medium/high vol

==================================================

# 7. Quadrant path (synthetic)

Implement:

```python
def generate_quadrant_path(
    n_months: int = 120
) -> pd.Series:
    """
    Generates macro quadrants for each month:
    ['Recovery','Overheat','Stagflation','Recession'].

    The index must match generate_monthly_returns().
    You can use:
        - patterns (12 months per quadrant), or
        - np.random.choice
    """
```

==================================================

# 8. High-level dataset builder

Implement:

```python
from typing import Tuple

def create_mock_dataset(
    n_months: int = 120
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Returns:
        w_saa      : shape (16,)
        returns_df : (n_months × 16) DataFrame
        quadrants  : (n_months,) Series
    """
```

==================================================

# 9. Saving to ../data/ (RELATIVE PATH, must be correct)

IMPORTANT:

`mock_data.py` resides in:

```
core/mock_data.py
```

`data/` is:

```
../data/
```

Therefore use:

```python
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
```

Then save:

* `mock_saa_weights.csv`
  index = strategy names
  column = "weight"

* `mock_returns.csv`
  index = dates
  columns = strategy names

* `mock_quadrants.csv`
  index = dates
  column = "quadrant"

==================================================

# 10. Main demo block

At bottom:

```python
if __name__ == "__main__":
    # 1) Create dataset
    # 2) Print human-readable summary
    # 3) Save all three CSV files into ../data/
    # 4) Print absolute paths of saved files.
```

==================================================

# 11. Final output rules

* Return ONLY the Python code for mock_data.py
* No explanation outside comments/docstrings
* No markdown fences
* File must run as:

  python core/mock_data.py

from ANY directory, thanks to relative path usage.