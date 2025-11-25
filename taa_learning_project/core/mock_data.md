You are a senior quantitative Python developer.
My goal is to learn Python, NumPy, and pandas by writing all key modules of a Tactical Asset Allocation (TAA) project.  
This file is the first step, and must be fully self-contained and easy to read.

==================================================
# 1. File goal (mock_data.py)

Generate **synthetic** TAA dataset inputs needed by the project:

- a 16-dim SAA weight vector  
- a (T × 16) DataFrame of synthetic monthly returns  
- a (T × 1) Series of macro quadrants  

This file focuses on:
- NumPy vector operations  
- pandas DataFrame construction  
- reproducible random data generation  
- clear comments & docstrings  

The output will be consumed by other core modules.

==================================================
# 2. Project constraints (must follow)

- Target Python version: **3.9** (should also run on 3.11)
- Allowed libraries: **numpy**, **pandas**, **python stdlib only**
- No sklearn, no scipy, no statsmodels, no torch, no tensorflow
- Must be a **single Python file running standalone**
- Use `np.random.seed(42)` for reproducibility
- Make code extremely readable for a beginner learning NumPy/pandas
- Showcase best practices: type hints, docstrings, simple functions

==================================================
# 3. Outputs (interface contract — MUST follow)

This file must expose:

1) `generate_saa_weights(n_strategies: int = 16) -> np.ndarray`
   - shape (16,)
   - non-negative, sum to 1

2) `generate_monthly_returns(n_months: int = 120, n_strategies: int = 16) -> pd.DataFrame`
   - shape (T, 16)
   - columns must be strategy names
   - values are decimal returns (e.g., 0.01 for 1%)

3) `generate_quadrant_path(n_months: int = 120) -> pd.Series`
   - 4 quadrants as strings:
     ["Recovery", "Overheat", "Stagflation", "Recession"]

4) `create_mock_dataset(n_months: int = 120)`
   - return `(w_saa, returns_df, quadrants)`

5) A runnable `if __name__ == "__main__":` demo that:
   - prints SAA weights (strategy + weight)
   - prints head/tail of returns_df
   - prints quadrant distribution

==================================================
# 4. Strategy names (fixed list to ensure consistency)

Use the following 16 strategy names in order:

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

These must become the DataFrame columns for returns_df.

==================================================
# 5. Implementation details (must follow)

### (1) SAA generation
- Draw random positive numbers
- Normalize to sum to 1
- Return as np.ndarray (length 16)

### (2) Synthetic returns
- Use simple Gaussian random generation
- You may give each asset a different mean/vol for realism, but keep code simple
- Use: mean in [0.001, 0.01], vol in [0.01, 0.05]
- Convert to pandas DataFrame

### (3) Quadrant path
Simple models accepted:
- random choice with equal probability  
OR
- block structure (e.g., each quadrant lasts 12 months)

Return pandas Series with monthly index.

### (4) create_mock_dataset()
Pack everything cleanly into a tuple.

### (5) Main demo
Make the outputs very readable.
Show real NumPy/pandas usage (Series, DataFrame, indexing).

==================================================
# 6. Final output rules

- Return **only** the final Python code (no markdown fences)
- No extra explanations outside comments/docstrings
- The file must run successfully via:
  
  python mock_data.py
