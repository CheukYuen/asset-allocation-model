先直接给你「怎么写单文件 Prompt」这件事的答案，然后顺带把项目拆分和示例都给出来，你可以直接 copy 到 Cursor 用。

---

## 一、先把项目脑中模型定一下（很简短）

根据 TAA 市场信号模块 PRD，这个小项目可以拆成几类单文件脚本：

1. **mock_data_saa_and_returns.py**

   * 生成 16 维 SAA 权重、16 个子策略的虚拟月度收益、象限路径。

2. **taa_signal_engine.py**

   * 输入：SAA 权重、象限序列
   * 输出：每月的 Δw_strategy,t 和 w_final,t
   * 内含：大类偏移映射、SAA 分摊到 16 子策略、β 调整、Normalize。

3. **backtest_compare_saa_vs_taa.py**

   * 用真实或虚拟 r_t，计算
   * r_SAA,t、r_final,t，年化收益、波动、夏普、MDD + 判优。

4. **mc_simulation_30y.py**

   * 用 CME 的 μ、Σ（先用虚拟 μ、Σ）
   * 做 30 年 Monte Carlo，跑出终值中位数、区间、下行风险、胜率。

你现在要做的，就是给 Cursor 一次只描述好「其中一个文件要干什么」，让它写出**可单独运行的脚本**，用虚拟数据、只用 `numpy` 和 `pandas`（不依赖 `scipy`）。

---

## 二、通用「单文件 Prompt 模板」

下面这个模板是**英文版**，你可以每次复制一份，改掉「File goal / Inputs / Outputs / Functions」那几块内容，把它贴给 Cursor 让它生成一个 `.py` 文件。

> 建议：把「Python 3.9」「only use numpy & pandas」这些通用约束都固定在模板里。

---

### 🔧 Single Python File Prompt Template（可直接复用）

````text
You are a senior quantitative Python developer.

I am learning Python, NumPy, and pandas by building a small Tactical Asset Allocation (TAA) project based on an internal PRD. For this step, I want you to generate ONE SINGLE self-contained Python file.

## 1. File goal

- Implement a small, focused module for the TAA project.
- The file must be runnable on its own: `python this_file.py` should work without any external files.
- Use **only** built-in Python standard library, `numpy`, and `pandas`. Do NOT use `scipy` or any other third-party libraries.

## 2. Runtime & style constraints

- Target Python version: **3.9** (compatible with 3.11 locally).
- Use type hints where reasonable.
- Add clear docstrings and comments explaining the math and data transformations, so that a beginner can learn from the code.
- Prefer simple, explicit code over clever one-liners.

## 3. Data source

- For now, **use synthetic data only**.
- If the module needs:
  - SAA weights: generate example 16-dimensional vectors that sum to 1.
  - Monthly returns: generate NumPy arrays or pandas DataFrames using random draws.
  - Quadrant paths: generate a pandas Series of integers or strings representing the 4 macro quadrants.
- Always set a random seed at the top of the file, e.g. `np.random.seed(42)`, to keep results reproducible.

## 4. What this file must do (module-specific spec)

[!!! REPLACE THIS BLOCK FOR EACH FILE YOU WANT TO BUILD !!!]

Describe in detail:
- The main goal of this file.
- What inputs are expected (shape, meaning, units).
- What outputs it should produce.
- Which formulas from the PRD it should implement.
- What helper functions it should expose (with names and signatures).
- What should be demonstrated in the `if __name__ == "__main__":` section.

## 5. Structure and API

- Organize the code into small functions.
- At the bottom, include:

```python
if __name__ == "__main__":
    # 1) construct synthetic inputs
    # 2) call the main functions
    # 3) print key results in a readable way
````

* Ensure the script can be run standalone without any external configuration.

## 6. Output format

* Return only the final Python code for this single file.
* Do NOT include Markdown fences like ```python.
* Do NOT explain the code in prose; use comments and docstrings inside the code instead.

````

你可以把这段当成「壳」，每次只改第 4 部分。

---

## 三、基于 PRD 的三个具体单文件 Prompt 示范

下面我直接帮你把「第 4 部分」填好，变成**三份可直接丢给 Cursor 的完整 Prompt**，分别对应：

1. 虚拟数据 + SAA 权重生成文件  
2. TAA Δw + w_final 计算引擎  
3. 回测指标与判优

你可以先从 1 → 2 → 3 这样练。

---

### 1️⃣ 文件一：mock_data_saa_and_returns.py

**用途**：先有一份能生成「SAA 权重 + 子策略月度收益 + 象限序列」的脚本，后面所有文件都可以直接 import 里边的函数或 copy 逻辑。

完整 Prompt：

```text
You are a senior quantitative Python developer.

I am learning Python, NumPy, and pandas by building a small Tactical Asset Allocation (TAA) project based on an internal PRD. For this step, I want you to generate ONE SINGLE self-contained Python file.

## 1. File goal

- File name (suggested): `mock_data_saa_and_returns.py`.
- Goal: generate synthetic data for:
  - A 16-dimensional SAA weight vector `w_saa` (sum to 1).
  - Monthly returns for 16 sub-strategies over T months.
  - A monthly macro quadrant path over the same T months.

## 2. Runtime & style constraints

- Target Python version: **3.9** (compatible with 3.11 locally).
- Use **only**: Python standard library, `numpy`, `pandas`.
- Do NOT use `scipy` or any other third-party packages.
- Use type hints where reasonable.
- Add docstrings and comments so that a beginner can learn NumPy and pandas from this file.
- Keep the code explicit and easy to read.

## 3. Data source

- Use **synthetic data only**.
- Set `np.random.seed(42)` at the top for reproducibility.

## 4. What this file must do (module-specific spec)

1. Define constants for the 16 strategies.
   - Use a simple list of strategy names like:
     - "Cash", "DepositFixedIncome", "PureBond", "NonStandardFI", "FixedIncomePlus",
       "OverseasBond", "BalancedFund", "EquityA", "EquityOverseas",
       "OverseasBalanced", "CommodityMacro", "QuantHedge",
       "RealEstateEquity", "PE_VC", "OverseasAlternative", "StructuredProduct".
   - Keep these names in a Python list or tuple.

2. Generate a synthetic SAA weight vector `w_saa`:
   - Shape: `(16,)`.
   - All elements >= 0.
   - Sum to 1.0 (use normalization).
   - Provide a function:

   ```python
   def generate_saa_weights(n_strategies: int = 16) -> np.ndarray:
       ...
````

3. Generate synthetic monthly returns:

   * Provide a function:

   ```python
   def generate_monthly_returns(
       n_months: int = 120,
       n_strategies: int = 16
   ) -> pd.DataFrame:
       """
       Returns a DataFrame of shape (n_months, n_strategies)
       with columns equal to the strategy names and values
       representing monthly returns in decimal form (e.g., 0.01 for 1%).
       """
   ```

   * Use a simple normal distribution per strategy (e.g. `np.random.normal`).
   * You can give different mean/vol parameters for rough realism (e.g. cash small mean & low vol, equity higher mean & higher vol).

4. Generate a synthetic macro quadrant path:

   * There are 4 quadrants: "Recovery", "Overheat", "Stagflation", "Recession".
   * Provide a function:

   ```python
   def generate_quadrant_path(n_months: int = 120) -> pd.Series:
       """
       Returns a pandas Series indexed the same way as the returns DataFrame index.
       Each value is one of the four quadrants as a string.
       """
   ```

   * Start with a very simple model: randomly draw quadrants with equal probability, or in blocks (e.g. each quadrant lasts 12 months).

5. Provide a convenience function that bundles everything:

   ```python
   def create_mock_dataset(
       n_months: int = 120
   ) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
       """
       Returns (w_saa, returns_df, quadrant_series).
       """
   ```

6. Main section:

   Include an `if __name__ == "__main__":` block that:

   * Calls `create_mock_dataset` for, say, 120 months (10 years).
   * Prints:

     * The SAA weights as a table (strategy name + weight).
     * The head/tail of the returns DataFrame.
     * The value counts of each quadrant.

## 5. Structure and API

* Organize the code into small functions.
* Use clear naming.
* Do not read or write any external files; everything stays in memory.

## 6. Output format

* Return only the final Python code for this single file.
* Do NOT include Markdown fences like ```python.
* Do NOT explain the code in prose; use comments and docstrings inside the code instead.

````

---

### 2️⃣ 文件二：taa_signal_engine.py（Δw + w_final）

这个文件练的是：矩阵运算、分组归一化、映射逻辑，直接对应 PRD 中的公式：  
Δw_asset,t → Δw_strategy^(0) → 乘 β → Normalize → w_final,t:contentReference[oaicite:1]{index=1}

完整 Prompt：

```text
You are a senior quantitative Python developer.

I am learning Python, NumPy, and pandas by building a small Tactical Asset Allocation (TAA) project based on an internal PRD. For this step, I want you to generate ONE SINGLE self-contained Python file.

## 1. File goal

- File name (suggested): `taa_signal_engine.py`.
- Goal: implement the TAA weight-adjustment engine that:
  - Reads SAA weights for 16 strategies.
  - Applies macro-quadrant-based asset-class tilts (Δw_asset,t).
  - Maps tilts down to 16 strategies using SAA weights and sensitivity coefficients β.
  - Produces final monthly weights `w_final,t` for each of 16 strategies.

We assume the 16 strategies are already mapped to 5 asset classes:
- Equity, Bond, Commodity, Gold, Cash.

## 2. Runtime & style constraints

- Target Python version: **3.9**.
- Use only: Python standard library, `numpy`, `pandas`.
- Do NOT use `scipy` or any other third-party packages.
- Use type hints and docstrings.
- Add comments explaining each step and formula, suitable for a beginner learning NumPy and pandas.

## 3. Data source

- For now, use synthetic or simple hard-coded data:
  - Accept SAA weights as a NumPy array of shape `(16,)`.
  - Accept a pandas Series of quadrants over time.
- You may optionally import and call functions from a hypothetical `mock_data_saa_and_returns` module, but the file must also work if the user just constructs arrays manually in the `__main__` block.

## 4. What this file must do (module-specific spec)

1. Define the 16 strategies, their asset-class mapping, and β-sensitivity coefficients.

   - Use a pandas DataFrame or a simple Python dict/list to encode:

     - `strategy_name`
     - `asset_class` (e.g., "Equity", "Bond", "Commodity", "Gold", "Cash")
     - `beta` (float)

   - Use β values consistent with the PRD example (can be in code as a constant mapping).

2. Define the quadrant → asset-class tilt matrix.

   - Hard-code a mapping that matches the PRD’s idea:

     - Quadrants: "Recovery", "Overheat", "Stagflation", "Recession".
     - Asset classes: "Equity", "Bond", "Commodity", "Gold", "Cash".

   - Represent this as a pandas DataFrame or dictionary of dictionaries, e.g.:

     ```python
     delta_asset = {
         "Recovery": {"Equity": 0.05, "Bond": -0.03, "Commodity": 0.0, "Gold": -0.02, "Cash": 0.0},
         ...
     }
     ```

   - These are per-month tilts Δw_asset,t.

3. Implement a function to compute raw strategy tilts Δw_strategy^(0):

   ```python
   def compute_raw_strategy_tilts(
       w_saa: np.ndarray,
       quadrants: pd.Series,
   ) -> pd.DataFrame:
       """
       For each month t, given the quadrant, compute Δw_strategy^(0) for all 16 strategies
       by distributing Δw_asset,t within each asset class proportional to SAA weights.
       Returns a DataFrame of shape (n_months, 16).
       """
````

* For each asset class AC and each month t:

  Δw_strategy^(0)*{i,t} = Δw_asset,t(AC) * w_saa[i] / sum*{j in AC} w_saa[j]

* Be careful with division by zero (if sum of SAA weights in an asset class is zero).

4. Apply β sensitivity:

   ```python
   def apply_beta_adjustment(
       delta_w_raw: pd.DataFrame,
       betas: np.ndarray
   ) -> pd.DataFrame:
       """
       Δw_strategy = β_i * Δw_strategy^(0)_i,t
       """
   ```

   * Here `betas` is a 1D NumPy array of length 16 aligned with strategies.

5. Implement a Normalize function that works on each row of weights:

   ```python
   def normalize_weights(
       w: np.ndarray
   ) -> np.ndarray:
       """
       Given a 1D array of tentative weights w, apply:
           1) Set negative values to 0
           2) Renormalize so that sum = 1
       If all values are <= 0, fall back to a uniform allocation.
       """
   ```

6. Combine everything into a function to get final weights over time:

   ```python
   def compute_final_weights_over_time(
       w_saa: np.ndarray,
       quadrants: pd.Series
   ) -> pd.DataFrame:
       """
       Returns a DataFrame of shape (n_months, 16) with final weights w_final,t for each month t.
       Uses:
           w_final,t = Normalize(w_saa + Δw_strategy,t)
       """
   ```

7. Main section (`if __name__ == "__main__":`):

   * Generate a mock `w_saa` (or import from mock_data module).
   * Generate a mock quadrant path for, say, 120 months.
   * Call `compute_final_weights_over_time`.
   * Print:

     * First few rows of final weights.
     * A simple check that each row sums to 1.
     * Optionally, the average tilt per strategy over time.

## 5. Structure and API

* Organize the code into:

  * constants / metadata (strategy list, betas, asset-class mapping),
  * pure functions (compute_raw_strategy_tilts, apply_beta_adjustment, normalize_weights, compute_final_weights_over_time),
  * main demo section.

## 6. Output format

* Return only the final Python code for this single file.
* Do NOT include Markdown fences like ```python.
* Do NOT explain the code in prose; use comments and docstrings inside the code instead.

````

---

### 3️⃣ 文件三：backtest_compare_saa_vs_taa.py（回测 + 判优）

这个文件对应 PRD 里的：  
- r_SAA,t = w_SAAᵀ r_t  
- r_final,t = w_final,tᵀ r_t  
- 年化收益 μ、年化波动 σ、Sharpe、MDD、判优条件:contentReference[oaicite:2]{index=2}

完整 Prompt：

```text
You are a senior quantitative Python developer.

I am learning Python, NumPy, and pandas by building a small Tactical Asset Allocation (TAA) project based on an internal PRD. For this step, I want you to generate ONE SINGLE self-contained Python file.

## 1. File goal

- File name (suggested): `backtest_compare_saa_vs_taa.py`.
- Goal: implement a simple backtest that compares:
  - A fixed SAA portfolio with weights w_SAA
  - A TAA-adjusted portfolio with time-varying weights w_final,t
- Compute performance metrics and a simple “is TAA better?” decision.

## 2. Runtime & style constraints

- Target Python version: **3.9**.
- Use only: Python standard library, `numpy`, `pandas`.
- Do NOT use `scipy` or other third-party packages.
- Use type hints and docstrings.
- Add comments teaching a beginner how the metrics are computed.

## 3. Data source

- Use synthetic data or import from other modules (mock_data and taa_signal_engine) conceptually.
- For robustness, the file must also be able to run completely standalone by generating its own mock data in the `__main__` block.

## 4. What this file must do (module-specific spec)

1. Define metric functions:

   - Portfolio returns:

     ```python
     def compute_portfolio_returns(
         weights: pd.DataFrame,
         returns: pd.DataFrame
     ) -> pd.Series:
         """
         weights: (n_months, n_strategies), each row sums to 1
         returns: (n_months, n_strategies), monthly returns in decimal
         returns a Series of length n_months with portfolio returns.
         """
     ```

     - For SAA (constant weights), you can either:
       - Broadcast a 1D array to all months, or
       - Build a constant-weight DataFrame.

   - Annualized return:

     ```python
     def annualized_return(monthly_returns: pd.Series) -> float:
         """
         μ = 12 * mean(monthly_returns)
         """
     ```

   - Annualized volatility:

     ```python
     def annualized_volatility(monthly_returns: pd.Series) -> float:
         """
         σ = sqrt(12) * std(monthly_returns, ddof=1)
         """
     ```

   - Sharpe ratio with a constant monthly risk-free rate:

     ```python
     def sharpe_ratio(
         monthly_returns: pd.Series,
         rf_monthly: float = 0.0
     ) -> float:
         """
         Excess returns r_excess = monthly_returns - rf_monthly
         Sharpe = (12 * mean(r_excess)) / (sqrt(12) * std(r_excess))
         Handle the case where std is very close to 0.
         """
     ```

   - Maximum drawdown (MDD):

     ```python
     def max_drawdown(monthly_returns: pd.Series) -> float:
         """
         Compute equity curve as cumulative product of (1 + r_t),
         then compute maximum drawdown in percentage terms.
         """
     ```

2. Define a simple decision function that implements the PRD-style rules:

   ```python
   @dataclass
   class BacktestResult:
       annual_return: float
       annual_vol: float
       sharpe: float
       mdd: float

   @dataclass
   class ComparisonResult:
       saa: BacktestResult
       taa: BacktestResult
       is_taa_better: bool
````

* And a function:

```python
def compare_saa_vs_taa(
    saa_returns: pd.Series,
    taa_returns: pd.Series
) -> ComparisonResult:
    """
    Implements a simple decision rule, for example:
        1) TAA annual_return >= SAA annual_return
        2) TAA annual_vol <= SAA annual_vol + 0.01
        3) TAA mdd <= SAA mdd + 0.02
        4) TAA sharpe >= SAA sharpe
    """
```

3. Main section (`if __name__ == "__main__":`):

   * Generate or import:

     * A 16-dim SAA weight vector `w_saa`.
     * A DataFrame of monthly strategy returns (e.g. 120 months).
     * A monthly quadrant path (if you also recreate TAA weights here).
   * Create:

     * A constant SAA weight DataFrame for all months.
     * A time-varying TAA weight DataFrame per month (either:

       * call a simplified version of the TAA engine, or
       * generate synthetic but different weights from SAA for demonstration).
   * Compute:

     * `saa_portfolio_returns`
     * `taa_portfolio_returns`
   * Pass both into `compare_saa_vs_taa`.
   * Print:

     * A small summary table with metrics for SAA vs TAA.
     * A clear line: `"TAA better according to rules: True/False"`.

## 5. Structure and API

* Keep functions pure and testable.
* Separate data-generation logic from metric computations as much as possible.

## 6. Output format

* Return only the final Python code for this single file.
* Do NOT include Markdown fences like ```python.
* Do NOT explain the code in prose; use comments and docstrings inside the code instead.

```

---

## 四、你可以怎么用这些 Prompt

一个简单的实践路径（也顺便练 Python / NumPy / pandas）：

1. 在 Cursor 里新建 repo，比如 `taa_learning_project/`。
2. 先用「文件一 Prompt」让它生成 `mock_data_saa_and_returns.py`。  
   - 自己跑一遍，改一点参数，看看 DataFrame / Series 结构。
3. 再用「文件二 Prompt」生成 `taa_signal_engine.py`。  
   - 在 `__main__` 里从前一个文件导入 mock 数据（或者先 copy paste）。
4. 然后用「文件三 Prompt」生成 `backtest_compare_saa_vs_taa.py`。  
   - 自己多改几次判优规则，看看指标变化。

如果你愿意，下一步我可以再帮你写第 4 个文件（Monte Carlo 模拟那块）的专用 Prompt，或者帮你把这几个文件的「互相 import」关系也规范一下。
```
