---

# **AGENTS.md（最终版）：TAA Learning Project Global Rules**

本项目用于练习 Python + NumPy + pandas，通过实现 TAA（战术资产配置）计算链路来学习数据结构、向量化运算与工程化规范。
以下所有规则为本仓库的全局约束，所有自动生成的文件都必须遵循。

---

# **1｜项目目录结构（Project Structure）**

```
taa_learning_project/
│
├── data/                       # (optional) 存放 mock 或未来真实数据
│   ├── mock_returns.csv
│   ├── mock_quadrants.csv
│   └── mock_saa_weights.csv
│
├── core/                       # 核心数学逻辑（核心模块）
│   ├── mock_data.py            # 生成 SAA / 收益率 / 象限 → synthetic 数据
│   ├── utils.py                # Normalize / seed / 通用工具
│   ├── taa_signal_engine.py    # Δw + β + Normalize → w_final,t
│   ├── backtest_engine.py      # 回测指标 + 判优
│   └── mc_simulation.py        # 多期蒙特卡罗模拟（未来扩展）
│
├── scripts/                    # 可执行脚本，调用 core 模块
│   ├── run_mock_data.py
│   ├── run_taa_signal.py
│   ├── run_backtest.py
│   └── run_mc_simulation.py
│
└── AGENTS.md                   # 全局项目规范（本文件）
```

目标：

* `core/` 只放数学与逻辑；
* `scripts/` 执行单次任务；
* `data/` 可选用于缓存。

---

# **2｜跨文件 import 关系（Import Dependency Rules）**

这一节是本项目**最关键的工程规范**：
避免循环依赖、保证层次清晰、让 Cursor 自动生成的代码不会乱。

---

## **2.1 底层（无依赖层）**

### **mock_data.py**

只允许：

```python
import numpy as np
import pandas as pd
```

不得 import core 中其他模块。

---

### **utils.py**

只允许：

```python
import numpy as np
```

不得 import mock_data 或其他文件。

---

## **2.2 中层（核心逻辑层）**

### **taa_signal_engine.py**

允许：

```python
import numpy as np
import pandas as pd
from core.utils import normalize_weights
```

不推荐 import mock_data（避免强耦合）。
测试数据应在 `__main__` 中调用 mock_data 生成。

---

### **backtest_engine.py**

允许：

```python
import numpy as np
import pandas as pd
from core.utils import normalize_weights
from core.taa_signal_engine import compute_final_weights_over_time
```

不允许 import mock_data（保持独立）。

---

## **2.3 脚本层（最高层）**

`scripts/*.py` 允许 import core 中所有模块，例如：

```python
from core.mock_data import create_mock_dataset
from core.taa_signal_engine import compute_final_weights_over_time
from core.backtest_engine import compare_saa_vs_taa
```

脚本层不得被 core 层 import。

---

## **2.4 整体依赖图（Dependency Graph）**

```
mock_data.py         utils.py
      ↓                 ↓
      ↓                 ↓
   taa_signal_engine.py
            ↓
     backtest_engine.py
            ↓
         scripts/*.py
```

**禁止反向 import**（下层永不 import 上层）。

---

# **3｜统一 Outputs 规范（Output Interfaces）**

为保证核心文件之间接口一致性，Cursor 生成代码时必须使用以下统一输出格式。

---

## **3.1 mock_data 输出**

函数：`create_mock_dataset()` 返回：

```python
w_saa: np.ndarray        # shape (16,)
returns_df: pd.DataFrame # shape (T, 16)
quadrants: pd.Series     # shape (T,)
```

CSV 统一格式（可选）：

```
mock_returns.csv      # 16 列，每列一个子策略
mock_quadrants.csv    # 1 列：quadrant
mock_saa_weights.csv  # 1 列：weight
```

---

## **3.2 taa_signal_engine 输出**

核心函数：`compute_final_weights_over_time()` 返回：

```python
final_weights_df: pd.DataFrame  # shape (T, 16)
# 每行 sum=1，已经 Normalize
```

列名必须与 `returns_df` 对齐。

---

## **3.3 backtest_engine 输出**

定义以下数据类：

```python
BacktestResult:
    annual_return: float
    annual_vol: float
    sharpe: float
    mdd: float

ComparisonResult:
    saa: BacktestResult
    taa: BacktestResult
    is_taa_better: bool
```

核心函数：`compare_saa_vs_taa()` 返回：

```python
ComparisonResult
```

---

## **3.4 Monte-Carlo（mc_simulation）输出**

数据类：

```python
MCResult:
    median: float
    p5: float
    p95: float
    worst_5pct: float
    all_paths: np.ndarray   # optional, shape (N_paths, T)
```

核心函数返回：

```python
MCResult
```

---

# **4｜全局编程规范（Coding Standards）**

所有 core 与 scripts 文件必须遵循本节规则。

---

## **4.1 Python 版本与依赖**

* **Python 3.9（兼容 3.11）**
* 只允许：

```
numpy
pandas
```

禁止：

```
scipy, sklearn, statsmodels, numba, pytorch, tf …
```

---

## **4.2 风格（Cursor 容易遵守的规范）**

* 函数必须带 docstring（用途、参数类型、返回类型）
* 尽量使用 typed signatures (`np.ndarray`, `pd.DataFrame`)
* 保持显式、清晰，不写炫技 one-liner
* 每个核心模块都要实现 `if __name__ == "__main__":`（用于本地练习）

---

## **4.3 数据要求**

测试数据必须使用：

```python
np.random.seed(42)
```

以确保结果可复现。

---

# **5｜数据加工成信号的流程（以 PMI 为例）**

本节说明：**原始宏观数据如何一步步加工成美林时钟的象限信号**，供周期判定与 TAA 使用。  
本项目中 **象限（quadrants）** 由上游模块或 mock 给出，TAA 引擎只消费 `quadrants`，不直接读 PMI/CPI 等原始指标。此处文档化“若从零实现指标→象限”时应遵循的加工链路，并以 **PMI（制造业 PMI）** 为例逐步展开。

---

## **5.1 全链路概览**

```
原始数据（发布值）
    → 结构化指标（indicator_id, ref_month, value, unit, data_type）
    → 单指标规则（如 PMI ≥50 / 3 月趋势）
    → 增长/通胀维度状态（Growth 扩张 or 收缩；Inflation 升 or 降）
    → 多指标综合 + 最短判断长度（3～6 月）
    → 象限（Recovery / Overheat / Stagflation / Recession）
    → TAA 权重（由 taa_signal_engine 根据象限计算）
```

---

## **5.2 第一步：原始数据 → 结构化指标**

- **原始形态**：例如统计局发布的「2024 年 11 月制造业 PMI 50.3」。
- **项目内建议结构**（便于程序做周期判定与回测）：

| 字段 | 含义 | PMI 示例 |
|------|------|----------|
| `indicator_id` | 指标唯一标识 | `PMI_MANU` |
| `ref_month` | 统计月份 | `2024-11` |
| `value` | 数值 | `50.3` |
| `unit` | 单位（可选） | 空（Level） |
| `adjustment` | 调整方式 | `Level`（水平值） |
| `source` | 来源 | `NBS` |
| `data_type` | 用于筛选的维度 | `Growth`（经济增长类） |

- **用途**：`data_type` 为 Growth / Inflation / Credit / Rate，便于按「增长 vs 通胀」自动筛选，供后续规则使用。

---

## **5.3 第二步：单指标规则（以 PMI 为例）**

- **业务含义**：PMI ≥50 表示制造业环比扩张，&lt;50 表示收缩；单月易受季节与噪音影响。
- **稳健做法**（与「最短判断长度」一致）：**至少 3 个月** 连续/趋势一致再确认。

**PMI 加工成“增长方向”信号的步骤：**

1. **取最近 3 个月**  
   对当前判定月 \(t\)，取 `ref_month` 为 \(t, t-1, t-2\) 的 PMI_MANU 记录，得到三个 `value`：\(v_t, v_{t-1}, v_{t-2}\)。

2. **水平条件**  
   - 若 \(v_t \geq 50\) 且 \(v_{t-1} \geq 50\) 且 \(v_{t-2} \geq 50\) → 满足「连续 3 个月 ≥50」。
   - 否则不视为**稳健扩张**（可记为收缩或中性，依规则定义）。

3. **趋势条件（可选加强）**  
   - 3 月均值：\(\bar{v}_3 = (v_t + v_{t-1} + v_{t-2})/3\)。  
   - 上季度均值：例如 \(t-3,t-4,t-5\) 的均值 \(\bar{v}_{prev}\)。  
   - 若要求「3 月均值高于上季度」：\(\bar{v}_3 > \bar{v}_{prev}\) → 扩张趋势在加强，再输出**增长扩张**信号。

4. **输出（单指标）**  
   - 满足上述条件 → **Growth 维度**：扩张（Up）。  
   - 不满足 → 收缩（Down）或中性（依产品规则）。

**小结**：  
- **输入**：结构化 PMI 序列（按 `ref_month` 排序）。  
- **输出**：当前月 \(t\) 的「增长」方向信号（扩张/收缩/中性）。  
- **时间长度**：至少 3 个月数据，才产生一次稳健信号。

---

## **5.4 第三步：多指标综合 → 象限**

- 美林时钟用 **增长（Growth）** 与 **通胀（Inflation）** 两个维度划分四象限：

| 象限 | 增长 | 通胀 |
|------|------|------|
| Recovery（复苏） | ↑ 扩张 | ↓ 偏低/降 |
| Overheat（过热） | ↑ 扩张 | ↑ 偏高/升 |
| Stagflation（滞胀） | ↓ 收缩 | ↑ 偏高/升 |
| Recession（衰退） | ↓ 收缩 | ↓ 偏低/降 |

- **PMI** 只贡献 **Growth** 维度；**Inflation** 需用 CPI_YOY、PPI_YOY 等，并同样做**趋势与最短长度**（如 6 个月 MA 或斜率）。
- **综合方式**：  
  - 增长类：PMI_MANU、PMI_SERV、INDUSTRY_YOY、TSF_STOCK_YOY 等，按 3/6 个月规则得到 Growth 状态。  
  - 通胀类：CPI_YOY、PPI_YOY 等，按 6 个月趋势得到 Inflation 状态。  
  - 将 **Growth 状态 + Inflation 状态** 映射到上表四象限，得到该月的 **quadrant**。

---

## **5.5 第四步：象限 → TAA 权重（本仓库已实现）**

- **输入**：`quadrants: pd.Series`（每月一个象限标签）、SAA 权重、倾斜矩阵等。  
- **逻辑**：`taa_signal_engine` 根据当月象限查找资产大类倾斜量，按 SAA 比例分配到 16 个子策略，归一化后得到当月 TAA 权重。  
- 详见 **§3.2** 与 `core/taa_signal_engine.py`。

---

## **5.6 小结（PMI 链）**

| 阶段 | 输入 | 输出 |
|------|------|------|
| 1. 结构化 | 原始发布值 | `indicator_id=PMI_MANU`, `ref_month`, `value`, `data_type=Growth` |
| 2. 单指标规则 | 最近 3 月 PMI 序列 | Growth 扩张/收缩（及可选趋势强度） |
| 3. 多指标综合 | Growth + Inflation 状态 | 象限（Recovery / Overheat / Stagflation / Recession） |
| 4. TAA | 象限 + SAA + 倾斜矩阵 | 当月 TAA 权重（16 维） |

**时间长度**：PMI、社融、利率等建议 **至少 3 个月** 连续/趋势一致再确认；工业、CPI、PPI 建议 **6 个月** 趋势，以减弱基数与扰动。  
这样，**数据**（如 PMI）经过「结构化 → 单指标规则 → 多指标综合」后，才成为**稳定支撑周期划分与资产配置**的**信号**（象限），再进入 TAA 引擎。

