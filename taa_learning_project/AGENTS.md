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

