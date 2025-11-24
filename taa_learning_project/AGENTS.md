下面我把你整个 TAA 学习项目的 **跨文件 import 规范** 和 **Outputs（数据/文件）建议** 完整整理出来，形成你在 Cursor 建项目时可以直接参照的统一标准。

内容已经完全基于 **TAA市场信号.md** PRD 逻辑整理（你之前已上传并引用）
（文件路径：`/mnt/data/TAA市场信号.md`）。

---

# ✅ 一、项目目录结构（推荐规范）

适合你在 Cursor 用来学习 Python / numpy / pandas，且未来可扩展到真实数据。

```
taa_learning_project/
│
├── data/                     # （可选）本地缓存模拟数据或未来真实数据
│   ├── mock_returns.csv
│   ├── mock_quadrants.csv
│   └── mock_saa_weights.csv
│
├── core/                     # 最核心的数学/引擎逻辑
│   ├── mock_data.py                # 生成 SAA、收益率、象限
│   ├── taa_signal_engine.py        # Δw + β + Normalize + w_final
│   ├── backtest_engine.py          # 回测指标
│   ├── mc_simulation.py            # 蒙特卡罗
│   └── utils.py                    # 通用工具：normalize, seed, helpers
│
├── scripts/                  # 单次可执行脚本（只 import core）
│   ├── run_mock_data.py
│   ├── run_taa_signal.py
│   ├── run_backtest.py
│   └── run_mc_simulation.py
│
└── README.md
```

你当前练习阶段要重点建立：

* `mock_data.py`
* `taa_signal_engine.py`
* `backtest_engine.py`
* `utils.py`

Scripts 先不用，但未来很好用。

---

# ✅ 二、跨文件 import 关系（最重要的规范）

这是你问的重点。下面给出最终版本标准。

---

## **1）mock_data.py**

无外部依赖，只依赖：

```python
import numpy as np
import pandas as pd
```

**别 import core 中其他文件。
mock_data 是最底层，无依赖。**

---

## **2）utils.py**

（Normalize 或 seed 等通用工具）

```python
import numpy as np
```

也 **不 import 项目其他文件**。

---

## **3）taa_signal_engine.py**

依赖：

```python
import numpy as np
import pandas as pd

from core.utils import normalize_weights
```

可选（不是必须）：

```python
from core.mock_data import generate_saa_weights, generate_quadrant_path
```

但建议 **不要 import mock_data**，因为那样 TAA 引擎会依赖 mock 层。
而你会希望引擎未来可以接真实数据。

**推荐写法：**
`__main__` 里才 import mock_data，用于测试。

---

## **4）backtest_engine.py**

依赖：

```python
import numpy as np
import pandas as pd

from core.utils import normalize_weights
from core.taa_signal_engine import compute_final_weights_over_time
```

同样不建议 import mock_data。测试区域再导入 mock。

---

## **依赖关系图（最清晰版）**

```
mock_data.py         utils.py
      ↓                 ↓
  (optional)        (normal)
      ↓                 ↓
   taa_signal_engine.py
            ↓
     backtest_engine.py
            ↓
         scripts/*.py
```

---

# ✅ 三、Outputs 是否需要补充？

你的观察很正确：**之前 Prompt 没有明确规范每个文件的“输出数据格式”！**

这会让 Cursor 难以保持 API 一致。

因此我给出统一规范：

---

## 🔻 **统一 Outputs 规范**

### **1）mock_data 输出**

所有生成函数统一如下结构：

```python
w_saa: np.ndarray           # shape (16,)
returns_df: pd.DataFrame    # shape (T, 16)
quadrants: pd.Series        # shape (T,)
```

可输出到 csv 时格式：

```
mock_returns.csv      # columns: strategy1...strategy16
mock_quadrants.csv    # column: quadrant
mock_saa_weights.csv  # column: weight
```

---

### **2）taa_signal_engine 输出**

核心输出：

```python
weights_final: pd.DataFrame
# shape: (T, 16)
# 每行 normalize 后 sum=1
```

列名必须与 `returns_df` 对齐。

---

### **3）backtest_engine 输出**

两个分层指标：

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

统一返回格式：

```python
ComparisonResult
```

---

### **4）Monte-Carlo（mc_simulation.py）输出**

标准统一为：

```python
MCResult:
    median: float
    p5: float
    p95: float
    worst_5pct: float
    all_paths: np.ndarray   # optional, shape (N_paths, T)
```

---

# ✅ 四、统一的单文件 Prompt 壳（含 import & outputs 标准）

你以后只要丢这个模板给 Cursor，就可以可靠输出一份规范文件。

---

## Template（你直接复制即可）

````text
You are a senior quantitative Python developer.
I am building a TAA learning project based on the PRD in /mnt/data/TAA市场信号.md.

For this step, generate ONE SINGLE python file following these rules:

==================================================
## 1. Project import hierarchy (VERY IMPORTANT)
- mock_data.py: only numpy/pandas
- utils.py: only numpy
- taa_signal_engine.py:
    from core.utils import normalize_weights
- backtest_engine.py:
    from core.utils import normalize_weights
    from core.taa_signal_engine import compute_final_weights_over_time

Never create circular imports.
Never make core files depend on scripts.
==================================================

## 2. Output interface (MUST follow)

### mock_data.py:
- w_saa: np.ndarray (16,)
- returns_df: pd.DataFrame (T,16)
- quadrants: pd.Series (T,)

### taa_signal_engine.py:
- final_weights_df: pd.DataFrame (T,16)

### backtest_engine.py:
Return a ComparisonResult dataclass:
- saa: BacktestResult
- taa: BacktestResult
- is_taa_better: bool

### mc_simulation.py:
Return MCResult dataclass:
- median
- p5
- p95
- worst_5pct
- all_paths (optional)

==================================================

## 3. Coding constraints
- Python 3.9 (must run on 3.11 locally too)
- Only numpy + pandas
- No scipy, no sklearn
- Use docstrings + comments to teach the logic
- Use synthetic data in __main__
- File must be runnable alone: `python filename.py`

==================================================

## 4. Replace this section with module-specific goals
[YOUR MODULE DESCRIPTION]

==================================================

## 5. Output rules
- Return ONLY python code
- No ``` fences
- No extra text
````

---
