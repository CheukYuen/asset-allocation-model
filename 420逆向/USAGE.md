# 独立脚本使用说明 | Standalone Script Usage

## 文件: `reverse_covariance.py`

这是一个**完全独立**的Python脚本,功能是从105套投资组合配置中反向推导协方差矩阵。

This is a **standalone** Python script that reverse-engineers a covariance matrix from 105 portfolio allocations.

---

## 快速开始 | Quick Start

### 1. 运行脚本

```bash
# 进入目录
cd 420逆向/

# 运行脚本 (需要Python 3.9+)
python3 reverse_covariance.py
```

### 2. 输出文件

脚本会自动生成4个CSV文件:

| 文件名 | 内容 |
|--------|------|
| `reverse_covariance_matrix.csv` | **协方差矩阵 Σ** (4×4) |
| `reverse_correlation_matrix.csv` | 相关性矩阵 ρ (4×4) |
| `reverse_volatility.csv` | 波动率向量 σ (4×1) |
| `reverse_portfolio_volatility.csv` | 105套组合的波动率 |

---

## 核心功能 | Core Features

### 输入 (Input)
- **105.csv**: 105套投资组合配置
  - 4大类资产: 现金(Cash)、债券(Bond)、权益(Equity)、另类资产(Commodity)
  - 5个风险等级: C1(保守) ~ C5(激进)

### 输出 (Output)
- **协方差矩阵 Σ** (4×4): 反映资产间的协方差关系
- **相关性矩阵 ρ** (4×4): 归一化的相关系数
- **波动率向量 σ** (4×1): 各资产的年化标准差

### 算法 (Algorithm)
- **反向优化 (Reverse Optimization)**: 最小二乘法
- **目标函数**: 最小化 ||w^T Σ w - 目标方差||²
- **约束**: 半正定 (PSD) 协方差矩阵

---

## 示例输出 | Example Output

### 协方差矩阵 Σ

```
               BOND      CASH  COMMODITY    EQUITY
BOND       0.353420 -0.156064  -0.439567 -0.122401
CASH      -0.156064  0.078589   0.227374  0.076644
COMMODITY -0.439567  0.227374   0.700864  0.262471
EQUITY    -0.122401  0.076644   0.262471  0.121795
```

### 波动率向量 σ

```
BOND:      0.594491 (59.45% 年化)
CASH:      0.280337 (28.03% 年化)
COMMODITY: 0.837176 (83.72% 年化)
EQUITY:    0.348991 (34.90% 年化)
```

### 相关性矩阵 ρ

```
               BOND      CASH  COMMODITY    EQUITY
BOND       1.000000 -0.936436  -0.883207 -0.589965
CASH      -0.936436  1.000000   0.968823  0.783405
COMMODITY -0.883207  0.968823   1.000000  0.898360
EQUITY    -0.589965  0.783405   0.898360  1.000000
```

---

## 技术细节 | Technical Details

### 依赖 (Dependencies)
```python
numpy>=1.20.0
pandas>=1.3.0
```

### Python版本
- **最低要求**: Python 3.9
- **兼容性**: 无SciPy依赖,仅使用numpy和pandas

### 算法步骤

1. **加载数据**: 从105.csv读取权重矩阵 W (105×4)
2. **定义目标**: 设定目标波动率 σ_target = [3%, 6%, 9%, 12%, 15%] for C1-C5
3. **构建设计矩阵**: A = [w_1^⊗ w_1, w_2^⊗ w_2, ..., w_105^⊗ w_105]
4. **最小二乘求解**: solve A @ σ_vec = σ_target²
5. **重构矩阵**: 将向量 σ_vec 重构为对称矩阵 Σ
6. **半正定投影**: 使用特征值分解确保 Σ ≥ 0

---

## 在代码中使用 | Use in Your Code

### 方法1: 直接运行脚本

```bash
python3 reverse_covariance.py
```

然后读取生成的CSV文件:

```python
import pandas as pd

# 读取协方差矩阵
cov_matrix = pd.read_csv('reverse_covariance_matrix.csv', index_col=0)
print(cov_matrix)
```

### 方法2: 导入模块使用

```python
from reverse_covariance import (
    load_portfolio_weights,
    reverse_optimize_covariance,
    cov_to_corr
)

# 加载数据
weights, risk_levels = load_portfolio_weights('105.csv')

# 估计协方差矩阵
cov_matrix = reverse_optimize_covariance(weights, risk_levels)

# 提取相关性矩阵
corr_matrix, volatility = cov_to_corr(cov_matrix)

print("协方差矩阵:")
print(cov_matrix)
```

---

## 与现有协方差矩阵对比 | Compare with Existing Σ

如果你想对比反向推导的Σ与现有的Σ (来自prompt.md):

```bash
# 运行完整分析 (包含对比)
python3 scripts/run_reverse_optimization.py

# 查看验证报告
cat results/validation_report.txt
```

---

## 参数调整 | Parameter Tuning

### 修改目标波动率

在 `_reverse_optimize_ls()` 函数中修改:

```python
# 默认: C1=3%, C2=6%, C3=9%, C4=12%, C5=15%
target_vols = 0.03 + (risk_levels - 1) * 0.03

# 自定义: C1=5%, C2=8%, C3=12%, C4=16%, C5=20%
target_vols = 0.05 + (risk_levels - 1) * 0.0375
```

### 选择优化方法

```python
# 方法1: 最小二乘法 (默认, 推荐)
cov_matrix = reverse_optimize_covariance(weights, risk_levels, method='least_squares')

# 方法2: 矩匹配法 (备选)
cov_matrix = reverse_optimize_covariance(weights, risk_levels, method='moment_matching')
```

---

## 常见问题 | FAQ

### Q1: 为什么相关性都是负的?

**A**: 这是反向优化的结果。由于105套组合的权重配置方式,算法推断出资产间存在负相关才能解释这些配置。这**不一定**反映真实市场相关性。

### Q2: 波动率为什么这么高?

**A**: 反向优化基于组合权重分散度估计协方差。如果目标波动率设置较高,或组合分散度大,估计的Σ会相应增大。可以通过调整目标波动率参数来缩放。

### Q3: 应该用反向推导的Σ还是历史数据的Σ?

**A**:
- **历史数据Σ** (prompt.md中的): 用于回测、风险计量、蒙特卡洛模拟
- **反向推导Σ**: 用于理解金融规划师的配置逻辑,检查组合一致性

两者各有用途,建议保留两个版本。

### Q4: 如何验证结果?

运行完整分析脚本:

```bash
python3 scripts/run_reverse_optimization.py
```

查看 `results/validation_report.txt` 了解详细对比。

---

## 进阶功能 | Advanced Features

### 自定义权重归一化

如果你的105.csv权重未归一化(总和≠100%):

```python
# 在 load_portfolio_weights() 中添加:
weights = weights / weights.sum(axis=1, keepdims=True)
```

### 添加正则化

在 `_reverse_optimize_ls()` 中添加L2正则化:

```python
# 在最小二乘求解前:
lambda_reg = 0.01  # 正则化系数
A_reg = np.vstack([A, np.sqrt(lambda_reg) * np.eye(10)])
target_vars_reg = np.hstack([target_vars, np.zeros(10)])

# 求解正则化问题
sigma_vec, _, _, _ = np.linalg.lstsq(A_reg, target_vars_reg, rcond=None)
```

---

## 文件结构总结 | File Structure Summary

```
420逆向/
├── reverse_covariance.py          ⭐ 独立脚本 (本文件)
├── 105.csv                        📊 输入数据
├── reverse_covariance_matrix.csv  📈 输出: 协方差矩阵
├── reverse_correlation_matrix.csv 📈 输出: 相关性矩阵
├── reverse_volatility.csv         📈 输出: 波动率向量
├── reverse_portfolio_volatility.csv 📈 输出: 组合波动率
└── USAGE.md                       📖 使用说明 (本文档)
```

---

## 许可证 | License

与主项目相同。

---

**最后更新**: 2026-01-23
**Python版本**: 3.9+
**依赖**: numpy, pandas
