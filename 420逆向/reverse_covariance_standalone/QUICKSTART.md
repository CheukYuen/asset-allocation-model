# 快速参考卡片 | Quick Reference

## ⚡ 一键运行

```bash
python reverse_covariance.py
```

**输出**: 4 个 CSV 文件自动生成在当前目录

---

## 📂 文件说明

| 文件 | 类型 | 说明 |
|------|------|------|
| `reverse_covariance.py` | 脚本 | 核心算法 (可直接运行) |
| `105.csv` | 输入 | 105个组合配置 |
| `reverse_covariance_matrix.csv` | 输出 | 4×4 协方差矩阵 |
| `reverse_correlation_matrix.csv` | 输出 | 4×4 相关系数矩阵 |
| `reverse_volatility.csv` | 输出 | 各资产波动率 |
| `reverse_portfolio_volatility.csv` | 输出 | 105个组合重构结果 |

---

## 📖 文档导航

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| `README.md` | 项目概览 + 快速入门 | 5 分钟 |
| `USAGE.md` | 完整使用手册 | 20 分钟 |
| `数据原理解释.md` | 数学原理推导 | 40 分钟 |
| `INDEX.md` | 目录索引 (快速查找) | - |

---

## 🎯 核心算法

**问题**: 从 105 个组合反推 4 种资产的协方差矩阵

**公式**:
```
min Σᵢ (σₚ,ᵢ² - σₜₐᵣ,ᵢ²)²
其中 σₚ,ᵢ² = wᵢᵀ Σ wᵢ
```

**求解**: 正规方程 `θ* = (AᵀA)⁻¹ Aᵀb`

---

## 📊 典型输出

### 协方差矩阵
```
        现金      债券      股票      另类
现金  0.3534  -0.1561  -0.4396  -0.1224
债券 -0.1561   0.0786   0.2274   0.0766
股票 -0.4396   0.2274   0.7009   0.2625
另类 -0.1224   0.0766   0.2625   0.1218
```

### 波动率
- 现金: 59.45%
- 债券: 28.03%
- 股票: 83.72%
- 另类: 34.90%

### 相关性 (关键发现)
- 现金 ↔ 债券: **-0.937** (强负相关)
- 债券 ↔ 股票: **+0.969** (强正相关)
- 股票 ↔ 另类: **+0.898** (强正相关)

---

## ❓ 快速 FAQ

**Q: 现金波动率为什么这么高？**
→ 现金占比差异大 (0%~90%)，作为组合"缓冲器"

**Q: 行顺序会影响结果吗？**
→ 不会，最小二乘法与顺序无关

**Q: 如何验证结果？**
→ 查看 `reverse_portfolio_volatility.csv` 的误差列

**Q: 如何修改风险等级？**
→ 编辑脚本中的 `RISK_LEVELS` 字典

---

## 🔧 Python 代码示例

### 读取协方差矩阵
```python
import pandas as pd
import numpy as np

# 读取协方差矩阵
cov_df = pd.read_csv('reverse_covariance_matrix.csv', index_col=0)
Sigma = cov_df.values

# 读取某个组合权重
w = np.array([0.15, 0.25, 0.50, 0.10])  # 现金, 债券, 股票, 另类

# 计算组合波动率
sigma_p = np.sqrt(w @ Sigma @ w)
print(f"组合波动率: {sigma_p:.2%}")
```

### 验证重构精度
```python
# 读取重构结果
results = pd.read_csv('reverse_portfolio_volatility.csv')

# 计算误差统计
errors = results['误差']
print(f"平均误差: {errors.mean():.4f}")
print(f"均方根误差 (RMSE): {np.sqrt((errors**2).mean()):.4f}")
print(f"最大误差: {errors.abs().max():.4f}")
```

---

## 🎓 学习路径

### 新手 (5分钟)
```
README.md → 运行脚本 → 查看输出
```

### 用户 (30分钟)
```
USAGE.md (输入/输出) → 修改参数 → 重新运行
```

### 专家 (1小时)
```
数据原理解释.md → 源代码阅读 → 添加约束
```

---

## 🔗 相关链接

- **完整文档**: [INDEX.md](INDEX.md)
- **数学原理**: [数据原理解释.md](数据原理解释.md)
- **GitHub**: [项目仓库]

---

## 📌 依赖要求

```bash
# Python 3.9+
pip install numpy pandas
```

**无需 scipy** (生产环境兼容)

---

## ✅ 验证清单

运行脚本后，检查：
- [ ] 生成了 4 个 CSV 文件
- [ ] 协方差矩阵是对称的
- [ ] 重构误差 RMSE < 2%
- [ ] 所有特征值 ≥ 0 (半正定)

---

**更新**: 2026-01-23 | **版本**: v1.0
