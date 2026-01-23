# 文档索引 | Documentation Index

快速找到你需要的文档 | Quick navigation to find what you need

---

## 🚀 快速开始

**我想立即获得协方差矩阵** → [`reverse_covariance.py`](./reverse_covariance.py)
```bash
python3 reverse_covariance.py
```

---

## 📖 文档导航

### 按使用目的

| 我想... | 查看文档 | 说明 |
|---------|---------|------|
| **立即运行脚本** | [`USAGE.md`](./USAGE.md) | 快速上手指南 |
| **理解数学原理** | [`MATHEMATICAL_PRINCIPLES.md`](./MATHEMATICAL_PRINCIPLES.md) | 完整数学推导 |
| **了解项目概况** | [`README.md`](./README.md) | 项目介绍 |
| **查看核心成果** | [`SUMMARY.md`](./SUMMARY.md) | 结果总结 |
| **比较两个协方差矩阵** | [`results/validation_report.txt`](./results/validation_report.txt) | 对比报告 |

---

## 📁 文件清单

### 核心代码

| 文件 | 说明 | 行数 | 用途 |
|------|------|------|------|
| **`reverse_covariance.py`** | ⭐ 独立脚本 | 489行 | 单文件完整实现 |
| `core/utils.py` | 工具函数 | 230行 | 矩阵操作、PSD投影 |
| `core/covariance_estimator.py` | 核心算法 | 280行 | 反向优化引擎 |
| `core/validation_metrics.py` | 验证指标 | 295行 | 对比分析 |
| `scripts/run_reverse_optimization.py` | 主流程 | 180行 | 完整分析流程 |
| `scripts/compare_matrices.py` | 对比工具 | 200行 | 矩阵对比显示 |

### 文档

| 文件 | 内容 | 适合人群 | 字数 |
|------|------|---------|------|
| **`MATHEMATICAL_PRINCIPLES.md`** | 数学原理详解 | 想深入理解 | ~8000字 |
| `USAGE.md` | 使用说明 | 实际使用者 | ~2500字 |
| `README.md` | 项目概述 | 初次了解 | ~3000字 |
| `SUMMARY.md` | 核心总结 | 快速回顾 | ~2000字 |
| `INDEX.md` | 本文档 | 导航索引 | ~500字 |

### 数据文件

| 文件 | 说明 | 大小 |
|------|------|------|
| `105.csv` | 输入: 105套组合配置 | 3.8KB |
| `prompt.md` | 输入: 现有协方差矩阵 | 1.2KB |
| `results/estimated_covariance.csv` | 输出: 估计的Σ | 374B |
| `results/validation_report.txt` | 输出: 验证报告 | ~3KB |

---

## 🎯 按技能水平导航

### 新手用户

1. 阅读 [`README.md`](./README.md) - 了解项目是什么
2. 阅读 [`USAGE.md`](./USAGE.md) - 学会运行脚本
3. 运行 `python3 reverse_covariance.py` - 获得结果

### 进阶用户

1. 阅读 [`MATHEMATICAL_PRINCIPLES.md`](./MATHEMATICAL_PRINCIPLES.md) 前3章 - 理解核心原理
2. 阅读代码注释 - 了解实现细节
3. 运行 `scripts/run_reverse_optimization.py` - 深度分析

### 专家用户

1. 阅读 [`MATHEMATICAL_PRINCIPLES.md`](./MATHEMATICAL_PRINCIPLES.md) 完整版 - 掌握所有细节
2. 研究源代码实现 - 理解算法优化
3. 自定义目标函数和约束 - 适配自己的需求

---

## 📊 核心概念速查

### 数学公式

**投资组合方差**:
```
σ²_p = w^T Σ w
```

**线性系统**:
```
A θ = b
其中 A (105×10), θ (10×1), b (105×1)
```

**最小二乘解**:
```
θ* = (A^T A)^{-1} A^T b
```

→ 详见 `MATHEMATICAL_PRINCIPLES.md` 第5章

### 三个维度

| 维度 | 作用方式 | 信息贡献 |
|------|---------|---------|
| `lifecycle` | 间接 (通过权重) | ~0.1% |
| `demand` | 间接 (通过权重) | ~0.1% |
| `risk_level` | 直接+间接 | ~1% |
| **权重矩阵** | 直接 | **~99%** |

→ 详见 `MATHEMATICAL_PRINCIPLES.md` 第4章

---

## 🔍 关键问题索引

### 理论问题

| 问题 | 答案位置 |
|------|---------|
| 为什么权重是主要信息？ | `MATHEMATICAL_PRINCIPLES.md` 第3章 |
| lifecycle 如何影响结果？ | `MATHEMATICAL_PRINCIPLES.md` 第4.2节 |
| demand 如何影响结果？ | `MATHEMATICAL_PRINCIPLES.md` 第4.3节 |
| risk_level 的双重作用？ | `MATHEMATICAL_PRINCIPLES.md` 第4.1节 |
| 最小二乘推导过程？ | `MATHEMATICAL_PRINCIPLES.md` 第5.1节 |

### 实践问题

| 问题 | 答案位置 |
|------|---------|
| 如何运行脚本？ | `USAGE.md` 快速开始 |
| 如何修改目标波动率？ | `USAGE.md` 参数调整 |
| 如何验证结果？ | `USAGE.md` Q4 |
| 两个Σ该用哪个？ | `SUMMARY.md` 使用建议 |
| 如何在代码中集成？ | `USAGE.md` 方法2 |

---

## 💻 代码导航

### 核心函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `load_portfolio_weights()` | `reverse_covariance.py:17` | 加载权重数据 |
| `reverse_optimize_covariance()` | `reverse_covariance.py:76` | 主优化函数 |
| `nearest_psd()` | `reverse_covariance.py:33` | PSD投影 |
| `cov_to_corr()` | `reverse_covariance.py:212` | 协方差→相关性 |
| `compute_portfolio_volatility()` | `reverse_covariance.py:229` | 计算波动率 |

### 算法实现

| 算法 | 位置 | 说明 |
|------|------|------|
| 最小二乘法 | `reverse_covariance.py:114` | 默认方法 |
| 矩匹配法 | `reverse_covariance.py:178` | 备选方法 |
| 特征值投影 | `reverse_covariance.py:33` | PSD约束 |

---

## 📈 结果文件导航

### 独立脚本输出

运行 `python3 reverse_covariance.py` 后生成:

```
reverse_covariance_matrix.csv      # 协方差矩阵 Σ (4×4)
reverse_correlation_matrix.csv     # 相关性矩阵 ρ (4×4)
reverse_volatility.csv              # 波动率向量 σ (4×1)
reverse_portfolio_volatility.csv   # 105个组合的波动率
```

### 完整系统输出

运行 `scripts/run_reverse_optimization.py` 后生成:

```
results/
├── estimated_covariance.csv        # 估计的Σ
├── existing_covariance.csv         # 现有的Σ
├── estimated_correlation.csv       # 估计的ρ
├── difference_matrix.csv           # Σ_est - Σ_exist
├── risk_aversion_by_level.csv      # 风险厌恶系数λ
├── portfolio_volatilities.csv      # 波动率详细对比
└── validation_report.txt           # ⭐ 完整验证报告
```

---

## 🌟 推荐阅读路径

### 路径1: 快速使用者 (15分钟)

1. `USAGE.md` 快速开始 → 运行脚本
2. `SUMMARY.md` 核心成果 → 理解结果
3. `results/validation_report.txt` → 查看对比

### 路径2: 深度学习者 (1-2小时)

1. `README.md` → 了解背景
2. `MATHEMATICAL_PRINCIPLES.md` 第1-4章 → 理解原理
3. `reverse_covariance.py` 源码 → 学习实现
4. `MATHEMATICAL_PRINCIPLES.md` 第5-8章 → 掌握细节

### 路径3: 研究开发者 (3-5小时)

1. 完整阅读 `MATHEMATICAL_PRINCIPLES.md`
2. 研究所有源代码实现
3. 运行完整分析并解读报告
4. 尝试修改参数和算法

---

## 🔗 外部参考

### 学术论文

1. **Markowitz (1952)** - Portfolio Selection
   - 现代投资组合理论基础

2. **Black & Litterman (1992)** - Global Portfolio Optimization
   - 反向优化的开创性工作

3. **Higham (1988)** - Nearest Positive Semidefinite Matrix
   - PSD投影的数学基础

### 在线资源

- NumPy文档: https://numpy.org/doc/
- Pandas文档: https://pandas.pydata.org/docs/
- 线性代数教程: https://www.khanacademy.org/math/linear-algebra

---

## 🆘 获取帮助

| 问题类型 | 解决方案 |
|---------|---------|
| 运行错误 | 检查 `USAGE.md` 常见问题 |
| 数学疑问 | 查阅 `MATHEMATICAL_PRINCIPLES.md` 第8章 |
| 结果解读 | 阅读 `SUMMARY.md` 关键发现 |
| 代码问题 | 查看源代码注释 |

---

**提示**: 使用 `Ctrl+F` 在本文档中搜索关键词快速定位！

*最后更新: 2026-01-23*
