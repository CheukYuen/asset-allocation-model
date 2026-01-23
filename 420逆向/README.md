# 协方差矩阵反向优化 (Covariance Matrix Reverse Optimization)

## 项目概述 (Overview)

该项目通过**反向优化**方法,从105套投资组合配置中反推估计协方差矩阵,并与现有的历史协方差矩阵进行对比验证。

This project uses **reverse optimization** to estimate a covariance matrix from 105 portfolio allocations and compares it with the existing historical covariance matrix.

---

## 输入数据 (Input Data)

### 1. 105套投资组合方案 (`105.csv`)

- **4大类资产**: 现金(Cash), 债券(Bond), 权益(Equity), 另类资产(Commodity)
- **3个维度组合**:
  - 7个人生阶段: 刚毕业、单身青年、二人世界、小孩学前、小孩成年前、子女成年、退休
  - 3种需求: 保值、增值、传承
  - 5个风险等级: C1(最保守) ~ C5(最激进)

### 2. 现有协方差矩阵 (`prompt.md`)

- 年化协方差矩阵 Σ (4×4)
- 年化波动率向量 σ (4×1)
- 长期相关性矩阵 ρ (4×4)

---

## 项目结构 (Project Structure)

```
420逆向/
├── core/                              # 核心模块
│   ├── utils.py                       # 工具函数 (加载矩阵, PSD投影, 转换)
│   ├── covariance_estimator.py        # 反向优化主引擎
│   └── validation_metrics.py          # 验证指标计算
├── scripts/                           # 执行脚本
│   ├── run_reverse_optimization.py    # 主执行脚本
│   └── compare_matrices.py            # 矩阵对比脚本
├── results/                           # 输出结果
│   ├── estimated_covariance.csv       # 估计的协方差矩阵
│   ├── existing_covariance.csv        # 现有协方差矩阵
│   ├── estimated_correlation.csv      # 估计的相关性矩阵
│   ├── difference_matrix.csv          # 差异矩阵
│   ├── risk_aversion_by_level.csv     # 各风险等级的风险厌恶系数
│   ├── portfolio_volatilities.csv     # 投资组合波动率对比
│   └── validation_report.txt          # 完整验证报告
├── 105.csv                            # 输入: 105套投资组合
├── prompt.md                          # 输入: 现有协方差矩阵
└── README.md                          # 本文件
```

---

## 使用方法 (Usage)

### 环境准备

```bash
# 激活虚拟环境 (如果存在)
cd /Users/zlin/Developer/github/asset-allocation-model
source venv/bin/activate

# 或使用项目Python
cd 420逆向
```

### 运行分析

```bash
# 1. 运行反向优化 (生成所有结果)
../venv/bin/python3 scripts/run_reverse_optimization.py

# 2. 查看矩阵对比 (可选)
../venv/bin/python3 scripts/compare_matrices.py
```

### 查看结果

```bash
# 查看验证报告
cat results/validation_report.txt

# 查看CSV结果
ls -lh results/*.csv
```

---

## 核心算法 (Core Algorithm)

### 1. 反向优化原理

在均值-方差优化框架下,最优投资组合权重满足:

```
w* = (1/λ) Σ^(-1) μ
```

反向求解隐含收益率:

```
μ_implied = λ Σ w*
```

### 2. 估计步骤

1. **按风险等级分层**: 将105套组合分为C1-C5五组
2. **估计风险厌恶系数λ**: 每个风险等级对应不同的λ值
3. **最小二乘求解**: 找到最佳拟合的协方差矩阵Σ
   ```
   min ||W^T Σ W - target_variances||²
   ```
4. **半正定约束**: 使用特征值分解确保Σ为半正定矩阵

### 3. 验证指标

- **Frobenius范数**: ||Σ_est - Σ_exist||_F
- **投资组合波动率RMSE**: 跨105套组合的波动率均方根误差
- **相关性对比**: 比较两个协方差矩阵的相关结构
- **特征值分析**: 比较主成分和风险因子

---

## 分析结果总结 (Analysis Results Summary)

### 关键发现 (Key Findings)

#### 1. 协方差矩阵差异巨大
- **Frobenius范数差异**: 1.152 (相对差异 23.97倍)
- **结论**: 估计的Σ与现有的Σ存在显著差异

#### 2. 投资组合波动率对比
- **波动率RMSE**: 0.0847
- **波动率相关性**: 0.97 (高度相关)
- **平均差异**: 各风险等级的波动率被系统性低估

| 风险等级 | 估计波动率 | 现有波动率 | 差异 |
|---------|----------|----------|------|
| C1 (最保守) | 0.1016 | 0.0281 | +261% |
| C2 | 0.1172 | 0.0463 | +153% |
| C3 | 0.1379 | 0.0647 | +113% |
| C4 | 0.1794 | 0.0884 | +103% |
| C5 (最激进) | 0.2154 | 0.1082 | +99% |

#### 3. 风险厌恶系数λ
```
C1: λ = 1.00   (最保守)
C2: λ = 1.32
C3: λ = 2.15
C4: λ = 3.87
C5: λ = 10.00  (最激进)
```

⚠️ **注意**: λ值未呈单调递减,说明105套组合并非完全基于均值-方差优化设计。

---

## 结论与建议 (Conclusions & Recommendations)

### 主要结论

1. **105套投资组合并非基于现有Σ的均值-方差最优化结果**
   - 现有协方差矩阵(基于历史数据)与组合配置逻辑存在显著差异
   - 金融规划师在设计组合时可能更多考虑主观因素(人生阶段、需求类型)

2. **估计的Σ更好地拟合105套组合**
   - 能够解释不同风险等级间的波动率差异
   - 但估计的Σ本身数值较大,可能存在过拟合风险

3. **两种Σ各有优势**:
   - **现有Σ**: 基于真实市场历史数据,反映资产间真实相关性
   - **估计Σ**: 反映金融规划师的主观风险判断和配置逻辑

### 推荐行动方案

#### 方案A: 保留现有Σ (推荐用于量化回测)
**适用场景**: 基于历史数据的回测分析、风险计量

**理由**:
- 现有Σ来自真实市场数据,更可靠
- 适合用于蒙特卡洛模拟、VaR计算等风险分析

#### 方案B: 使用估计Σ (推荐用于理解配置逻辑)
**适用场景**: 理解105套组合的内在一致性、反向工程金融规划师思路

**理由**:
- 估计Σ能够最佳解释现有配置方案
- 帮助识别哪些组合偏离了"理性"配置

#### 方案C: 混合方法 (推荐用于优化配置)
**加权平均**:
```
Σ_hybrid = α * Σ_existing + (1-α) * Σ_estimated
```
建议 α ∈ [0.6, 0.8],更重视历史数据

**理由**:
- 结合市场真实数据和主观配置逻辑
- 平衡量化分析与实务经验

#### 方案D: 重新校准投资组合 (长期改进)
**行动**:
1. 识别与现有Σ偏差最大的组合 (如#25号组合)
2. 基于现有Σ重新优化部分组合权重
3. 确保C1-C5风险等级的波动率呈单调递增

---

## 技术说明 (Technical Notes)

### Python兼容性
- **Python 3.9+** 兼容
- **无SciPy依赖**: 仅使用numpy和pandas
- 避免使用Python 3.10+特性 (match/case, 类型联合)

### 算法约束
- **半正定约束**: 使用特征值分解确保协方差矩阵有效
- **对称约束**: Σ = (Σ + Σ^T) / 2
- **非负权重**: 投资组合权重≥0且总和为1

### 性能
- 处理105套组合耗时: < 1秒
- 内存占用: < 50MB

---

## 文件说明 (File Descriptions)

### 核心模块

- **`core/utils.py`**:
  - 加载协方差矩阵 (`load_existing_covariance`)
  - 半正定投影 (`nearest_psd_matrix`)
  - 协方差-相关性转换 (`cov_to_corr`, `corr_to_cov`)

- **`core/covariance_estimator.py`**:
  - `CovarianceReverseOptimizer` 类
  - 反向优化主算法 (`reverse_optimize_covariance`)
  - 风险厌恶估计 (`estimate_risk_aversion`)

- **`core/validation_metrics.py`**:
  - 协方差矩阵对比 (`compare_covariance_matrices`)
  - 波动率对比 (`compare_volatilities`)
  - 生成验证报告 (`generate_validation_report`)

### 输出文件

- **`results/estimated_covariance.csv`**: 估计的4×4协方差矩阵
- **`results/existing_covariance.csv`**: 现有的4×4协方差矩阵
- **`results/difference_matrix.csv`**: Σ_est - Σ_exist
- **`results/portfolio_volatilities.csv`**: 105套组合的波动率对比详情
- **`results/validation_report.txt`**: 完整分析报告

---

## 参考文献 (References)

1. **Black-Litterman Model**:
   - Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.

2. **Reverse Optimization**:
   - He, G., & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. *Goldman Sachs Quantitative Resources Group*.

3. **Positive Semi-Definite Projection**:
   - Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite matrix. *Linear Algebra and its Applications*, 103, 103-118.

---

## 联系方式 (Contact)

如有问题或建议,请参考主项目README或提交Issue。

For questions or suggestions, please refer to the main project README or submit an Issue.

---

*生成时间: 2026-01-23*
*Python版本: 3.9+*
*依赖: numpy, pandas*
