# 1. 项目目标与范围

本项目构建一个独立的 **TAA 市场信号模块**，用于：

1. 验证 420 套 SAA 的长期合理性  
2. 在不改变风险等级前提下提供小幅偏移 Δw（±20%）  
3. 通过回测（10 年）与模拟（30 年）比较  
   **SAA vs SAA + TAA 市场信号模块**

模块定位：验证层 / 辅助优化层，不生成新的资产配置组合。


# 2. SAA 基线（420 套）

来源于现有资产配置体系：

$$
4 \times 7 \times 3 \times 5 = 420
$$

本项目不调整原有 SAA 权重。


## 2.1 SAA 的作用

- 长期战略底座  
- 不随市场短期波动调整  
- 风险等级固定  
- 本项目仅验证 + Δw 偏移  


# 3. TAA 市场信号模块

由两个部分组成：

1. CME：长期资本市场预期（验证层）  
2. 美林时钟：短期宏观象限（偏移层）  


## 3.1 美林时钟象限

| 象限 | 增长 | 通胀 | 偏好方向 |
|------|------|------|----------|
| Recovery | ↑ | ↓ | 股票、信用债 |
| Overheat | ↑ | ↑ | 商品 |
| Stagflation | ↓ | ↑ | 黄金、商品 |
| Recession | ↓ | ↓ | 长久期国债、现金 |


## 3.2 大类偏移规则（±20% 上限）

| 象限 | 股票 | 固收 | 商品 | 黄金 | 现金 |
|------|------|------|------|------|------|
| Recovery | +5% | -3% | 0 | -2% | 0 |
| Overheat | +2% | -3% | +5% | 0 | -4% |
| Stagflation | -5% | 0 | +3% | +5% | -3% |
| Recession | -5% | +5% | 0 | 0 | 0 |


# 4. 回测设计（10 年 · 月频）

对比：

**SAA（固定） vs SAA + Δw（TAA 模块）**


# 4.1 回测输入

## 4.1.1 SAA 权重（16 维）

$$
w_{\text{SAA}} \in \mathbb{R}^{16}
$$

## 4.1.2 16 个策略月收益

$$
r_t = [r_{1,t}, …, r_{16,t}]^\top
$$

## 4.1.3 月度象限序列  
由 PMI / GDP nowcast / CPI 等生成。


# 4.2 TAA 偏移（大类 → 子策略）

包含所有关键公式（仅出现一次）。


## (1) 大类偏移

$$
\Delta w_{\text{asset},t}
\in
\{\text{Equity},\text{Bond},\text{Commodity},\text{Gold},\text{Cash}\}
$$


## (2) 按 SAA 分摊到子策略

$$
\Delta w^{(0)}_{\text{strategy}, i,t}
=
\Delta w_{\text{asset}, t}
\cdot
\frac{
w_{\text{SAA}, i}
}{
\sum_{j\in AC} w_{\text{SAA},j}
}
$$


## (3) 子策略敏感度 β（待定）

| 子策略 | 大类 | β |
|--------|------|-----|
| 现金 | 现金 | 0 |
| 存款固收 | 固收 | 0 |
| 纯债 | 固收 | 0.8 |
| 非标 | 固收 | 0.1 |
| 固收+ | 固收 | 0.5 |
| 海外债券 | 固收 | 0.3 |
| 股债混合 | 股票 | 0.6 |
| 股票（A 股） | 股票 | 1.2 |
| 海外股票 | 股票 | 0.5 |
| 海外股债混合 | 股票 | 0.4 |
| 商品及宏观策略 | 另类 | 1.2 |
| 量化对冲 | 另类 | 0 |
| 房地产股权 | 另类 | 0.1 |
| PE/VC | 另类 | 0 |
| 海外另类 | 另类 | 0.1 |
| 结构化 | 另类 | 0 |


## (4) β 微调后的最终偏移

$$
\Delta w_{\text{strategy}, i,t}
=
\beta_i \cdot
\Delta w^{(0)}_{\text{strategy}, i,t}
$$


## (5) Normalize（统一定义一次）

$$
Normalize(x)
=
\frac{\max(x,0)}{\sum_i \max(x_i,0)}
$$


# 4.3 回测计算（关键公式）

## (1) SAA 组合收益

$$
r_{\text{SAA}, t} = w_{\text{SAA}}^\top r_t
$$

## (2) TAA 最终权重

$$
w_{\text{final}, t}
=
Normalize(w_{\text{SAA}} + \Delta w_{\text{strategy},t})
$$

## (3) TAA 组合收益

$$
r_{\text{final}, t} = w_{\text{final}, t}^\top r_t
$$

## (4) 年化收益

$$
\mu = 12 \cdot \operatorname{mean}(r_t)
$$

## (5) 年化波动

$$
\sigma = \sqrt{12} \cdot \operatorname{std}(r_t)
$$

## (6) 夏普比率

$$
r_t^{excess} = r_t - r_f^{monthly}
$$

$$
\text{Sharpe}
=
\frac{
12 \cdot \operatorname{mean}(r_t^{excess})
}{
\sqrt{12} \cdot \operatorname{std}(r_t^{excess})
}
$$


# 4.4 回测判优

满足：

1. 收益不下降  
2. 波动 ≤ SAA +1%  
3. MDD ≤ SAA +2%  
4. 夏普不下降  

判定：TAA 优于 SAA。


# 5. 模拟设计（30 年 · 月频 · 蒙特卡罗）

## 5.1 CME 输入

### (1) 年化均值

$$
\mu_{\text{annual}} \in \mathbb{R}^{16}
$$

月化：

$$
\mu_{\text{monthly}} = \frac{\mu_{\text{annual}}}{12}
$$

### (2) 年化协方差

$$
\Sigma_{\text{annual}} \in \mathbb{R}^{16\times 16}
$$

月化：

$$
\Sigma_{\text{monthly}} = \frac{\Sigma_{\text{annual}}}{12}
$$


## 5.2 象限路径（固定轮换：8 年）

可扩展为马尔可夫链。


## 5.3 模拟步骤（关键公式）

### (1) 生成收益

$$
r_t \sim \mathcal{N}(\mu_{\text{monthly}},\Sigma_{\text{monthly}})
$$

### (2) SAA 组合收益

$$
r_{\text{SAA}, t} = w_{\text{SAA}}^\top r_t
$$

### (3) TAA 偏移与权重  
（沿用回测定义）

### (4) 净值更新

$$
V_{t} = V_{t-1}(1 + r_t)
$$

累计形式：

$$
V_t = V_0 \cdot \prod_{\tau=1}^{t}(1 + r_\tau)
$$


# 5.4 模拟指标

## (1) 30 年终值中位数

$$
\tilde{V}_{30Y}
$$

## (2) 90% 区间

$$
[V_{5\%}, V_{95\%}]
$$

## (3) 5% 下行风险

$$
V_{30Y,5\%}
$$

## (4) 胜率

$$
P(V^{final}_{30Y} > V^{SAA}_{30Y})
$$


# 5.5 模拟判优

满足：

1. 终值中位数更高  
2. 胜率 > 50%  
3. 下行风险不恶化  

判定：TAA 模块具备长期升级价值。
