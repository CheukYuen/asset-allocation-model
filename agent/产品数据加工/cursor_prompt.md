# 产品池过滤与导出脚本 - 技术规格说明

## 【目标】
从 `raw_products.csv` 读取产品数据，包含 16 种一级策略（用字段 `sub_category` 表示一级策略，配置常量 `STRATEGY_COL`）。
按"4 策略族 z-score 模板（不做 winsorize）"进行策略内评分与过滤，并输出：
1. 每个一级策略单独一个 CSV（文件名 `outputs/by_strategy/{strategy_name}.csv`）
2. 每个一级策略的原始格式 CSV（`outputs/by_strategy/raw/{strategy_name}.csv`）
3. 全部策略合并 CSV（`outputs/all_strategies_pool.csv` 和 `outputs/all_strategies_pool_raw.csv`）

## 【输入 CSV 字段】
```
product_name, product_code, currency, asset_class, sub_category, risk_level,
return_1y, return_3y, return_5y,
volatility_1y, volatility_3y, volatility_5y,
max_drawdown_1y, max_drawdown_2y, max_drawdown_3y,
sharpe_ratio_1y, sharpe_ratio_3y, sharpe_ratio_5y
```

## 【核心指标选择（优先级口径）】
| 指标 | 优先级顺序 |
|------|-----------|
| return | `return_3y` > `return_5y` > `return_1y` |
| volatility | `volatility_1y` > `volatility_3y` > `volatility_5y` |
| drawdown | `max_drawdown_2y` > `max_drawdown_1y` > `max_drawdown_3y` |
| sharpe | `sharpe_ratio_3y` > `sharpe_ratio_1y` > `sharpe_ratio_5y` |

> 注意：drawdown 统一转换为回撤幅度 `metric_drawdown = abs(drawdown)`（正数越大越差）。

## 【4 策略族 z-score 评分模板】

### 为什么使用 z-score？

**问题**：四个核心指标的量纲完全不同，无法直接相加：
| 指标 | 量纲 | 数值范围示例 | 方向 |
|------|------|-------------|------|
| return | 百分比收益率 | -0.30 ~ +0.80 | 越大越好 |
| volatility | 年化标准差 | 0.05 ~ 0.40 | 越小越好 |
| drawdown | 回撤幅度 | 0.02 ~ 0.60 | 越小越好 |
| sharpe | 无量纲比率 | -1.0 ~ +3.0 | 越大越好 |

**解决方案**：z-score 标准化

```
z = (x - μ) / σ
```
- `x`：原始值
- `μ`：策略内均值
- `σ`：策略内标准差

**z-score 的作用**：
1. **消除量纲**：转换后所有指标都是"距离均值多少个标准差"，无量纲
2. **统一尺度**：均值 = 0，标准差 = 1，不同指标可直接比较
3. **保留相对排名**：原始值越高，z-score 越高（单调变换）
4. **策略内公平**：同一策略内的产品互相比较，避免跨策略干扰

**示例**：
| 产品 | return (原始) | z_return | 解读 |
|------|--------------|----------|------|
| A基金 | 25% | +1.5 | 高于均值 1.5 个标准差（优秀） |
| B基金 | 10% | -0.3 | 低于均值 0.3 个标准差（略差） |
| C基金 | 15% | +0.2 | 略高于均值（中等偏上） |

### z-score 计算范围
在"策略内（同一 sub_category）"对各指标计算 z-score：`z_return`, `z_sharpe`, `z_vol`, `z_dd`

### 综合分公式
```
score = w_ret × z_return + w_sharpe × z_sharpe - w_vol × z_vol - w_dd × z_dd
```

**符号说明**：
- `z_return`, `z_sharpe`：越大越好 → **正号**（+）
- `z_vol`, `z_dd`：越大越差 → **负号**（-）

### z-score 负数处理

**z-score 负数是正常情况**，不需要特殊处理：
- 负数含义：原始值 < 均值（低于平均水平）
- 正数含义：原始值 > 均值（高于平均水平）

**代码处理逻辑**：
1. `safe_zscore()` 只处理 `std=0` 的边界情况（返回 0）
2. 计算 score 时，NaN 替换为 0（相当于"平均水平"）

**负数 × 正负号 = 最终贡献**：
| 指标 | z-score | 公式符号 | 最终贡献 | 解读 |
|------|---------|---------|---------|------|
| z_return = -1.5 | 收益差于均值 | **+** | -1.5 × w | 扣分 ✓ |
| z_return = +1.5 | 收益优于均值 | **+** | +1.5 × w | 加分 ✓ |
| z_vol = -1.5 | 波动低于均值（好） | **-** | +1.5 × w | 加分 ✓ |
| z_vol = +1.5 | 波动高于均值（差） | **-** | -1.5 × w | 扣分 ✓ |

> 结论：公式中的 **+/-** 符号已经保证了"好的指标加分，差的指标扣分"。

四类权重配置：
| 策略族 | 说明 | ret | sharpe | vol | dd |
|--------|------|-----|--------|-----|-----|
| A | 权益进攻（Equity Attack） | 0.35 | 0.35 | 0.15 | 0.15 |
| B | 平衡增强/固收+（Balanced） | 0.20 | 0.40 | 0.15 | 0.25 |
| C | 纯防守（Defensive FI/Cash） | 0.10 | 0.45 | 0.15 | 0.30 |
| D | 分散/另类（Diversifier） | 0.25 | 0.35 | 0.20 | 0.20 |

## 【16 一级策略 → 4 策略族映射】
```python
STRATEGY_FAMILY_MAP = {
    # A - 权益进攻（Equity Attack）
    "股票型": "A",
    "股票增强型": "A",
    "行业主题型": "A",
    "量化多因子": "A",
    
    # B - 平衡增强/固收+（Balanced）
    "固收增强型": "B",
    "偏债混合型": "B",
    "二级债基": "B",
    "可转债型": "B",
    
    # C - 纯防守（Defensive FI/Cash）
    "纯债型": "C",
    "货币基金": "C",
    "现金类-其他": "C",
    "短债型": "C",
    
    # D - 分散/另类（Diversifier）
    "商品型": "D",
    "CTA策略": "D",
    "多策略对冲": "D",
    "市场中性": "D",
}
```

**Fallback 规则**（当策略不在映射表时）：
- `asset_class == "股票类"` → A
- `asset_class == "固收+"` 或 `sub_category` 包含 "固收" → B
- `asset_class == "现金类"` 或 `sub_category` 包含 "货币"/"短债"/"纯债" → C
- `sub_category` 包含 "商品"/"cta"/"对冲"/"另类" → D
- 以上均不匹配 → 默认 B，并输出 warning 日志

## 【过滤闸门配置（策略内分位数）】

### 什么是 Quantile（分位数）？

**分位数**是将数据从小到大排序后，处于某个百分比位置的值：
```
quantile(0.90) = 第 90% 位置的值（90% 的数据都比它小）
quantile(0.20) = 第 20% 位置的值（20% 的数据都比它小）
```

### 配置参数
```python
FILTER_CONFIG = {
    "volatility_quantile": 0.90,   # 剔除波动最差 10%
    "drawdown_quantile": 0.90,     # 剔除回撤最差 10%
    "sharpe_quantile": 0.20,       # 剔除夏普最差 20%
}
```

### 过滤规则图解

假设某策略有 100 个产品：

**波动率/回撤 `quantile(0.90)`**（越大越差）：
```
|-------- 90% 产品（波动低，保留） --------|-- 10% 最差（剔除）--|
低波动 ◄──────────────────────────────────► 高波动
                                    ▲
                              quantile(0.90)
```

**夏普 `quantile(0.20)`**（越大越好）：
```
|-- 20% 最差（剔除）--|-------- 80% 产品（夏普高，保留） --------|
低夏普 ◄──────────────────────────────────► 高夏普
                ▲
          quantile(0.20)
```

### 过滤逻辑代码
```python
# 计算分位数阈值
vol_threshold = strategy_df["metric_volatility"].quantile(0.90)
dd_threshold = strategy_df["metric_drawdown"].quantile(0.90)
sharpe_threshold = strategy_df["metric_sharpe"].quantile(0.20)

# 保留条件
keep_mask = (
    (volatility <= vol_threshold) &   # 波动 ≤ 90%分位 → 保留
    (drawdown <= dd_threshold) &      # 回撤 ≤ 90%分位 → 保留
    (sharpe >= sharpe_threshold)      # 夏普 ≥ 20%分位 → 保留
)
```

### 过滤规则总结
- 剔除波动最差 10%：`metric_volatility > quantile(0.90)`
- 剔除回撤最差 10%：`metric_drawdown > quantile(0.90)`
- 剔除夏普最差 20%：`metric_sharpe < quantile(0.20)`
- 产品数 < 3 的策略不做过滤
- 指标为 NaN 的行保留（避免误杀）

> **关键**：`>` 还是 `<` 取决于指标方向——越大越差用 `>`，越大越好用 `<`。

## 【缺失值处理】
- 最大允许缺失指标数：`MAX_MISSING_METRICS = 1`
- 四项核心指标中缺失超过 1 项的行直接剔除
- 缺失 1 项以内的按优先级口径补齐

## 【输出要求】
### 1. 策略 CSV（带评分）
路径：`outputs/by_strategy/{strategy_name}.csv`
列：
```
product_name, product_code, currency, asset_class, sub_category, risk_level,
metric_return, metric_volatility, metric_drawdown, metric_sharpe,
z_return, z_sharpe, z_vol, z_dd, score, family
```

### 2. 策略 CSV（原始格式）
路径：`outputs/by_strategy/raw/{strategy_name}.csv`
列顺序与 `raw_products.csv` 一致，按 score 降序排列

### 3. 全策略合并 CSV（带评分）
路径：`outputs/all_strategies_pool.csv`
额外增加列：`strategy_size_before`, `strategy_size_after`

### 4. 全策略合并 CSV（原始格式）
路径：`outputs/all_strategies_pool_raw.csv`

## 【项目结构】
```
agent/产品数据加工/
├── build_pools.py          # 主脚本
├── raw_products.csv        # 输入数据
├── cursor_prompt.md        # 本说明文档
└── outputs/
    ├── by_strategy/        # 各策略带评分CSV
    │   ├── {strategy}.csv
    │   └── raw/            # 各策略原始格式CSV
    │       └── {strategy}.csv
    ├── all_strategies_pool.csv      # 合并带评分
    └── all_strategies_pool_raw.csv  # 合并原始格式
```

## 【运行命令】
```bash
cd agent/产品数据加工
python build_pools.py
```

## 【日志输出】
- 读取行数
- 原始列名
- 缺失指标分布统计
- 策略族分布
- 各策略过滤前后数量
- 最终总行数
- 输出文件路径

## 【技术要点】
- Python 3.9+ 兼容
- 依赖：numpy 1.26.4, pandas 2.2.3
- z-score 计算处理 std=0 情况（返回 0）
- 浮点数输出：指标 4 位小数，z-score/score 6 位小数
- CSV 编码：utf-8-sig（兼容 Excel 中文显示）
