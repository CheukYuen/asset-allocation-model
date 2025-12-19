整体目标是：**从约 1200 只产品中，构建 5 个“质量可控、风格分散、体验一致”的候选桶，用于资产配置阶段的随机抽取。**

---

# 产品池筛选与分桶规则说明

## 一、整体设计目标（先把“为什么”说清楚）

本机制服务于**资产配置阶段的产品候选池构建**，而不是直接做“最优产品排名”。
核心目标分为四点：

1. **保证产品质量底线**：剔除明显弱势、风险收益比失衡的产品
2. **保证分布公平与多样性**：避免用户因为随机性拿到“明显更差的一桶”
3. **保留头部 Alpha 暴露**：每个桶都能接触到收益最亮眼的产品
4. **满足策略覆盖完整性**：16 种一级策略在整体池中都有代表

**输入规模**：约 1200 只产品
**输出结果**：5 个产品桶（Bucket 1–5）
**使用方式**：资产配置阶段，用户随机获取 1 个桶作为候选产品池

---

## 二、Stage A：强 Alpha 产品识别（亮点池）

### 🎯 目标

提前识别**“绝对收益表现最突出的产品”**，作为后续分桶的公共增强因子，而不是让它们只集中在某几个桶里。

### 📌 规则说明

* 在全量产品池（≈1200 只）中
* 按 **1Y 收益率（return_1y）** 指标排序
* 选取 **收益率 Top 10 的产品**

### 📦 产出

* `Top_Return_Set`（收益最亮眼产品集合）
* 该集合在 Stage B 的后续步骤中 **会被强制注入到每个桶中**

> 说明：
> Stage A 不做任何风控或多样性约束，目的**不是公平，而是识别"最强信号"**。

#### 🔧 代码实现

```python
def identify_top_alpha(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    识别收益率 Top N 的产品
    按 return_1y 降序排列，取前 N 个
    """
    sorted_df = df.sort_values('return_1y', ascending=False)
    top_alpha = sorted_df.head(top_n).copy()
    return top_alpha
```

---

## 三、Stage B：结构化过滤 + 分桶 + 再增强

Stage B 是核心阶段，负责把“可用产品”组织成 **5 个质量接近、结构可控的桶**。

---

### B1. 分位数计算方式（按一级策略分组）

* 按 **16 个一级策略**分别分组
* 在每个策略组内，分别对以下指标计算**横截面分位数（Percentile）**：

  * 收益率（return_3y）
  * 波动率（volatility_3y）
  * 夏普比率（sharpe_ratio_3y）

对策略 $s$ 下产品 $i$ 的指标 $x$，其分位数定义为：

$$
P_{i,s}^{(x)} = \frac{\#\{\, j \in s \mid x_{j,s} \le x_{i,s} \,\}}{N_s}
$$

其中：

* $P_{i,s}^{(x)} \in [0,1]$：表示该产品在**策略 $s$** 内该指标的相对位置
* $N_s$：策略 $s$ 下产品数量
* 分位数越接近 1，表示该指标在策略内越高（或越差，取决于指标含义）

> 约定说明：
>
> * 对 **收益率 / 夏普比率**：分位数越高 → 表现越好
> * 对 **波动率**：分位数越高 → 波动越大、风险越高（表现越差）

#### 🔧 代码实现

```python
def calculate_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    按一级策略（sub_category）分组计算分位数
    使用 pandas rank(pct=True) 实现 P(x) = rank(x) / N
    """
    result = df.copy()
    
    # 按策略分组计算分位数
    result['pct_return_3y'] = df.groupby('sub_category')['return_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    result['pct_sharpe_3y'] = df.groupby('sub_category')['sharpe_ratio_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    result['pct_volatility_3y'] = df.groupby('sub_category')['volatility_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    return result
```

---

### B2. 剔除规则（联合逻辑 · 分位数版本）

在每个一级策略内部，若产品满足以下**联合剔除逻辑中的任一条件**，则被过滤出候选池：

1. **收益率分位数处于该策略内最差 10%**

   $$
   P_{i,s}^{(\text{return\_3y})} \leq 0.10
   $$

2. **或 夏普比率分位数处于该策略内最差 10%**

   $$
   P_{i,s}^{(\text{sharpe\_ratio\_3y})} \leq 0.10
   $$

3. **或（波动率分位数处于最差/最高 10%，且收益率低于策略中位数）**

   $$
   P_{i,s}^{(\text{volatility\_3y})} \geq 0.90 \quad \text{且} \quad P_{i,s}^{(\text{return\_3y})} < 0.50
   $$

---

### 说明

* 所有判断均基于**一级策略内部的相对位置（分位数）**，不使用绝对数值或跨策略比较
* 该联合逻辑实现了：

  * 对“**高波动但高收益**”产品的容忍（不误伤进攻型产品）
  * 对“**高波动且收益偏弱**”产品的有效剔除
  * 对“**低风险但收益一般**”产品的保留（防守型底仓）
* 分位数版本天然鲁棒，对极端值不敏感，适合用于稳定的规则化筛选

#### 🔧 代码实现

```python
# 阈值常量
FILTER_BOTTOM_PERCENTILE = 0.10  # 最差 10%
FILTER_TOP_VOL_PERCENTILE = 0.90  # 波动率最高 10%
FILTER_RETURN_MEDIAN = 0.50      # 收益中位数

def apply_filter_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    应用剔除规则（联合 OR 逻辑）
    """
    # 剔除条件
    cond1 = df['pct_return_3y'] <= FILTER_BOTTOM_PERCENTILE
    cond2 = df['pct_sharpe_3y'] <= FILTER_BOTTOM_PERCENTILE
    cond3 = (df['pct_volatility_3y'] >= FILTER_TOP_VOL_PERCENTILE) & \
            (df['pct_return_3y'] < FILTER_RETURN_MEDIAN)
    
    # 满足任一条件则剔除
    exclude_mask = cond1 | cond2 | cond3
    
    filtered_pool = df[~exclude_mask].copy()
    excluded_df = df[exclude_mask].copy()
    
    return filtered_pool, excluded_df
```

---

### 📦 产出

* `Filtered_Pool`：通过策略内质量过滤闸门的产品集合

---

## Stage C：分桶、桶内优选与再注入

---

### C1. 收益排序的均匀分桶（5 桶）

#### 规则

1. 对 `Filtered_Pool` 按 **1Y 收益率（return_1y）从高到低排序**
2. 顺序编号：1, 2, 3, …, N
3. 采用轮询方式分为 5 个桶：

* Bucket 1：1, 6, 11, 16, …
* Bucket 2：2, 7, 12, 17, …
* Bucket 3：3, 8, 13, 18, …
* Bucket 4：4, 9, 14, 19, …
* Bucket 5：5, 10, 15, 20, …

#### 🔧 代码实现

```python
def assign_buckets(df: pd.DataFrame, num_buckets: int = 5) -> pd.DataFrame:
    """
    C1: 按 return_1y 降序排序，轮询分配到各桶
    """
    result = df.sort_values('return_1y', ascending=False).copy()
    result = result.reset_index(drop=True)
    
    # 轮询分配 bucket_id (1-5)
    # index: 0,1,2,3,4,5,6,7...
    # bucket: 1,2,3,4,5,1,2,3...
    result['bucket_id'] = (result.index % num_buckets) + 1
    
    return result
```

---

### C2. 桶内多维优选（分位数 + OR）

在每个桶内，保留满足以下任一条件（OR）的产品：

* **收益率处于桶内 Top 20% 分位（return_3y）**
* **夏普比率处于桶内 Top 20% 分位（sharpe_ratio_3y）**
* **波动率处于桶内最优 Top 20% 分位（低波动，volatility_3y）**

#### 🔧 代码实现

```python
BUCKET_TOP_PERCENTILE = 0.80  # Top 20% (即分位数 >= 0.80)

def bucket_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    C2: 桶内多维优选
    保留满足任一条件的产品（OR 逻辑）
    """
    result = df.copy()
    
    # 计算桶内分位数
    result['bucket_pct_return'] = df.groupby('bucket_id')['return_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    result['bucket_pct_sharpe'] = df.groupby('bucket_id')['sharpe_ratio_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    result['bucket_pct_volatility'] = df.groupby('bucket_id')['volatility_3y'].transform(
        lambda x: x.rank(pct=True, method='average')
    )
    
    # 保留条件（OR 逻辑）
    keep_return = result['bucket_pct_return'] >= BUCKET_TOP_PERCENTILE      # 收益 Top 20%
    keep_sharpe = result['bucket_pct_sharpe'] >= BUCKET_TOP_PERCENTILE      # 夏普 Top 20%
    keep_low_vol = result['bucket_pct_volatility'] <= 0.20                  # 波动率最低 20%
    
    keep_mask = keep_return | keep_sharpe | keep_low_vol
    selected = result[keep_mask].copy()
    
    return selected
```

---

### C3. 一级策略覆盖约束

* 在整体保留集合中：

  * **16 种一级策略均至少保留 1 个产品**
* 若某一级策略缺失：

  * 从该策略中按 **return_3y 排序的名次 rank（从 1 开始）** 产品补充进入对应桶

#### 🔧 代码实现

```python
def ensure_strategy_coverage(
    selected_df: pd.DataFrame,
    full_pool: pd.DataFrame,
    all_strategies: List[str],
    num_buckets: int = 5
) -> pd.DataFrame:
    """
    C3: 确保16种一级策略在每个桶中都有代表
    """
    result = selected_df.copy()
    all_strategies_set = set(all_strategies)
    
    # 1. 预计算：每个策略的 return_3y 最优产品
    best_by_strategy = {
        s: full_pool[full_pool['sub_category'] == s].nlargest(1, 'return_3y')
        for s in all_strategies
        if len(full_pool[full_pool['sub_category'] == s]) > 0
    }
    
    # 2. 逐桶补充缺失策略（收集待添加行，最后批量合并）
    rows_to_add = []
    
    for bucket_id in range(1, num_buckets + 1):
        bucket_mask = result['bucket_id'] == bucket_id
        existing_strategies = set(result.loc[bucket_mask, 'sub_category'])
        existing_products = set(result.loc[bucket_mask, 'product_code'])
        missing = all_strategies_set - existing_strategies
        
        for strategy in missing:
            if strategy not in best_by_strategy:
                continue
            
            best = best_by_strategy[strategy]
            code = best.iloc[0]['product_code']
            
            if code not in existing_products:
                row = best.copy()
                row['bucket_id'] = bucket_id
                rows_to_add.append(row)
                existing_products.add(code)
    
    # 3. 批量合并（单次 concat，性能更优）
    if rows_to_add:
        result = pd.concat([result] + rows_to_add, ignore_index=True)
    
    return result
```

---

### C4. 强 Alpha 再注入（去重）

* 将 `Top_Return_Set` 中的产品加入 **每一个桶**
* 若产品已存在于桶中，则跳过，不重复添加

#### 🔧 代码实现

```python
def inject_top_alpha(
    buckets_df: pd.DataFrame,
    top_alpha_df: pd.DataFrame,
    num_buckets: int = 5
) -> pd.DataFrame:
    """
    C4: 将强 Alpha 产品注入每个桶（去重）
    """
    result = buckets_df.copy()
    
    # 标记已存在的 Top Alpha
    result['is_top_alpha'] = result['product_code'].isin(top_alpha_df['product_code'])
    
    for bucket_id in range(1, num_buckets + 1):
        bucket_products = set(result[result['bucket_id'] == bucket_id]['product_code'])
        
        for _, alpha_row in top_alpha_df.iterrows():
            if alpha_row['product_code'] not in bucket_products:
                # 添加到该桶
                new_row = alpha_row.copy()
                new_row['bucket_id'] = bucket_id
                new_row['is_top_alpha'] = True
                result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                bucket_products.add(alpha_row['product_code'])
    
    return result
```

---

#### 📦 最终结果

* 每个桶同时具备：

  * 均衡的收益结构
  * 多维风格代表
  * 明确的头部 Alpha 暴露

---

## 四、最终效果总结（一句话版本）

> 从 1200 只产品中，通过 **质量闸门 → 收益同构分桶 → 多维优选 → 强 Alpha 注入**，
> 构建 5 个 **统计性质一致、体验公平、风格多样、但都不失进攻性的产品桶**，
> 支撑资产配置阶段的随机抽桶机制，而不牺牲专业性与稳定性。

---

## 五、完整流程代码

```python
def run_bucket_filter(input_file: str, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    执行完整的分桶流程
    """
    # 1. 加载与清洗数据（剔除关键指标缺失的产品）
    cleaned_df, removed_df = load_and_clean_data(input_file)
    
    # 获取所有策略类型
    all_strategies = cleaned_df['sub_category'].unique().tolist()
    
    # 2. Stage A: 识别 Top Alpha（return_1y Top 10）
    top_alpha = identify_top_alpha(cleaned_df)
    
    # 3. Stage B: 分位数计算与过滤
    df_with_pct = calculate_percentiles(cleaned_df)
    filtered_pool, excluded = apply_filter_rules(df_with_pct)
    
    # 4. Stage C1: 轮询分桶
    bucketed = assign_buckets(filtered_pool)
    
    # 5. Stage C2: 桶内优选
    selected = bucket_selection(bucketed)
    
    # 6. Stage C3: 策略覆盖
    with_coverage = ensure_strategy_coverage(selected, filtered_pool, all_strategies)
    
    # 7. Stage C4: Alpha 注入
    final_buckets = inject_top_alpha(with_coverage, top_alpha)
    
    return {
        'top_alpha': top_alpha,
        'filtered_pool': filtered_pool,
        'final_buckets': final_buckets,
        'excluded': excluded
    }
```

### 流程图

```
原始产品池 (≈1200)
    │
    ▼
┌─────────────────────────────────────┐
│  Stage A: 识别 Top 10 Alpha         │
│  (按 return_1y 排序取前 10)         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage B1: 按策略分组计算分位数      │
│  (return_3y, volatility_3y, sharpe) │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage B2: 剔除低质量产品            │
│  - 收益最差 10%                     │
│  - 夏普最差 10%                     │
│  - 高波动 + 收益偏弱                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage C1: 轮询分桶 (5 桶)           │
│  按 return_1y 排序后轮询分配         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage C2: 桶内多维优选              │
│  保留 Top 20% 收益/夏普/低波动       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage C3: 策略覆盖补充              │
│  确保 16 种策略都有代表              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage C4: Top Alpha 注入           │
│  每个桶都包含 Top 10 Alpha 产品      │
└─────────────────────────────────────┘
    │
    ▼
  5 个产品桶 (Bucket 1-5)
```

---

## 六、脚本使用说明

完整实现代码见 `bucket_filter.py`，运行方式：

```bash
python bucket_filter.py
```

输出文件结构：

```
outputs/
├── top_return_set.csv              # Top 10 强 Alpha 产品
├── filtered_pool.csv               # 过滤后候选池
├── bucket_1.csv ~ bucket_5.csv     # 含分桶元信息
└── raw_format/
    └── bucket_1_raw.csv ~ bucket_5_raw.csv  # 原始格式（与输入一致）
```
