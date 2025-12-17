你是资深量化工程师 + 数据工程师。请为我生成一套可运行的 Python（pandas/numpy）代码，用于“产品池过滤与导出”。

【目标】
从 raw_products.csv 读取约 1000 支产品，包含 16 种一级策略（用字段 sub_category 表示一级策略；如果你发现 sub_category 不是 16 策略字段，请在代码中以常量 STRATEGY_COL 统一配置，默认先用 sub_category）。
按“4 策略族 z-score 模板（不做 winsorize）”进行策略内评分与过滤，并输出：
1) 每个一级策略单独一个 CSV（共 16 份，文件名可用 outputs/by_strategy/{strategy_name}.csv）
2) 全部策略合并一个 CSV（outputs/all_strategies_pool.csv）

【输入 CSV 字段（存在冗余，但至少包含这些）】
product_name,product_code,currency,asset_class,sub_category,
return_1y,return_3y,return_5y,
volatility_1y,volatility_3y,volatility_5y,
max_drawdown_1y,max_drawdown_2y,max_drawdown_3y,
sharpe_ratio_1y,sharpe_ratio_3y,sharpe_ratio_5y,
risk_level

【核心指标选择（默认口径）】
- return = return_3y（如缺失则用 return_1y 或 return_5y 兜底，按 3y > 5y > 1y 优先级）
- volatility = volatility_1y（如缺失则 volatility_3y > volatility_5y）
- drawdown = max_drawdown_2y（如缺失则 max_drawdown_1y > max_drawdown_3y）
- sharpe = sharpe_ratio_3y（如缺失则 sharpe_ratio_1y > sharpe_ratio_5y）
注意：drawdown 可能是负数（-0.20），请统一转换为“回撤幅度 dd_mag=abs(drawdown)”（正数越大越差）。

【4 策略族 z-score 评分模板（不做 winsorize）】
统一 z-score 定义：在“策略内（同一 sub_category）”对各指标计算 z-score：
z_return, z_sharpe, z_vol, z_dd
综合分公式（统一形式）：
score = w_ret*z_return + w_sha*z_sharpe - w_vol*z_vol - w_dd*z_dd

四类权重（写在配置里）：
A 权益进攻（Equity Attack）：ret=0.35, sharpe=0.35, vol=0.15, dd=0.15
B 平衡增强/固收+（Balanced）：ret=0.20, sharpe=0.40, vol=0.15, dd=0.25
C 纯防守（Defensive FI/Cash）：ret=0.10, sharpe=0.45, vol=0.15, dd=0.30（若是现金类可 ret=0.05, sharpe=0.50, vol=0.20, dd=0.25）
D 分散/另类（Diversifier）：ret=0.25~0.30, sharpe=0.30~0.35, vol=0.20, dd=0.20（给出一个固定值，比如 ret=0.25, sharpe=0.35, vol=0.20, dd=0.20）

【16 一级策略 → 4 策略族映射】
由于我未在此提供 16 策略的精确名称，请实现一个可编辑的映射字典 STRATEGY_FAMILY_MAP：
- key: 一级策略名称（sub_category 的值）
- value: "A"/"B"/"C"/"D"
并提供一个 fallback 规则：
- asset_class == "股票类" -> A
- asset_class == "固收+" 或 sub_category 包含 "固收" -> B
- asset_class == "现金类" 或 sub_category 包含 "货币"/"短债"/"纯债" -> C
- sub_category 包含 "商品"/"CTA"/"对冲"/"另类" -> D
若仍无法判断，则默认 B，并在日志里输出 warning。

【过滤闸门（不极端，策略内）】
对每个一级策略（同一 sub_category）分别计算分位数并剔除尾部：
- 剔除波动最差 10%：volatility > quantile(0.90)
- 剔除回撤最差 10%：dd_mag > quantile(0.90)
- 剔除夏普最差 20%：sharpe < quantile(0.20)
（股票类可在配置里放宽到 0.85/0.85/0.20，但默认用上述统一值即可，写成可配置常量。）

【输出要求】
1) 每个策略 CSV：至少包含这些列（字段名统一）：
product_name,product_code,currency,asset_class,sub_category,risk_level,
metric_return,metric_volatility,metric_drawdown,metric_sharpe,
z_return,z_sharpe,z_vol,z_dd,score,
family
2) 全策略合并 CSV：把所有策略的过滤结果 concat 后输出，额外增加：
strategy_size_before, strategy_size_after（可按 sub_category groupby 统计后 merge 回每行）
3) 所有输出目录若不存在要自动创建。
4) 代码要有 main()，可直接 `python build_pools.py` 运行。
5) 代码要打印关键日志：读取行数、每策略过滤前后数量、最终总行数、输出文件路径。
6) 请把项目结构也给出建议（最小即可）：
- scripts/build_pools.py
- data/raw_products.csv
- outputs/by_strategy/*.csv
- outputs/all_strategies_pool.csv

【风格要求】
- 不要写占位符；必须给出完整可运行代码。
- pandas 计算 z-score 时要考虑 std=0 的情况（返回 0 避免 NaN）。
- 处理缺失值：核心指标缺失太多的行（四项中缺两项以上）直接剔除；缺一项按兜底口径补齐。

【重要数据口径约定（必须遵守）】
raw_products.csv 中，收益 / 波动 / 回撤字段的数值可能以“百分数制”存储（例如 9.25 表示 9.25%）。
在进入任何计算（z-score、分位数、评分、过滤）之前，必须统一将这些字段转换为“小数制”（9.25 → 0.0925）。
请在代码中实现一个通用的 normalize_percent_series 函数：
- 若某列 median(|x|) > 1.5，则整体除以 100；
- 否则保持不变。
夏普比率（sharpe_ratio_*）不参与该转换。


请直接输出：
1) build_pools.py 完整代码（一个文件即可）
2) 运行命令与输出文件说明
