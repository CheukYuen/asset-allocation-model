结论先说：这 4 个 Benchmark，**用 AkShare 基本都能搞定或有很接近的替代物**；yfinance 对 A 股只在「沪深300全收益」这一块比较好用，其余更多是用 ETF 做 proxy。

我给你整理成一个工程视角的总表 + 分项说明和代码思路，你可以直接丢给数据同学。

---

## 1. 总览：4 个 Benchmark 的可获取情况

| 大类 | 目标基准                     | AkShare 能否直接拿到                                                                                                              | yfinance 能否直接拿到                                                            | 推荐做法（免费方案）                                                                         |
| -- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 固收 | **中债综合全价指数 (CBCFPI)**    | ✅ 是，用 `bond_composite_index_cbond` 抓中国债券信息网数据 ([akshare.akfamily.xyz][1])                                                   | ❌ 没有 CBCFPI 本尊，只能用境外 China bond ETF 近似 ([Yahoo Finance][2])                | **固收 Benchmark 建议直接用 AkShare 的中债综合全价（或财富指数），必要时用境外 China Bond ETF 做 sanity check** |
| 股票 | **沪深300全收益（CSI 300 TR）** | ⚠️ 只方便拿到价格指数 000300，TR 不一定有官方接口                                                                                             | ✅ 有：`H00300.SS`（Total Return），`N00300.SS`（Net TR）历史数据 ([Yahoo Finance][3]) | **A 股权益 Benchmark：建议用 yfinance 上的 CSI300 TR；国内环境不便时用 000300 指数 + ETF 分红近似**        |
| 商品 | **南华商品指数 (NHCI)**        | ✅ 有：`futures_nh_return_index(symbol="NHCI")` / 价格指数接口，可获取完整历史 ([AKShare][4])                                                | ❌ 没有 NHCI 本身，只能找商品 ETF 近似                                                  | **商品 Benchmark：建议直接用 AkShare 南华商品指数 NHCI（收益率指数），或价格指数**                            |
| 现金 | **货币基金指数 / 全市场货基收益率**    | ✅ 有：申万宏源**货币基金指数**（807400），接口 `index_realtime_fund_sw` + `基金指数历史行情`；以及所有货基历史收益数据，可自己算「全市场货基收益率」 ([akshare.akfamily.xyz][5]) | ❌ 没有中国货基整体指数，顶多是个别货币基金或短债 ETF                                              | **现金 Benchmark：短期可以用 807400「申万货币基金指数」；长期可以用 AkShare 所有货基 7 日年化的均值 / 中位数自建指数**      |

---

## 2. 分项说明 + 代码思路

### 2.1 中债综合全价指数（固收 Benchmark）

**AkShare**

* 接口：`bond_composite_index_cbond`
* 数据源：中债指数官网 `single_index_query`，AkShare 已封装好 ([akshare.akfamily.xyz][1])
* 关键参数：

  * `indicator`：选择 `"全价"` 或 `"全价指数涨跌幅"` 等（视你需要是指数点位还是收益率）
  * `period`：选择 `"总值"`（全久期），也可以细分 1–3 年、3–5 年等

示意代码（固收 Benchmark 用「全价指数点位 → 日/月收益率」）：

```python
import akshare as ak
import pandas as pd
import numpy as np

# 1) 抓中债综合全价指数（总值）
df = ak.bond_composite_index_cbond(indicator="全价", period="总值")
# df: [date, value]

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# 2) 转换为日度对数收益率
df["ret_d"] = np.log(df["value"] / df["value"].shift(1))

# 3) 聚合成月度收益率（做 Benchmark 用）
ret_m = df["ret_d"].resample("M").sum()  # 日对数收益累加
```

**yfinance**

* 目前没看到 CBCFPI 的直接 ticker，只能用境外的 China Bond ETF，如：

  * `CBON`（VanEck China Bond ETF，美国上市）([Yahoo Finance][6])
  * 新加坡一些 China Bond ETF，如 `ZHS.SI` / `ZHD.SI` 等 ([Yahoo Finance][7])
* 这些更适合做 sanity check 或对外展示，不太适合作为「中债综合」的严格替代。

👉 **工程建议**：
回测 / Benchmark 计算优先用 AkShare 直接拉中债综合全价；若你将来做境外版本或需要多源验证，再补充 ETF 数据。

---

### 2.2 沪深300全收益（股票 Benchmark）

你要的是 **Total Return**，而 A 股公开接口通常只给价格指数 000300。这里 yfinance 反而是优势。

**yfinance**

* 关键 ticker：

  * `H00300.SS`：**CSI 300 Total Return Index** ([Yahoo Finance][3])
  * `N00300.SS`：**CSI 300 Net Total Return Index**（扣了预提税） ([Yahoo Finance][8])

示意代码（用 TR 指数）：

```python
import yfinance as yf
import numpy as np

ticker = yf.Ticker("H00300.SS")  # 或 N00300.SS
df = ticker.history(start="2010-01-01", auto_adjust=False)
# df.index 为日期，"Close" 为指数点位

df["ret_d"] = np.log(df["Close"] / df["Close"].shift(1))
ret_m = df["ret_d"].resample("M").sum()
```

**AkShare 替代方案**

* 可以直接拉 **CSI 300 价格指数**：

  * 如 `stock_zh_index_daily(symbol="sh000300")` 或中证指数接口（视 AkShare 版本） ([akshare.akfamily.xyz][9])
* 然后用 ETF 分红或中证指数的「股息率」近似 total return：

  * 简化：直接用价格指数作为 Benchmark（说明书里写清楚「未含红利」）
  * 稍复杂：用 510300 ETF 历史「复权净值」当作总收益代理（AkShare / yfinance 都能抓）

👉 **工程建议**：

* **如果允许连外网**：直接用 `H00300.SS` / `N00300.SS`。
* **如果只能用国内源**：用 000300 价格指数或 510300 复权净值，接受「略低估收益」的偏差，在 PRD / 内部说明里写清楚。

---

### 2.3 南华商品指数 NHCI（商品 Benchmark）

**AkShare**

* 接口：`futures_nh_return_index`（收益率指数） ([AKShare][4])
* `futures_nh_index_symbol_table()` 会告诉你所有可用代码，其中 `NHCI` 对应「南华商品指数」 ([Tencent Cloud][10])

示意代码：

```python
import akshare as ak
import numpy as np

# 1) 确认 NHCI 存在
symbols = ak.futures_nh_index_symbol_table()
# 过滤 name == "南华商品指数" -> code=NHCI

df = ak.futures_nh_return_index(symbol="NHCI")
# 一般会给出日期、指数点位/收益率等字段（视版本）

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# 假设有字段 index_close
df["ret_d"] = np.log(df["index_close"] / df["index_close"].shift(1))
ret_m = df["ret_d"].resample("M").sum()
```

**yfinance**

* 没有看到 NHCI 的直接 ticker，更多是南华期货（603093.SS）股票本身或相关 ETF，无法作为商品大类 Benchmark。([Yahoo Finance][11])

👉 **工程建议**：
商品大类 Benchmark 完全可以用 AkShare 的 NHCI。
NHCI 的研究报告还给出了年化收益、波动、Sharpe 等，可以用来 sanity check 自己算的结果。([Tencent Cloud][12])

---

### 2.4 货币基金指数 / 全市场货基收益率（现金 Benchmark）

这里有两条路径：

#### 路径 A：直接用「申万货币基金指数」（指数级 Benchmark）

**AkShare**

* 指数数据模块里的「申万宏源基金指数」一节：

  * 接口：`index_realtime_fund_sw(symbol="基础一级")`
  * 返回一系列基金指数，其中：

    * `指数代码=807400`，`指数名称=申万宏源货币基金指数` ([akshare.akfamily.xyz][5])
* 对应还有「基金指数历史行情」接口（文档里有），可以拉时间序列。

工程思路：

```python
import akshare as ak
import pandas as pd

# 1) 实时看一下有哪些基金指数
rt = ak.index_realtime_fund_sw(symbol="基础一级")
# 过滤 row['指数名称'] == "申万宏源货币基金指数" -> code 807400

# 2) 用“基金指数历史行情”接口拉 807400 的历史收盘
hist = ak.index_hist_fund_sw(symbol="807400")  # 名字可能略有差异，需以文档为准
# 然后同样转成日收益 / 月收益即可
```

优点：

* 做现金 Benchmark 非常省事，一条指数搞定，指数本身就是「全市场货基」的代表。

缺点：

* 指数口径由申万宏源定义，不一定和「你们内部货基池」完全一致。

#### 路径 B：自己聚合所有货币基金（自建「全市场货基收益率」）

**AkShare**

* 公募基金数据模块有：

  * 「货币型基金-实时数据」：`fund_money_fund_daily_em`（包含 7 日年化、万份收益等） ([AKShare][13])
  * 对应历史数据接口（同一文档页里有），可以按基金收集历史 7 日年化。

工程思路（简化版）：

1. 每个交易日：

   * 抓所有货基的 7 日年化收益率（或万份收益）
   * 过滤规模太小 / 刚成立的基金
   * 取「资产规模加权平均」或「中位数」
2. 得到日度「全市场货基收益率」，再转成年化/月度 Benchmark。

示意伪代码：

```python
import akshare as ak
import pandas as pd
import numpy as np

# 假设 fund_money_hist_df 是你整理好的 [date, fund_code, seven_day_yield] 面板数据
# 这里略过抓取环节，只写聚合逻辑

def build_money_mkt_index(panel_df):
    # 每日按中位数聚合
    daily = (
        panel_df
        .dropna(subset=["seven_day_yield"])
        .groupby("date")["seven_day_yield"]
        .median()
        .to_frame("mm_7d_annual")
    )
    # 近似日收益率 = 7日年化 / 365
    daily["ret_d"] = daily["mm_7d_annual"] / 100 / 365.0
    # 转月度对数收益
    daily["log_ret"] = np.log1p(daily["ret_d"])
    ret_m = daily["log_ret"].resample("M").sum()
    return ret_m
```

👉 **工程建议：**

* **MVP 阶段**：直接用 **807400 申万货币基金指数** 做现金 Benchmark，简单可靠。
* **未来想和「内部货基池」更强绑定**：再迭代到路径 B，自建「内部货基中位收益率」指数。

---

## 3. 小结：从「能抓到」到「能用作 Benchmark」

结合你前面搭的 Benchmark 体系，这 4 条数据的**可行免费组合**可以是：

1. **权益大类 Benchmark**

   * 主：CSI 300 Total Return（`H00300.SS` or `N00300.SS`）
   * 备：000300 价格指数 / 510300 复权净值（国内源）

2. **固收大类 Benchmark**

   * 主：中债综合全价 / 财富指数（日度 → 月度）

3. **商品大类 Benchmark**

   * 主：南华商品指数 NHCI（收益率指数）

4. **现金大类 Benchmark**

   * 主：807400 申万货币基金指数
   * 备：全市场货基 7 日年化的中位数 / 加权平均（自建）

如果你愿意，我可以下一步帮你：

* 直接写一个 `build_benchmark_curves.py` 的完整脚本 Prompt，把上面 4 条曲线全部拉好、聚合成月度收益 csv，接入你前面那套 C1–C5 组合 Benchmark。

[1]: https://akshare.akfamily.xyz/data/bond/bond.html?utm_source=chatgpt.com "AKShare 债券数据"
[2]: https://sg.finance.yahoo.com/quote/%5ECHNB-NV/?utm_source=chatgpt.com "Global X GF China Bond ETF (^CHNB-NV)"
[3]: https://finance.yahoo.com/quote/H00300.SS/?utm_source=chatgpt.com "CSI 300 Total Return Index (H00300.SS)"
[4]: https://akshare-hh.readthedocs.io/en/latest/data/futures/futures.html?utm_source=chatgpt.com "AKShare 期货数据"
[5]: https://akshare.akfamily.xyz/data/index/index.html?utm_source=chatgpt.com "A股股票指数- 实时行情数据 - AKShare"
[6]: https://finance.yahoo.com/quote/CBON/?utm_source=chatgpt.com "VanEck China Bond ETF (CBON) Stock Price, News, Quote ..."
[7]: https://finance.yahoo.com/quote/ZHS.SI/?utm_source=chatgpt.com "Amova-ICBCSG China Bond Index ETF SGD (ZHS.SI)"
[8]: https://finance.yahoo.com/quote/N00300.SS/history/?utm_source=chatgpt.com "CSI 300 Net Total Return Index (N00300.SS) Historical Data"
[9]: https://akshare.akfamily.xyz/data/index/index.html "AKShare 指数数据 — AKShare 1.17.93 文档"
[10]: https://cloud.tencent.com/developer/article/1926590?utm_source=chatgpt.com "AKShare-期货数据-南华指数"
[11]: https://finance.yahoo.com/quote/603093.SS/?utm_source=chatgpt.com "Nanhua Futures Co., Ltd. (603093.SS)"
[12]: https://cloud.tencent.com/developer/article/1983969?utm_source=chatgpt.com "AKShare-期货数据-板块指数涨跌"
[13]: https://akshare-hh.readthedocs.io/en/latest/data/fund/fund_public.html?utm_source=chatgpt.com "AKShare 公募基金数据"
