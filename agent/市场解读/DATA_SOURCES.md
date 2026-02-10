# 市场行情数据来源

更新文件: `market_data_20260122.csv`

## 数据来源

### 美股 / 国际指数

| ticker | 名称 | 推荐来源 | 搜索关键词 | 备用来源 |
|--------|------|----------|-----------|----------|
| SPX | S&P 500 | Google Finance | `S&P 500` → .INX | Yahoo Finance (^GSPC) / TradingView (SPX) |
| IXIC | Nasdaq Composite | Google Finance | `Nasdaq Composite` → .IXIC | Yahoo Finance (^IXIC) / TradingView (IXIC) |
| NI225 | 日经225 | Google Finance | `Nikkei 225` → NI225 | Yahoo Finance (^N225) / TradingView (NI225) |
| HSI | 恒生指数 | Google Finance | `Hang Seng Index` → HSI | Yahoo Finance (^HSI) / TradingView (HSI) |

### A股指数

| ticker | 名称 | 推荐来源 | 页面路径 | 备用来源 |
|--------|------|----------|---------|----------|
| 000001.SH | 上证综指 | 东方财富 | quote.eastmoney.com/zs000001.html | 同花顺 / 上交所 (sse.com.cn) |
| 399001.SZ | 深证成指 | 东方财富 | quote.eastmoney.com/zs399001.html | 同花顺 / 深交所 (szse.cn) |
| 399006.SZ | 创业板指 | 东方财富 | quote.eastmoney.com/zs399006.html | 同花顺 / 深交所 (szse.cn) |

## CSV 格式

列定义:
```
asset_class,market,sector,ticker,ref_date,value,unit,data_type
```

示例行:
```csv
Equity,US,Broad_Market_Index,SPX,2026-02-06,6932.30,Index_Level,Equity
Equity,China_A,Broad_Market_Index,000001.SH,2026-02-06,4065.58,Index_Level,Equity
Equity,China_A,Growth_Style_Index,399006.SZ,2026-02-06,3236.46,Index_Level,Equity
```

字段说明:
- `asset_class`: Equity
- `market`: US / Japan / HongKong / China_A
- `sector`: Broad_Market_Index / Growth_Style_Index
- `ref_date`: YYYY-MM-DD 格式
- `value`: 指数点位 (收盘价)
- `unit`: Index_Level
- `data_type`: Equity

## 更新步骤

1. 确定更新日期 (ref_date)
2. 按下表逐项获取收盘价:

| # | ticker | 去哪查 |
|---|--------|--------|
| 1 | SPX | Google 搜 "S&P 500" |
| 2 | IXIC | Google 搜 "Nasdaq Composite" |
| 3 | NI225 | Google 搜 "Nikkei 225" |
| 4 | HSI | Google 搜 "Hang Seng Index" |
| 5 | 000001.SH | 东方财富搜 "上证综指" |
| 6 | 399001.SZ | 东方财富搜 "深证成指" |
| 7 | 399006.SZ | 东方财富搜 "创业板指" |

3. 每个 ticker 需要两行数据: 基准日期 + 更新日期
4. 复制 CSV 格式填入文件，注意保留表头行
