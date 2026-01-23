做资产配置方案， 4大类是： 现金，权益，固收，另类资产。 生成105套只有大类资产比例的方案。
基于以下3个纬度的组合。 
纬度如下：
1. 7 个人生阶段： 刚毕业，单身青年，二人世界， 小孩学前，小孩成年前,  子女成年，退休
2. 3 种需求：保值、增值、传承
3. 5 个风险等级：C1 ~C5

## 年化波动率向量（σ）
```csv
asset_class,sigma_ann
BOND,0.0201
CASH,0.0032
COMMODITY,0.1013
EQUITY,0.2163
```

## 年化协方差矩阵（Σ）
```csv
asset,bond_cn_composite_fullprice_return,cash_cn_mmf_return,commodity_cn_nhci_return,equity_cn_csi300_return
bond_cn_composite_fullprice_return,0.000403,0.000004,-0.000575,-0.000676
cash_cn_mmf_return,0.000004,0.000010,-0.000024,0.000059
commodity_cn_nhci_return,-0.000575,-0.000024,0.010262,0.002503
equity_cn_csi300_return,-0.000676,0.000059,0.002503,0.046798
```

## 长期相关性矩阵（ρ）
```csv
asset,bond_cn_composite_fullprice_return,cash_cn_mmf_return,commodity_cn_nhci_return,equity_cn_csi300_return
bond_cn_composite_fullprice_return,1.000,0.062,-0.282,-0.155
cash_cn_mmf_return,0.062,1.000,-0.074,0.085
commodity_cn_nhci_return,-0.282,-0.074,1.000,0.114
equity_cn_csi300_return,-0.155,0.085,0.114,1.000
```