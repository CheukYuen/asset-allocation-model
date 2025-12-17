#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 200 条 mock 产品数据
- 符合 cursor_prompt.md 的字段要求
- 0.00 改为 0
- 某些指标随机为空
- 兼容 Python 3.9
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random

# 设置随机种子确保可复现
np.random.seed(42)
random.seed(42)

# ===================== 配置区 =====================

# 16 个一级策略 (sub_category) 及其对应的资产大类和策略族
STRATEGY_CONFIG = {
    # 股票类 - A族 (权益进攻)
    "股票型": {"asset_class": "股票类", "family": "A", "risk_levels": ["R3", "R4", "R5"]},
    "股票增强型": {"asset_class": "股票类", "family": "A", "risk_levels": ["R4", "R5"]},
    "行业主题型": {"asset_class": "股票类", "family": "A", "risk_levels": ["R4", "R5"]},
    "量化多因子": {"asset_class": "股票类", "family": "A", "risk_levels": ["R3", "R4"]},
    
    # 固收+ - B族 (平衡增强)
    "固收增强型": {"asset_class": "固收+", "family": "B", "risk_levels": ["R2", "R3"]},
    "偏债混合型": {"asset_class": "固收+", "family": "B", "risk_levels": ["R2", "R3"]},
    "可转债型": {"asset_class": "固收+", "family": "B", "risk_levels": ["R3", "R4"]},
    "二级债基": {"asset_class": "固收+", "family": "B", "risk_levels": ["R2", "R3"]},
    
    # 现金类 - C族 (纯防守)
    "货币基金": {"asset_class": "现金类", "family": "C", "risk_levels": ["R1"]},
    "短债型": {"asset_class": "现金类", "family": "C", "risk_levels": ["R1", "R2"]},
    "纯债型": {"asset_class": "现金类", "family": "C", "risk_levels": ["R1", "R2"]},
    "现金类-其他": {"asset_class": "现金类", "family": "C", "risk_levels": ["R1"]},
    
    # 另类 - D族 (分散/另类)
    "商品型": {"asset_class": "另类", "family": "D", "risk_levels": ["R3", "R4", "R5"]},
    "CTA策略": {"asset_class": "另类", "family": "D", "risk_levels": ["R3", "R4"]},
    "市场中性": {"asset_class": "另类", "family": "D", "risk_levels": ["R2", "R3"]},
    "多策略对冲": {"asset_class": "另类", "family": "D", "risk_levels": ["R3", "R4"]},
}

# 每种策略的数量分布 (总计约200)
STRATEGY_COUNTS = {
    "股票型": 20,
    "股票增强型": 12,
    "行业主题型": 15,
    "量化多因子": 10,
    "固收增强型": 18,
    "偏债混合型": 15,
    "可转债型": 10,
    "二级债基": 12,
    "货币基金": 20,
    "短债型": 15,
    "纯债型": 18,
    "现金类-其他": 8,
    "商品型": 8,
    "CTA策略": 7,
    "市场中性": 6,
    "多策略对冲": 6,
}

# 各策略族的收益/波动/回撤/夏普参数范围
FAMILY_PARAMS = {
    "A": {  # 权益进攻 - 高收益高波动
        "return_1y": (0.05, 0.35),
        "return_3y": (0.06, 0.25),
        "return_5y": (0.05, 0.20),
        "volatility_1y": (0.12, 0.30),
        "volatility_3y": (0.14, 0.28),
        "volatility_5y": (0.15, 0.26),
        "max_drawdown_1y": (-0.35, -0.10),
        "max_drawdown_2y": (-0.45, -0.15),
        "max_drawdown_3y": (-0.50, -0.18),
        "sharpe_ratio_1y": (0.20, 1.20),
        "sharpe_ratio_3y": (0.30, 1.00),
        "sharpe_ratio_5y": (0.35, 0.90),
    },
    "B": {  # 平衡增强 - 中等收益中等波动
        "return_1y": (0.03, 0.15),
        "return_3y": (0.04, 0.12),
        "return_5y": (0.04, 0.10),
        "volatility_1y": (0.04, 0.12),
        "volatility_3y": (0.05, 0.11),
        "volatility_5y": (0.05, 0.10),
        "max_drawdown_1y": (-0.15, -0.03),
        "max_drawdown_2y": (-0.20, -0.05),
        "max_drawdown_3y": (-0.25, -0.06),
        "sharpe_ratio_1y": (0.50, 1.50),
        "sharpe_ratio_3y": (0.60, 1.30),
        "sharpe_ratio_5y": (0.55, 1.20),
    },
    "C": {  # 纯防守 - 低收益低波动
        "return_1y": (0.015, 0.045),
        "return_3y": (0.018, 0.040),
        "return_5y": (0.020, 0.038),
        "volatility_1y": (0, 0.02),
        "volatility_3y": (0, 0.018),
        "volatility_5y": (0, 0.015),
        "max_drawdown_1y": (-0.02, 0),
        "max_drawdown_2y": (-0.03, 0),
        "max_drawdown_3y": (-0.04, 0),
        "sharpe_ratio_1y": (1.50, 3.00),
        "sharpe_ratio_3y": (1.40, 2.80),
        "sharpe_ratio_5y": (1.30, 2.50),
    },
    "D": {  # 分散/另类 - 不相关收益
        "return_1y": (0.02, 0.20),
        "return_3y": (0.03, 0.15),
        "return_5y": (0.03, 0.12),
        "volatility_1y": (0.06, 0.20),
        "volatility_3y": (0.07, 0.18),
        "volatility_5y": (0.07, 0.16),
        "max_drawdown_1y": (-0.25, -0.05),
        "max_drawdown_2y": (-0.30, -0.08),
        "max_drawdown_3y": (-0.35, -0.10),
        "sharpe_ratio_1y": (0.30, 1.30),
        "sharpe_ratio_3y": (0.40, 1.10),
        "sharpe_ratio_5y": (0.45, 1.00),
    },
}

# 产品名称前缀模板
PRODUCT_NAME_TEMPLATES = {
    "股票型": ["华夏沪深300ETF", "易方达中证500", "南方创业板", "嘉实沪深300", "博时上证50", "广发中证1000", "富国沪深300", "汇添富中证800"],
    "股票增强型": ["景顺长城量化精选", "华泰柏瑞量化", "国泰量化增强", "银华量化优选", "兴全量化精选"],
    "行业主题型": ["中欧医疗健康", "招商白酒", "天弘新能源", "国联安半导体", "华宝科技龙头", "易方达消费精选"],
    "量化多因子": ["富国量化多因子", "华安量化优选", "长信量化先锋", "中金量化优选"],
    "固收增强型": ["易方达稳健收益", "招商产业债", "工银双利", "华夏回报", "嘉实增强收益"],
    "偏债混合型": ["广发稳健增长", "南方安心优选", "博时信用债", "中银稳健增利"],
    "可转债型": ["兴全可转债", "富国可转债", "长信可转债", "华宝可转债"],
    "二级债基": ["易方达安心债券", "招商安泰债券", "南方宝元债券", "博时稳定价值"],
    "货币基金": ["天弘余额宝", "易方达易理财", "南方现金增利", "华安日日鑫", "博时现金收益"],
    "短债型": ["嘉实超短债", "广发安泽短债", "易方达短债", "南方短债"],
    "纯债型": ["招商安瑞进取", "工银纯债", "易方达纯债", "华夏纯债", "博时稳定纯债"],
    "现金类-其他": ["余额宝", "零钱通", "理财通", "招财宝"],
    "商品型": ["华安黄金ETF", "国泰黄金ETF", "博时黄金", "易方达黄金ETF"],
    "CTA策略": ["九坤CTA优选", "明汯CTA精选", "灵均CTA策略", "幻方CTA"],
    "市场中性": ["中信中性策略", "九坤对冲精选", "明汯中性优选"],
    "多策略对冲": ["高毅多策略", "景林对冲精选", "淡水泉多策略"],
}


def generate_random_value(low: float, high: float) -> float:
    """生成随机值，使用正态分布让数据更自然"""
    mean = (low + high) / 2
    std = (high - low) / 4
    value = np.random.normal(mean, std)
    return max(low, min(high, value))


def format_value(value: float, is_zero_ok: bool = True, to_percent: bool = False) -> float:
    """
    格式化数值:
    - 保留合理精度
    - 0.00 改为 0
    - to_percent=True 时，将小数转为百分数制 (0.0925 -> 9.25)
    """
    if value is None:
        return None
    
    # 转换为百分数制
    if to_percent:
        value = value * 100
    
    rounded = round(value, 2)
    
    # 0.00 改为 0
    if is_zero_ok and abs(rounded) < 0.01:
        return 0
    
    return rounded


def should_be_missing(col_name: str) -> bool:
    """
    随机决定某个字段是否为空
    - 1y 指标缺失率: 2%
    - 3y 指标缺失率: 8%
    - 5y 指标缺失率: 15%
    """
    if "1y" in col_name:
        return random.random() < 0.02
    elif "3y" in col_name:
        return random.random() < 0.08
    elif "5y" in col_name:
        return random.random() < 0.15
    return False


def generate_product_code(strategy: str, idx: int) -> str:
    """生成产品代码"""
    prefix_map = {
        "股票型": "51",
        "股票增强型": "52",
        "行业主题型": "53",
        "量化多因子": "54",
        "固收增强型": "10",
        "偏债混合型": "11",
        "可转债型": "12",
        "二级债基": "13",
        "货币基金": "00",
        "短债型": "01",
        "纯债型": "02",
        "现金类-其他": "00",
        "商品型": "51",
        "CTA策略": "88",
        "市场中性": "89",
        "多策略对冲": "90",
    }
    prefix = prefix_map.get(strategy, "99")
    return f"{prefix}{idx:04d}"


def generate_product_name(strategy: str, idx: int) -> str:
    """生成产品名称"""
    templates = PRODUCT_NAME_TEMPLATES.get(strategy, [strategy])
    base_name = templates[idx % len(templates)]
    suffix = chr(65 + (idx // len(templates)))  # A, B, C, ...
    if idx >= len(templates):
        return f"{base_name}{suffix}"
    return base_name


def generate_products() -> pd.DataFrame:
    """生成所有产品数据"""
    products = []
    
    for strategy, count in STRATEGY_COUNTS.items():
        config = STRATEGY_CONFIG[strategy]
        family = config["family"]
        asset_class = config["asset_class"]
        risk_levels = config["risk_levels"]
        params = FAMILY_PARAMS[family]
        
        for i in range(count):
            product = {
                "product_name": generate_product_name(strategy, i),
                "product_code": generate_product_code(strategy, i),
                "currency": "CNY",
                "asset_class": asset_class,
                "sub_category": strategy,
                "risk_level": random.choice(risk_levels),
            }
            
            # 生成各项指标
            # 需要转换为百分数制的字段 (0.0925 -> 9.25)
            percent_cols = ["return_1y", "return_3y", "return_5y",
                           "volatility_1y", "volatility_3y", "volatility_5y",
                           "max_drawdown_1y", "max_drawdown_2y", "max_drawdown_3y"]
            
            for col in ["return_1y", "return_3y", "return_5y",
                       "volatility_1y", "volatility_3y", "volatility_5y",
                       "max_drawdown_1y", "max_drawdown_2y", "max_drawdown_3y",
                       "sharpe_ratio_1y", "sharpe_ratio_3y", "sharpe_ratio_5y"]:
                
                if should_be_missing(col):
                    product[col] = None
                else:
                    low, high = params[col]
                    value = generate_random_value(low, high)
                    # 收益/波动/回撤转换为百分数制，夏普比率保持原样
                    to_percent = col in percent_cols
                    product[col] = format_value(value, to_percent=to_percent)
            
            products.append(product)
    
    df = pd.DataFrame(products)
    
    # 打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """主函数"""
    print("开始生成 mock 数据...")
    
    df = generate_products()
    
    # 统计信息
    print(f"\n总产品数: {len(df)}")
    print(f"\n各策略分布:")
    print(df["sub_category"].value_counts().to_string())
    
    print(f"\n各资产大类分布:")
    print(df["asset_class"].value_counts().to_string())
    
    print(f"\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    if len(missing_stats) > 0:
        print(missing_stats.to_string())
    else:
        print("无缺失值")
    
    # 保存CSV
    output_path = "/Users/zlin/Developer/github/asset-allocation-model/agent/产品数据加工/raw_products.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n已保存到: {output_path}")
    print(f"\n前5行预览:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()

