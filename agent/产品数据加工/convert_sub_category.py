# -*- coding: utf-8 -*-
"""
将 raw_products.csv 的 sub_category 映射到新的16子策略分类体系
"""
import pandas as pd
from typing import Dict

# 子策略映射表：旧分类 -> 新分类（16子策略标准口径）
SUB_CATEGORY_MAPPING: Dict[str, str] = {
    # 现金类（1个子策略）
    "货币基金": "现金类",
    "现金类-其他": "现金类",
    
    # 固收类（5个子策略：存款类固收、纯债、非标固收、固收+、海外债券投资）
    "纯债型": "纯债",
    "短债型": "纯债",
    "固收增强型": "固收+",
    "偏债混合型": "固收+",
    "二级债基": "固收+",
    "可转债型": "固收+",
    
    # 股票类（4个子策略：股债混合、股票型、海外股票投资、海外股债混合）
    "股票型": "股票型",
    "股票增强型": "股票型",
    "量化多因子": "股票型",
    "行业主题型": "股票型",
    
    # 另类（6个子策略：商品及宏观策略、量化对冲、房地产股权、PE/VC股权、海外另类、结构性产品）
    "商品型": "商品及宏观策略",
    "CTA策略": "商品及宏观策略",
    "市场中性": "量化对冲",
    "多策略对冲": "量化对冲",
}

# asset_class 也需要同步更新
ASSET_CLASS_MAPPING: Dict[str, str] = {
    "现金类": "现金类",
    "纯债": "固收类",
    "固收+": "固收类",
    "股票型": "股票类",
    "股债混合": "股票类",
    "商品及宏观策略": "另类",
    "量化对冲": "另类",
}


def convert_sub_category(input_path: str, output_path: str) -> None:
    """
    读取 CSV，转换 sub_category 和 asset_class，输出新 CSV
    """
    df = pd.read_csv(input_path)
    
    # 记录转换前的唯一值
    print("转换前 sub_category 唯一值:")
    print(df["sub_category"].value_counts())
    print()
    
    # 转换 sub_category
    df["sub_category"] = df["sub_category"].map(SUB_CATEGORY_MAPPING)
    
    # 检查是否有未映射的值
    unmapped = df[df["sub_category"].isna()]
    if len(unmapped) > 0:
        print("警告：以下行的 sub_category 未能映射:")
        print(unmapped[["product_name", "product_code"]])
    
    # 根据新的 sub_category 更新 asset_class
    df["asset_class"] = df["sub_category"].map(ASSET_CLASS_MAPPING)
    
    # 记录转换后的唯一值
    print("转换后 sub_category 唯一值:")
    print(df["sub_category"].value_counts())
    print()
    
    print("转换后 asset_class 唯一值:")
    print(df["asset_class"].value_counts())
    print()
    
    # 保存
    df.to_csv(output_path, index=False)
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    input_file = "raw_products.csv"
    output_file = "raw_products_converted.csv"
    convert_sub_category(input_file, output_file)

