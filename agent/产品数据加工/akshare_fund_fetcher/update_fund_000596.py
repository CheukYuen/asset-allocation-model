#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新基金代码 000961 为正确的 000596，并重新获取数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fetch_fund_data import get_fund_info
import csv

def update_fund_in_csv():
    """更新 CSV 文件中的基金信息"""
    fund_code = "000596"
    csv_file = "真实产品数据v1.csv"
    
    print(f"正在获取基金 {fund_code} 的数据...")
    print("=" * 60)
    
    # 获取基金数据
    fund_data = get_fund_info(fund_code)
    
    if not fund_data:
        print(f"\n✗ 获取基金 {fund_code} 数据失败")
        print("请检查:")
        print("1. akshare 是否已安装")
        print("2. 网络连接是否正常")
        print("3. 基金代码是否正确")
        return False
    
    print(f"\n✓ 成功获取数据:")
    print(f"  基金名称: {fund_data.get('product_name', 'N/A')}")
    print(f"  基金代码: {fund_data.get('product_code', 'N/A')}")
    print(f"  资产类别: {fund_data.get('asset_class', 'N/A')}")
    print(f"  子类别: {fund_data.get('sub_category', 'N/A')}")
    print(f"  1年收益率: {fund_data.get('return_1y', 0):.4f}")
    print(f"  3年收益率: {fund_data.get('return_3y', 0):.4f}")
    print(f"  3年最大回撤: {fund_data.get('max_drawdown_3y', 0):.4f}")
    print(f"  风险等级: {fund_data.get('risk_level', 'N/A')}")
    
    # 读取现有 CSV
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # 找到 000596 或 000961 的行并替换
    updated = False
    for i, row in enumerate(rows):
        if row.get('product_code') in ['000596', '000961']:
            print(f"\n找到需要更新的行 (第 {i+2} 行):")
            print(f"  原基金名称: {row.get('product_name')}")
            print(f"  原基金代码: {row.get('product_code')}")
            print(f"  原3年最大回撤: {row.get('max_drawdown_3y')}")
            print(f"  原1年收益率: {row.get('return_1y')}")
            print(f"  原3年收益率: {row.get('return_3y')}")
            
            # 更新数据
            for key in fieldnames:
                if key in fund_data:
                    row[key] = fund_data[key]
            
            print(f"\n✓ 已更新为:")
            print(f"  新基金名称: {row.get('product_name')}")
            print(f"  新基金代码: {row.get('product_code')}")
            print(f"  新3年最大回撤: {row.get('max_drawdown_3y')}")
            print(f"  新1年收益率: {row.get('return_1y')}")
            print(f"  新3年收益率: {row.get('return_3y')}")
            updated = True
            break
    
    if updated:
        # 保存更新后的 CSV
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✓ CSV 文件已更新: {csv_file}")
        return True
    else:
        print(f"\n✗ 未找到基金代码 000961 的记录")
        return False


if __name__ == "__main__":
    success = update_fund_in_csv()
    sys.exit(0 if success else 1)

