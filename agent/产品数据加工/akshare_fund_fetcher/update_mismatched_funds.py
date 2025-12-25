#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新验证中发现数据不匹配的基金
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fetch_fund_data import get_fund_info
import csv

def update_mismatched_funds():
    """更新数据不匹配的基金"""
    # 需要更新的基金代码列表
    fund_codes_to_update = ['510500', '159915', '161725']
    csv_file = "真实产品数据v1.csv"
    
    print("=" * 80)
    print("更新数据不匹配的基金")
    print("=" * 80)
    print(f"需要更新的基金: {', '.join(fund_codes_to_update)}\n")
    
    # 读取 CSV 文件
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    updated_count = 0
    
    for fund_code in fund_codes_to_update:
        print(f"\n正在更新基金 {fund_code}...")
        print("-" * 80)
        
        # 获取最新数据
        fund_data = get_fund_info(fund_code)
        
        if not fund_data:
            print(f"  ✗ 无法获取基金 {fund_code} 的数据")
            continue
        
        # 找到对应的行并更新
        for i, row in enumerate(rows):
            if row.get('product_code') == fund_code:
                old_name = row.get('product_name')
                old_return_1y = row.get('return_1y')
                old_max_dd_3y = row.get('max_drawdown_3y')
                
                print(f"  原数据:")
                print(f"    名称: {old_name}")
                print(f"    1年收益率: {old_return_1y}")
                print(f"    3年最大回撤: {old_max_dd_3y}")
                
                # 更新所有字段
                for key in fieldnames:
                    if key in fund_data:
                        row[key] = fund_data[key]
                
                # 保持原有基金名称（如果akshare获取的名称不完整）
                if old_name and not old_name.startswith('基金'):
                    row['product_name'] = old_name
                
                print(f"  新数据:")
                print(f"    名称: {row.get('product_name')}")
                print(f"    1年收益率: {row.get('return_1y')}")
                print(f"    3年最大回撤: {row.get('max_drawdown_3y')}")
                print(f"  ✓ 已更新")
                
                updated_count += 1
                break
    
    if updated_count > 0:
        # 保存更新后的 CSV
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n" + "=" * 80)
        print(f"✓ 成功更新 {updated_count} 个基金的数据")
        print(f"✓ CSV 文件已保存: {csv_file}")
    else:
        print(f"\n✗ 没有基金被更新")
    
    return updated_count


if __name__ == "__main__":
    updated = update_mismatched_funds()
    sys.exit(0 if updated > 0 else 1)

