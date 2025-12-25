#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 CSV 文件中所有基金产品是否使用 akshare 真实数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fetch_fund_data import get_fund_info
import csv
import pandas as pd

def verify_all_funds():
    """验证所有基金数据"""
    csv_file = "真实产品数据v1.csv"
    
    # 读取 CSV 文件
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print("=" * 80)
    print("开始验证所有基金产品数据")
    print("=" * 80)
    print(f"共 {len(rows)} 个基金产品需要验证\n")
    
    results = []
    
    for i, row in enumerate(rows, 1):
        fund_code = row.get('product_code', '')
        fund_name = row.get('product_name', '')
        
        print(f"[{i}/{len(rows)}] 验证基金: {fund_name} ({fund_code})")
        print("-" * 80)
        
        # 从 akshare 获取最新数据
        try:
            fund_data = get_fund_info(fund_code)
            
            if not fund_data:
                print(f"  ✗ 无法从 akshare 获取数据")
                results.append({
                    'fund_code': fund_code,
                    'fund_name': fund_name,
                    'status': '无法获取',
                    'match': False,
                    'details': 'akshare 无法获取该基金数据'
                })
                print()
                continue
            
            # 对比关键指标
            csv_return_1y = float(row.get('return_1y', 0) or 0)
            csv_return_3y = float(row.get('return_3y', 0) or 0)
            csv_max_dd_3y = float(row.get('max_drawdown_3y', 0) or 0)
            
            ak_return_1y = fund_data.get('return_1y', 0)
            ak_return_3y = fund_data.get('return_3y', 0)
            ak_max_dd_3y = fund_data.get('max_drawdown_3y', 0)
            
            # 计算差异（允许小的浮点误差）
            diff_return_1y = abs(csv_return_1y - ak_return_1y)
            diff_return_3y = abs(csv_return_3y - ak_return_3y)
            diff_max_dd_3y = abs(csv_max_dd_3y - ak_max_dd_3y)
            
            tolerance = 0.01  # 允许1%的误差
            
            match_return_1y = diff_return_1y <= tolerance
            match_return_3y = diff_return_3y <= tolerance
            match_max_dd_3y = diff_max_dd_3y <= tolerance
            
            all_match = match_return_1y and match_return_3y and match_max_dd_3y
            
            # 显示对比结果
            print(f"  1年收益率:")
            print(f"    CSV:  {csv_return_1y:.4f}")
            print(f"    AK:   {ak_return_1y:.4f}")
            print(f"    差异: {diff_return_1y:.4f} {'✓' if match_return_1y else '✗'}")
            
            print(f"  3年收益率:")
            print(f"    CSV:  {csv_return_3y:.4f}")
            print(f"    AK:   {ak_return_3y:.4f}")
            print(f"    差异: {diff_return_3y:.4f} {'✓' if match_return_3y else '✗'}")
            
            print(f"  3年最大回撤:")
            print(f"    CSV:  {csv_max_dd_3y:.4f}")
            print(f"    AK:   {ak_max_dd_3y:.4f}")
            print(f"    差异: {diff_max_dd_3y:.4f} {'✓' if match_max_dd_3y else '✗'}")
            
            # 其他指标对比
            csv_vol_1y = float(row.get('volatility_1y', 0) or 0)
            ak_vol_1y = fund_data.get('volatility_1y', 0)
            diff_vol_1y = abs(csv_vol_1y - ak_vol_1y)
            match_vol_1y = diff_vol_1y <= tolerance
            
            csv_sharpe_1y = float(row.get('sharpe_ratio_1y', 0) or 0)
            ak_sharpe_1y = fund_data.get('sharpe_ratio_1y', 0)
            diff_sharpe_1y = abs(csv_sharpe_1y - ak_sharpe_1y)
            match_sharpe_1y = diff_sharpe_1y <= 0.1  # 夏普比率允许0.1的误差
            
            print(f"  1年波动率:")
            print(f"    CSV:  {csv_vol_1y:.4f}")
            print(f"    AK:   {ak_vol_1y:.4f}")
            print(f"    差异: {diff_vol_1y:.4f} {'✓' if match_vol_1y else '✗'}")
            
            print(f"  1年夏普比率:")
            print(f"    CSV:  {csv_sharpe_1y:.2f}")
            print(f"    AK:   {ak_sharpe_1y:.2f}")
            print(f"    差异: {diff_sharpe_1y:.2f} {'✓' if match_sharpe_1y else '✗'}")
            
            # 基金名称对比
            csv_name = fund_name
            ak_name = fund_data.get('product_name', '')
            name_match = csv_name == ak_name or ak_name.startswith('基金') or csv_name in ak_name or ak_name in csv_name
            
            print(f"  基金名称:")
            print(f"    CSV:  {csv_name}")
            print(f"    AK:   {ak_name}")
            print(f"    匹配: {'✓' if name_match else '✗'}")
            
            # 总结
            if all_match and name_match:
                status = "✓ 数据匹配"
                print(f"\n  ✓ 验证通过: 数据与 akshare 一致")
            elif all_match:
                status = "⚠ 数据匹配但名称不同"
                print(f"\n  ⚠ 数据匹配但名称不同")
            else:
                status = "✗ 数据不匹配"
                print(f"\n  ✗ 验证失败: 数据与 akshare 不一致")
            
            results.append({
                'fund_code': fund_code,
                'fund_name': fund_name,
                'status': status,
                'match': all_match,
                'details': {
                    'return_1y_diff': diff_return_1y,
                    'return_3y_diff': diff_return_3y,
                    'max_dd_3y_diff': diff_max_dd_3y,
                    'vol_1y_diff': diff_vol_1y,
                    'sharpe_1y_diff': diff_sharpe_1y,
                    'name_match': name_match
                }
            })
            
        except Exception as e:
            print(f"  ✗ 验证过程出错: {e}")
            results.append({
                'fund_code': fund_code,
                'fund_name': fund_name,
                'status': '验证出错',
                'match': False,
                'details': str(e)
            })
        
        print()
    
    # 生成总结报告
    print("=" * 80)
    print("验证总结报告")
    print("=" * 80)
    
    total = len(results)
    matched = sum(1 for r in results if r.get('match', False))
    failed = total - matched
    
    print(f"总基金数: {total}")
    print(f"数据匹配: {matched} ({matched/total*100:.1f}%)")
    print(f"数据不匹配: {failed} ({failed/total*100:.1f}%)")
    print()
    
    print("详细结果:")
    for r in results:
        status_icon = "✓" if r.get('match') else "✗"
        print(f"  {status_icon} {r['fund_code']}: {r['fund_name']} - {r['status']}")
    
    # 找出需要更新的基金
    need_update = [r for r in results if not r.get('match', False) and r['status'] != '无法获取']
    if need_update:
        print(f"\n需要更新的基金 ({len(need_update)} 个):")
        for r in need_update:
            print(f"  - {r['fund_code']}: {r['fund_name']}")
    
    return results


if __name__ == "__main__":
    results = verify_all_funds()
    sys.exit(0)

