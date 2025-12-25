#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证基金 000961 是否存在于 akshare 并分析 max_drawdown_3y 值
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

fund_code = "000961"
print(f"正在验证基金 {fund_code} 是否存在于 akshare...")
print("=" * 60)

# 尝试多种接口
fund_data = None
method_used = None

# 方法1: fund_em_fund_info
try:
    print("\n[方法1] 尝试 fund_em_fund_info...")
    fund_info = ak.fund_em_fund_info(fund=fund_code, indicator="单位净值走势")
    if not fund_info.empty:
        print(f"  ✓ 成功获取数据，共 {len(fund_info)} 条记录")
        print(f"  列名: {list(fund_info.columns)}")
        print(f"  数据范围: {fund_info.iloc[0, 0]} 到 {fund_info.iloc[-1, 0]}")
        fund_data = fund_info
        method_used = "fund_em_fund_info"
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 方法2: fund_open_fund_info
if fund_data is None:
    try:
        print("\n[方法2] 尝试 fund_open_fund_info...")
        fund_info = ak.fund_open_fund_info(fund=fund_code)
        if not fund_info.empty:
            print(f"  ✓ 成功获取数据")
            print(f"  数据类型: {type(fund_info)}")
            print(f"  数据预览:")
            print(fund_info.head(10).to_string())
            fund_data = fund_info
            method_used = "fund_open_fund_info"
    except Exception as e:
        print(f"  ✗ 失败: {e}")

# 方法3: fund_etf_hist_em
if fund_data is None:
    try:
        print("\n[方法3] 尝试 fund_etf_hist_em...")
        fund_info = ak.fund_etf_hist_em(symbol=fund_code)
        if not fund_info.empty:
            print(f"  ✓ 成功获取数据，共 {len(fund_info)} 条记录")
            print(f"  列名: {list(fund_info.columns)}")
            fund_data = fund_info
            method_used = "fund_etf_hist_em"
    except Exception as e:
        print(f"  ✗ 失败: {e}")

# 如果有数据，详细分析
if fund_data is not None:
    print(f"\n{'='*60}")
    print(f"使用 {method_used} 获取的数据进行分析")
    print(f"{'='*60}")
    
    # 确定日期列和净值列
    date_col = None
    nav_col = None
    
    if '净值日期' in fund_data.columns:
        date_col = '净值日期'
        nav_col = '单位净值'
    elif '日期' in fund_data.columns:
        date_col = '日期'
        # 对于ETF数据，应该使用收盘价作为净值
        if '收盘' in fund_data.columns:
            nav_col = '收盘'
        elif '净值' in fund_data.columns:
            nav_col = '净值'
        else:
            nav_col = fund_data.columns[1]
    else:
        date_col = fund_data.columns[0]
        nav_col = fund_data.columns[1]
    
    print(f"\n使用的列: 日期={date_col}, 净值={nav_col}")
    
    # 处理数据
    fund_data = fund_data.copy()
    fund_data[date_col] = pd.to_datetime(fund_data[date_col])
    fund_data = fund_data.sort_values(date_col).reset_index(drop=True)
    
    # 显示基本信息
    print(f"\n数据统计:")
    print(f"  总记录数: {len(fund_data)}")
    print(f"  日期范围: {fund_data[date_col].min()} 到 {fund_data[date_col].max()}")
    print(f"  净值范围: {fund_data[nav_col].min():.4f} 到 {fund_data[nav_col].max():.4f}")
    print(f"  最新净值: {fund_data.iloc[-1][nav_col]:.4f}")
    
    # 计算3年最大回撤
    end_date = fund_data[date_col].max()
    start_date = end_date - timedelta(days=3 * 365)
    period_data = fund_data[fund_data[date_col] >= start_date].copy()
    
    print(f"\n3年数据统计 (从 {start_date.date()} 到 {end_date.date()}):")
    print(f"  数据点数: {len(period_data)}")
    
    if len(period_data) > 0:
        valid_data = period_data[[date_col, nav_col]].dropna()
        if len(valid_data) > 0:
            nav_values = valid_data[nav_col].values
            
            print(f"  有效数据点数: {len(nav_values)}")
            print(f"  起始净值: {nav_values[0]:.4f}")
            print(f"  结束净值: {nav_values[-1]:.4f}")
            print(f"  最高净值: {nav_values.max():.4f}")
            print(f"  最低净值: {nav_values.min():.4f}")
            
            # 计算最大回撤
            cummax = np.maximum.accumulate(nav_values)
            drawdown = (nav_values - cummax) / cummax
            max_dd = np.min(drawdown)
            max_dd_idx = np.argmin(drawdown)
            
            print(f"\n最大回撤分析:")
            print(f"  最大回撤值: {abs(max_dd):.4f} ({abs(max_dd)*100:.2f}%)")
            print(f"  CSV中的值: 0.9194")
            print(f"  差异: {abs(abs(max_dd) - 0.9194):.4f}")
            
            # 显示回撤最大的时间段
            print(f"\n最大回撤发生位置:")
            print(f"  日期: {valid_data.iloc[max_dd_idx][date_col]}")
            print(f"  净值: {nav_values[max_dd_idx]:.4f}")
            print(f"  相对最高点: {cummax[max_dd_idx]:.4f}")
            print(f"  回撤幅度: {abs(drawdown[max_dd_idx]):.4f} ({abs(drawdown[max_dd_idx])*100:.2f}%)")
            
            # 检查是否有异常值
            print(f"\n异常值检查:")
            if abs(max_dd) > 0.5:
                print(f"  ⚠️  警告: 最大回撤超过50%，可能存在数据异常")
            if nav_values.min() < 0.1:
                print(f"  ⚠️  警告: 最低净值低于0.1，可能数据有误")
            
            # 显示净值走势的关键点
            print(f"\n净值走势关键点 (前10个和后10个):")
            print(f"  前10个数据点:")
            for i in range(min(10, len(valid_data))):
                print(f"    {valid_data.iloc[i][date_col].date()}: {nav_values[i]:.4f}")
            print(f"  后10个数据点:")
            for i in range(max(0, len(valid_data)-10), len(valid_data)):
                print(f"    {valid_data.iloc[i][date_col].date()}: {nav_values[i]:.4f}")
        else:
            print("  ✗ 没有有效数据")
    else:
        print("  ✗ 3年数据不足")
else:
    print("\n" + "="*60)
    print("❌ 所有方法均失败，该基金代码可能不存在于 akshare")
    print("="*60)
    print("\n可能的原因:")
    print("1. 基金代码 000961 可能不是 akshare 支持的格式")
    print("2. 该基金可能已清盘或更名")
    print("3. akshare 可能不支持该基金的数据源")
    print("\n建议:")
    print("- 检查基金代码是否正确")
    print("- 尝试使用其他数据源验证")
    print("- 查看基金公司官网确认基金状态")

