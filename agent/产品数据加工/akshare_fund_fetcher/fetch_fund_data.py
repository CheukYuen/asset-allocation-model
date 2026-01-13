#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 akshare 获取基金数据并生成产品信息 CSV
兼容 Python 3.9 + numpy 1.26.4 + pandas 2.2.3
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# 无风险利率（用于计算夏普比率，假设为 3%）
RISK_FREE_RATE = 0.03

# 交易日天数（用于年化计算）
TRADING_DAYS_PER_YEAR = 252


def calculate_returns(nav_data: pd.DataFrame, periods: List[int]) -> Dict[str, float]:
    """
    计算不同期限的年化收益率
    
    参数:
        nav_data: 净值数据，包含日期和净值列
        periods: 期限列表（年），如 [1, 3, 5]
    
    返回:
        字典，key 为 'return_1y', 'return_3y' 等
    """
    if nav_data.empty or len(nav_data) < 2:
        return {}
    
    # 确定日期列和净值列
    date_col, nav_col = _identify_columns(nav_data)
    if date_col is None or nav_col is None:
        return {}
    
    # 确保日期列是 datetime 类型并排序
    nav_data = nav_data.copy()
    nav_data[date_col] = pd.to_datetime(nav_data[date_col])
    nav_data = nav_data.sort_values(date_col).reset_index(drop=True)
    
    results = {}
    end_date = nav_data[date_col].max()
    
    for period in periods:
        start_date = end_date - timedelta(days=period * 365)
        period_data = nav_data[nav_data[date_col] >= start_date].copy()
        
        if len(period_data) < 2:
            results[f'return_{period}y'] = 0.0
            continue
        
        # 获取第一个和最后一个有效净值
        valid_data = period_data[[date_col, nav_col]].dropna()
        if len(valid_data) < 2:
            results[f'return_{period}y'] = 0.0
            continue
        
        start_nav = valid_data.iloc[0][nav_col]
        end_nav = valid_data.iloc[-1][nav_col]
        
        if pd.isna(start_nav) or pd.isna(end_nav) or start_nav <= 0:
            results[f'return_{period}y'] = 0.0
            continue
        
        # 计算总收益率
        total_return = (end_nav / start_nav) - 1
        
        # 年化收益率
        start_dt = valid_data.iloc[0][date_col]
        end_dt = valid_data.iloc[-1][date_col]
        actual_days = (end_dt - start_dt).days
        
        if actual_days > 0:
            annualized_return = (1 + total_return) ** (365.0 / actual_days) - 1
        else:
            annualized_return = 0.0
        
        results[f'return_{period}y'] = round(annualized_return, 4)
    
    return results


def calculate_volatility(nav_data: pd.DataFrame, periods: List[int]) -> Dict[str, float]:
    """
    计算不同期限的年化波动率
    
    参数:
        nav_data: 净值数据
        periods: 期限列表（年）
    
    返回:
        字典，key 为 'volatility_1y', 'volatility_3y' 等
    """
    if nav_data.empty or len(nav_data) < 2:
        return {}
    
    # 确定列名
    date_col, nav_col = _identify_columns(nav_data)
    if date_col is None or nav_col is None:
        return {}
    
    # 确保日期列是 datetime 类型并排序
    nav_data = nav_data.copy()
    nav_data[date_col] = pd.to_datetime(nav_data[date_col])
    nav_data = nav_data.sort_values(date_col).reset_index(drop=True)
    
    # 计算日收益率
    nav_data['daily_return'] = nav_data[nav_col].pct_change().dropna()
    
    results = {}
    end_date = nav_data[date_col].max()
    
    for period in periods:
        start_date = end_date - timedelta(days=period * 365)
        period_data = nav_data[nav_data[date_col] >= start_date].copy()
        
        if len(period_data) < 2:
            results[f'volatility_{period}y'] = 0.0
            continue
        
        daily_returns = period_data['daily_return'].dropna()
        if len(daily_returns) == 0:
            results[f'volatility_{period}y'] = 0.0
            continue
        
        # 计算日波动率，然后年化（假设 252 个交易日）
        daily_vol = daily_returns.std()
        annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        results[f'volatility_{period}y'] = round(annualized_vol, 4)
    
    return results


def calculate_max_drawdown(nav_data: pd.DataFrame, periods: List[int]) -> Dict[str, float]:
    """
    计算不同期限的最大回撤
    
    参数:
        nav_data: 净值数据
        periods: 期限列表（年）
    
    返回:
        字典，key 为 'max_drawdown_1y', 'max_drawdown_2y' 等
    """
    if nav_data.empty or len(nav_data) < 2:
        return {}
    
    # 确定列名
    date_col, nav_col = _identify_columns(nav_data)
    if date_col is None or nav_col is None:
        return {}
    
    # 确保日期列是 datetime 类型并排序
    nav_data = nav_data.copy()
    nav_data[date_col] = pd.to_datetime(nav_data[date_col])
    nav_data = nav_data.sort_values(date_col).reset_index(drop=True)
    
    results = {}
    end_date = nav_data[date_col].max()
    
    for period in periods:
        start_date = end_date - timedelta(days=period * 365)
        period_data = nav_data[nav_data[date_col] >= start_date].copy()
        
        if len(period_data) < 2:
            results[f'max_drawdown_{period}y'] = 0.0
            continue
        
        # 获取有效净值数据
        valid_data = period_data[[date_col, nav_col]].dropna()
        if len(valid_data) < 2:
            results[f'max_drawdown_{period}y'] = 0.0
            continue
        
        nav_values = valid_data[nav_col].values
        
        # 计算累计最大值
        cummax = np.maximum.accumulate(nav_values)
        # 计算回撤（负数表示下跌）
        drawdown = (nav_values - cummax) / cummax
        
        max_dd = np.min(drawdown)
        results[f'max_drawdown_{period}y'] = round(abs(max_dd), 4) if not np.isnan(max_dd) else 0.0
    
    return results


def calculate_sharpe_ratio(nav_data: pd.DataFrame, periods: List[int], risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, float]:
    """
    计算不同期限的夏普比率
    
    参数:
        nav_data: 净值数据
        periods: 期限列表（年）
        risk_free_rate: 无风险利率（年化）
    
    返回:
        字典，key 为 'sharpe_ratio_1y', 'sharpe_ratio_3y' 等
    """
    if nav_data.empty or len(nav_data) < 2:
        return {}
    
    # 先计算收益率和波动率
    returns = calculate_returns(nav_data, periods)
    volatilities = calculate_volatility(nav_data, periods)
    
    results = {}
    for period in periods:
        ret_key = f'return_{period}y'
        vol_key = f'volatility_{period}y'
        
        if ret_key in returns and vol_key in volatilities:
            excess_return = returns[ret_key] - risk_free_rate
            vol = volatilities[vol_key]
            
            if vol > 0:
                sharpe = excess_return / vol
            else:
                sharpe = 0.0
            
            results[f'sharpe_ratio_{period}y'] = round(sharpe, 2)
        else:
            results[f'sharpe_ratio_{period}y'] = 0.0
    
    return results


def determine_risk_level(volatility_3y: float, max_drawdown_3y: float) -> str:
    """
    根据波动率和最大回撤确定风险等级
    
    参数:
        volatility_3y: 3 年年化波动率
        max_drawdown_3y: 3 年最大回撤
    
    返回:
        风险等级字符串 (R1-R5)
    """
    if volatility_3y < 0.05 and max_drawdown_3y < 0.05:
        return 'R1'
    elif volatility_3y < 0.10 and max_drawdown_3y < 0.10:
        return 'R2'
    elif volatility_3y < 0.15 and max_drawdown_3y < 0.20:
        return 'R3'
    elif volatility_3y < 0.25 and max_drawdown_3y < 0.30:
        return 'R4'
    else:
        return 'R5'


def _identify_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    识别日期列和净值列
    
    返回:
        (date_col, nav_col) 元组
    """
    # 常见的日期列名
    date_candidates = ['净值日期', '日期', 'date', 'Date', '交易日期']
    # 常见的净值列名
    nav_candidates = ['单位净值', '净值', 'nav', 'NAV', '累计净值', '收盘', 'close', 'Close']
    
    date_col = None
    nav_col = None
    
    # 查找日期列
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break
    
    # 如果没找到，尝试第一列
    if date_col is None and len(df.columns) > 0:
        first_col = df.columns[0]
        # 检查第一列是否看起来像日期
        try:
            pd.to_datetime(df[first_col].iloc[0])
            date_col = first_col
        except:
            pass
    
    # 查找净值列
    for col in nav_candidates:
        if col in df.columns and col != date_col:
            nav_col = col
            break
    
    # 如果没找到，尝试第二列
    if nav_col is None and len(df.columns) > 1:
        second_col = df.columns[1]
        if second_col != date_col:
            nav_col = second_col
    
    return date_col, nav_col


def get_fund_info(fund_code: str) -> Optional[Dict]:
    """
    获取单个基金的完整信息
    
    参数:
        fund_code: 基金代码（6 位数字字符串）
    
    返回:
        包含所有字段的字典，如果获取失败返回 None
    """
    try:
        fund_info = None
        
        # 尝试多种接口获取基金数据
        # 方法1: fund_em_fund_info (东方财富接口)
        try:
            fund_info = ak.fund_em_fund_info(fund=fund_code, indicator="单位净值走势")
            if not fund_info.empty:
                print(f"  ✓ 使用 fund_em_fund_info 获取数据")
        except Exception as e1:
            # 方法2: fund_open_fund_info (开放式基金)
            try:
                fund_info = ak.fund_open_fund_info(fund=fund_code)
                if not fund_info.empty:
                    print(f"  ✓ 使用 fund_open_fund_info 获取数据")
            except Exception as e2:
                # 方法3: fund_etf_hist_em (ETF 基金)
                try:
                    fund_info = ak.fund_etf_hist_em(symbol=fund_code)
                    if not fund_info.empty:
                        # 标准化列名
                        if '日期' in fund_info.columns:
                            fund_info = fund_info.rename(columns={'日期': '净值日期'})
                        if '收盘' in fund_info.columns:
                            fund_info = fund_info.rename(columns={'收盘': '单位净值'})
                        print(f"  ✓ 使用 fund_etf_hist_em 获取数据")
                except Exception as e3:
                    print(f"  ✗ 所有接口均失败: {e1}, {e2}, {e3}")
                    return None
        
        if fund_info is None or fund_info.empty:
            return None
        
        # 获取基金名称和基本信息
        product_name = f"基金{fund_code}"
        fund_type = '未知'
        asset_class = '权益类'
        sub_category = '股票型'
        
        try:
            fund_name_df = ak.fund_em_fund_name()
            if not fund_name_df.empty:
                fund_name_info = fund_name_df[fund_name_df['基金代码'] == fund_code]
                if not fund_name_info.empty:
                    product_name = fund_name_info.iloc[0]['基金简称']
                    fund_type = fund_name_info.iloc[0].get('基金类型', '未知')
                    
                    # 根据基金类型确定资产类别和子类别
                    if '货币' in str(fund_type) or '货基' in str(fund_type):
                        asset_class = '现金类'
                        sub_category = '现金类-其他'
                    elif '债券' in str(fund_type) or '债基' in str(fund_type):
                        asset_class = '固收类'
                        sub_category = '债券型'
                    elif '混合' in str(fund_type):
                        asset_class = '混合类'
                        sub_category = '混合型'
                    elif 'ETF' in str(fund_type) or '指数' in str(fund_type):
                        asset_class = '权益类'
                        sub_category = '股票型'
        except Exception as e:
            print(f"  ⚠ 获取基金名称失败，使用默认值: {e}")
        
        # 计算所有指标
        returns = calculate_returns(fund_info, [1, 3, 5])
        volatilities = calculate_volatility(fund_info, [1, 3, 5])
        max_drawdowns = calculate_max_drawdown(fund_info, [1, 2, 3])
        sharpe_ratios = calculate_sharpe_ratio(fund_info, [1, 3, 5])
        
        # 确定风险等级
        risk_level = determine_risk_level(
            volatilities.get('volatility_3y', 0.0),
            max_drawdowns.get('max_drawdown_3y', 0.0)
        )
        
        # 组装结果
        result = {
            'product_name': product_name,
            'product_code': fund_code,
            'currency': 'CNY',
            'asset_class': asset_class,
            'sub_category': sub_category,
            **returns,
            **volatilities,
            **max_drawdowns,
            **sharpe_ratios,
            'risk_level': risk_level
        }
        
        return result
        
    except Exception as e:
        print(f"获取基金 {fund_code} 数据失败: {e}")
        return None


def generate_fund_csv(fund_codes: List[str], output_file: str = "真实产品数据v1.csv"):
    """
    批量获取基金数据并生成 CSV
    
    参数:
        fund_codes: 基金代码列表
        output_file: 输出文件名
    """
    # CSV 列顺序（必须与真实产品数据v1.csv 格式一致）
    columns = [
        'product_name', 'product_code', 'currency', 'asset_class', 'sub_category', 'risk_level',
        'return_1y', 'return_3y', 'return_5y',
        'volatility_1y', 'volatility_3y', 'volatility_5y',
        'max_drawdown_1y', 'max_drawdown_2y', 'max_drawdown_3y',
        'sharpe_ratio_1y', 'sharpe_ratio_3y', 'sharpe_ratio_5y'
    ]
    
    results = []
    
    print(f"开始获取 {len(fund_codes)} 个基金的数据...")
    print("=" * 60)
    
    for i, fund_code in enumerate(fund_codes, 1):
        print(f"[{i}/{len(fund_codes)}] 正在获取基金 {fund_code}...")
        fund_info = get_fund_info(str(fund_code))
        
        if fund_info:
            results.append(fund_info)
            print(f"  ✓ 成功: {fund_info['product_name']}")
        else:
            print(f"  ✗ 获取失败，跳过")
        
        # 避免请求过快
        if i < len(fund_codes):
            time.sleep(0.5)
    
    if not results:
        print("\n❌ 未获取到任何基金数据！")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 确保所有列都存在
    for col in columns:
        if col not in df.columns:
            if 'return' in col or 'volatility' in col or 'max_drawdown' in col or 'sharpe' in col:
                df[col] = 0.0
            else:
                df[col] = ''
    
    # 按指定顺序排列列
    df = df[columns]
    
    # 保存为 CSV（使用 utf-8-sig 编码以支持 Excel 打开）
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 60)
    print(f"✓ 成功生成 CSV 文件: {output_file}")
    print(f"  共 {len(df)} 条记录")
    print(f"\n前 3 条数据预览:")
    print(df.head(3).to_string())
    print("\n" + "=" * 60)


def main():
    """主函数"""
    # 示例：10 个基金代码（你可以替换为任意基金代码）
    # 这里使用一些常见的基金代码作为示例
    fund_codes_example = [
        "000001",  # 华夏成长
        "000198",  # 余额宝
        "510300",  # 沪深300ETF
        "510500",  # 中证500ETF
        "159919",  # 沪深300ETF
        "159915",  # 创业板ETF
        "000961",  # 天弘沪深300
        "001632",  # 天弘中证500
        "110022",  # 易方达消费行业
        "161725",  # 招商中证白酒
    ]
    
    # 生成 CSV（输出到当前目录）
    output_path = "真实产品数据v1.csv"
    generate_fund_csv(fund_codes_example, output_path)


if __name__ == "__main__":
    main()

