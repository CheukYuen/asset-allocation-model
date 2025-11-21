"""
资产组合管理模块
提供基础的资产组合分析功能
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class Portfolio:
    """
    资产组合类
    
    用于管理和分析一个资产组合
    """
    
    def __init__(self, tickers):
        """
        初始化资产组合
        
        参数:
            tickers (list): 股票代码列表，例如 ['AAPL', 'GOOGL', 'MSFT']
        """
        self.tickers = tickers
        self.prices = None
        self.returns = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取历史价格数据
        
        参数:
            start_date (str): 开始日期，格式 'YYYY-MM-DD'
            end_date (str): 结束日期，格式 'YYYY-MM-DD'
        """
        # 如果没有指定日期，默认获取最近一年的数据
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print(f"正在获取数据: {start_date} 到 {end_date}")
        
        # 下载数据
        self.prices = yf.download(self.tickers, start=start_date, end=end_date)['Adj Close']
        
        # 如果只有一个股票，转换为DataFrame
        if len(self.tickers) == 1:
            self.prices = pd.DataFrame(self.prices, columns=self.tickers)
            
        print(f"成功获取 {len(self.prices)} 天的数据")
        
    def calculate_returns(self):
        """
        计算日收益率
        
        返回:
            DataFrame: 每日收益率
        """
        if self.prices is None:
            raise ValueError("请先使用 fetch_data() 获取价格数据")
        
        # 计算百分比变化（日收益率）
        self.returns = self.prices.pct_change().dropna()
        return self.returns
    
    def get_statistics(self):
        """
        获取资产组合的统计信息
        
        返回:
            DataFrame: 包含年化收益率和年化波动率的统计信息
        """
        if self.returns is None:
            self.calculate_returns()
        
        # 计算年化收益率（假设252个交易日）
        annual_returns = self.returns.mean() * 252
        
        # 计算年化波动率
        annual_volatility = self.returns.std() * np.sqrt(252)
        
        # 创建统计信息DataFrame
        stats = pd.DataFrame({
            '年化收益率': annual_returns,
            '年化波动率': annual_volatility,
            '夏普比率': annual_returns / annual_volatility  # 简化版，假设无风险利率为0
        })
        
        return stats
    
    def equal_weight_allocation(self):
        """
        计算等权重配置
        
        返回:
            dict: 每个资产的权重
        """
        weight = 1.0 / len(self.tickers)
        weights = {ticker: weight for ticker in self.tickers}
        return weights
    
    def calculate_portfolio_performance(self, weights):
        """
        计算给定权重下的组合表现
        
        参数:
            weights (dict): 资产权重字典，例如 {'AAPL': 0.5, 'GOOGL': 0.5}
        
        返回:
            tuple: (年化收益率, 年化波动率)
        """
        if self.returns is None:
            self.calculate_returns()
        
        # 将权重字典转换为数组
        weight_array = np.array([weights[ticker] for ticker in self.tickers])
        
        # 计算组合收益率
        portfolio_returns = (self.returns * weight_array).sum(axis=1)
        
        # 计算年化指标
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        return annual_return, annual_volatility


def main():
    """
    示例：演示如何使用Portfolio类
    """
    # 创建一个包含三只股票的组合
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    portfolio = Portfolio(tickers)
    
    # 获取数据
    portfolio.fetch_data()
    
    # 计算收益率
    portfolio.calculate_returns()
    
    # 显示统计信息
    print("\n各资产统计信息:")
    print(portfolio.get_statistics())
    
    # 等权重配置
    weights = portfolio.equal_weight_allocation()
    print("\n等权重配置:")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.2%}")
    
    # 计算组合表现
    annual_return, annual_volatility = portfolio.calculate_portfolio_performance(weights)
    print(f"\n组合年化收益率: {annual_return:.2%}")
    print(f"组合年化波动率: {annual_volatility:.2%}")
    print(f"组合夏普比率: {annual_return/annual_volatility:.2f}")


if __name__ == "__main__":
    main()

