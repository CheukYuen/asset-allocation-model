"""
基础示例：如何使用资产配置模型

这个示例展示了最基本的使用方法：
1. 创建资产组合
2. 获取历史数据
3. 分析资产表现
4. 计算简单的等权重配置
"""

import sys
from pathlib import Path

# 将src目录添加到Python路径中
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from portfolio import Portfolio


def main():
    print("=" * 60)
    print("资产配置模型 - 基础示例")
    print("=" * 60)
    
    # 步骤1: 定义你想要分析的资产
    # 这里使用美国科技股作为示例
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    print(f"\n分析的资产: {', '.join(tickers)}")
    
    # 步骤2: 创建资产组合对象
    portfolio = Portfolio(tickers)
    
    # 步骤3: 获取历史数据（默认最近一年）
    print("\n正在获取历史数据...")
    portfolio.fetch_data()
    
    # 步骤4: 计算收益率
    portfolio.calculate_returns()
    
    # 步骤5: 查看各资产的统计信息
    print("\n" + "=" * 60)
    print("各资产统计信息")
    print("=" * 60)
    stats = portfolio.get_statistics()
    print(stats.to_string())
    
    # 步骤6: 计算等权重配置
    print("\n" + "=" * 60)
    print("等权重配置策略")
    print("=" * 60)
    weights = portfolio.equal_weight_allocation()
    for ticker, weight in weights.items():
        print(f"{ticker:10s}: {weight:>6.2%}")
    
    # 步骤7: 计算组合表现
    print("\n" + "=" * 60)
    print("组合整体表现")
    print("=" * 60)
    annual_return, annual_volatility = portfolio.calculate_portfolio_performance(weights)
    sharpe_ratio = annual_return / annual_volatility
    
    print(f"年化收益率:   {annual_return:>7.2%}")
    print(f"年化波动率:   {annual_volatility:>7.2%}")
    print(f"夏普比率:     {sharpe_ratio:>7.2f}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

