"""
独立脚本: 从投资组合配置反向推导协方差矩阵
Standalone Script: Reverse-engineer covariance matrix from portfolio allocations

功能 (Function):
- 读取 105.csv 中的资产配置比例
- 使用反向优化方法估计协方差矩阵
- 输出 4x4 协方差矩阵 (现金、债券、权益、另类资产)

数据维度说明 (Data Dimensions):
105套组合 = 7个人生阶段 × 3种需求 × 5个风险等级

三个维度的作用:
1. lifecycle (人生阶段) - 间接作用
   - 影响权重配置的基调
   - 例如: 退休→高现金, 刚毕业→高权益
   - 完全编码在权重矩阵中

2. demand (理财需求) - 间接作用
   - 影响风险偏好和配置风格
   - 例如: 保值→债券为主, 增值→权益提升
   - 完全编码在权重矩阵中

3. risk_level (风险等级) - 直接+间接作用
   - 直接: 设定目标波动率 (C1=3%, C5=15%)
   - 间接: 影响权重配置的激进程度
   - 在算法中同时使用

数学原理:
- 主要信息源: 105个组合的权重 (420个数值, 99%)
- 辅助信息: risk_level 设定目标方差 (1%)
- 核心: 求解超定线性系统 (105个方程, 10个未知数)

详细数学推导请参阅: MATHEMATICAL_PRINCIPLES.md

Python 3.9+ 兼容, 无需 SciPy
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def load_portfolio_weights(csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载投资组合权重数据
    Load portfolio weight data

    Parameters:
    -----------
    csv_file : str
        105.csv 文件路径

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (weights, risk_levels)
        - weights: (105, 4) 权重矩阵, 列顺序为 [BOND, CASH, COMMODITY, EQUITY]
        - risk_levels: (105,) 风险等级 (1-5)
    """
    df = pd.read_csv(csv_file)

    # 提取权重列 (百分比转为小数)
    weight_cols = ['cash_pct', 'bond_pct', 'equity_pct', 'commodity_pct']
    weights = df[weight_cols].values / 100.0

    # 重新排序为 [BOND, CASH, COMMODITY, EQUITY] (字母顺序)
    # CSV 顺序: cash, bond, equity, commodity
    # 目标顺序: bond, cash, commodity, equity
    reorder_idx = [1, 0, 3, 2]
    weights = weights[:, reorder_idx]

    # 提取风险等级 (C1 -> 1, C2 -> 2, ...)
    risk_levels = df['risk_level'].apply(lambda x: int(x[1])).values

    return weights, risk_levels


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    将矩阵投影到最近的半正定矩阵
    Project matrix to nearest positive semi-definite matrix

    Parameters:
    -----------
    matrix : np.ndarray
        输入矩阵
    epsilon : float
        最小特征值阈值

    Returns:
    --------
    np.ndarray
        半正定矩阵
    """
    # 确保对称
    matrix_sym = (matrix + matrix.T) / 2

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(matrix_sym)

    # 裁剪负特征值
    eigvals_pos = np.maximum(eigvals, epsilon)

    # 重构矩阵
    matrix_psd = eigvecs @ np.diag(eigvals_pos) @ eigvecs.T

    # 再次确保对称
    matrix_psd = (matrix_psd + matrix_psd.T) / 2

    return matrix_psd


def reverse_optimize_covariance(
    weights: np.ndarray,
    risk_levels: np.ndarray,
    method: str = 'least_squares'
) -> np.ndarray:
    """
    通过反向优化估计协方差矩阵
    Estimate covariance matrix via reverse optimization

    Parameters:
    -----------
    weights : np.ndarray
        投资组合权重矩阵 (105, 4)
    risk_levels : np.ndarray
        风险等级 (105,)
    method : str
        优化方法 ('least_squares' 或 'moment_matching')

    Returns:
    --------
    np.ndarray
        估计的协方差矩阵 (4, 4)
    """
    n_portfolios, n_assets = weights.shape

    if method == 'least_squares':
        return _reverse_optimize_ls(weights, risk_levels, n_assets)
    elif method == 'moment_matching':
        return _reverse_optimize_mm(weights, risk_levels, n_assets)
    else:
        raise ValueError(f"未知方法: {method}")


def _reverse_optimize_ls(
    weights: np.ndarray,
    risk_levels: np.ndarray,
    n_assets: int
) -> np.ndarray:
    """
    最小二乘法估计协方差矩阵
    Least squares method for covariance estimation

    策略:
    1. 定义目标方差: 基于风险等级缩放 (C1最低, C5最高)
    2. 求解 Σ 使得 w^T Σ w ≈ 目标方差
    3. 使用最小二乘法求解
    """
    n_portfolios = len(weights)

    # 定义目标波动率 (基于风险等级)
    # C1: 3%, C2: 6%, C3: 9%, C4: 12%, C5: 15%
    target_vols = 0.03 + (risk_levels - 1) * 0.03
    target_vars = target_vols ** 2

    # 构建设计矩阵 A
    # 对称矩阵 4x4 有 10 个独特元素
    # 索引: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    A = np.zeros((n_portfolios, 10))

    for i in range(n_portfolios):
        w = weights[i, :]
        idx = 0
        for j in range(n_assets):
            for k in range(j, n_assets):
                if j == k:
                    # 对角元素: w_j^2
                    A[i, idx] = w[j] ** 2
                else:
                    # 非对角元素: 2 * w_j * w_k (对称矩阵)
                    A[i, idx] = 2 * w[j] * w[k]
                idx += 1

    # 求解最小二乘: A @ sigma_vec = target_vars
    sigma_vec, residuals, rank, s = np.linalg.lstsq(A, target_vars, rcond=None)

    # 重构协方差矩阵
    cov_matrix = np.zeros((n_assets, n_assets))
    idx = 0
    for j in range(n_assets):
        for k in range(j, n_assets):
            cov_matrix[j, k] = sigma_vec[idx]
            cov_matrix[k, j] = sigma_vec[idx]
            idx += 1

    # 确保半正定
    cov_matrix = nearest_psd(cov_matrix)

    return cov_matrix


def _reverse_optimize_mm(
    weights: np.ndarray,
    risk_levels: np.ndarray,
    n_assets: int
) -> np.ndarray:
    """
    矩匹配法估计协方差矩阵
    Moment matching method for covariance estimation

    策略:
    1. 将每个组合权重视为"样本"
    2. 计算加权样本协方差
    3. 按风险等级加权 (高风险组合权重更大)
    """
    # 风险等级加权
    risk_weights = risk_levels / risk_levels.sum()

    # 加权均值
    w_mean = (weights.T @ risk_weights).reshape(-1, 1)

    # 去中心化
    centered = weights - w_mean.T

    # 加权协方差
    cov_matrix = (centered.T @ np.diag(risk_weights) @ centered)

    # 缩放到合理数量级 (经验系数)
    cov_matrix = cov_matrix * 0.01

    # 确保半正定
    cov_matrix = nearest_psd(cov_matrix)

    return cov_matrix


def compute_portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    计算投资组合波动率
    Compute portfolio volatility

    Formula: σ_p = sqrt(w^T * Σ * w)

    Parameters:
    -----------
    weights : np.ndarray
        权重矩阵 (n_portfolios, n_assets)
    cov_matrix : np.ndarray
        协方差矩阵 (n_assets, n_assets)

    Returns:
    --------
    np.ndarray
        波动率向量 (n_portfolios,)
    """
    n_portfolios = weights.shape[0]
    volatilities = np.zeros(n_portfolios)

    for i in range(n_portfolios):
        w = weights[i, :]
        var = w @ cov_matrix @ w
        volatilities[i] = np.sqrt(var)

    return volatilities


def cov_to_corr(cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    协方差矩阵转相关性矩阵
    Convert covariance to correlation matrix

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵 (n, n)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (相关性矩阵, 波动率向量)
    """
    # 提取波动率 (标准差)
    volatility = np.sqrt(np.diag(cov_matrix))

    # 计算相关性: ρ_ij = Σ_ij / (σ_i * σ_j)
    inv_vol = 1.0 / volatility
    corr_matrix = (cov_matrix * inv_vol[:, np.newaxis]) * inv_vol[np.newaxis, :]

    # 确保对角线为1
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix, volatility


def print_matrix(matrix: np.ndarray, asset_names: list, title: str) -> None:
    """
    格式化打印矩阵
    Print matrix with formatting
    """
    print(f"\n{title}")
    print("=" * 80)

    # 创建DataFrame以便格式化显示
    df = pd.DataFrame(matrix, index=asset_names, columns=asset_names)
    print(df.to_string())
    print()


def main():
    """主函数 Main function"""

    print("=" * 80)
    print("协方差矩阵反向优化 | Reverse Covariance Matrix Optimization")
    print("=" * 80)
    print()

    # 配置
    csv_file = '105.csv'
    asset_names = ['BOND', 'CASH', 'COMMODITY', 'EQUITY']

    # 步骤1: 加载数据
    print(f"步骤1: 加载投资组合数据 ({csv_file})...")
    weights, risk_levels = load_portfolio_weights(csv_file)
    print(f"  ✓ 加载 {len(weights)} 套投资组合")
    print(f"  ✓ 资产类别: {', '.join(asset_names)}")
    print()

    # 显示数据摘要
    print("数据摘要:")
    for level in range(1, 6):
        mask = risk_levels == level
        n = mask.sum()
        avg_weights = weights[mask].mean(axis=0)
        print(f"  风险等级 C{level} ({n}套):")
        for i, name in enumerate(asset_names):
            print(f"    {name}: {avg_weights[i]:.2%}")
    print()

    # 步骤2: 反向优化协方差矩阵
    print("步骤2: 反向优化协方差矩阵...")
    print("  方法: 最小二乘法 (Least Squares)")
    cov_matrix = reverse_optimize_covariance(weights, risk_levels, method='least_squares')
    print("  ✓ 完成")
    print()

    # 步骤3: 验证半正定性
    print("步骤3: 验证协方差矩阵性质...")
    eigvals = np.linalg.eigvalsh(cov_matrix)
    is_psd = np.all(eigvals >= -1e-8)
    print(f"  半正定 (PSD): {is_psd}")
    print(f"  最小特征值: {eigvals.min():.6e}")
    print(f"  最大特征值: {eigvals.max():.6e}")
    print()

    # 步骤4: 提取相关性矩阵和波动率
    print("步骤4: 提取相关性矩阵和波动率向量...")
    corr_matrix, volatility = cov_to_corr(cov_matrix)
    print("  ✓ 完成")
    print()

    # 步骤5: 计算投资组合波动率
    print("步骤5: 计算投资组合波动率...")
    portfolio_vols = compute_portfolio_volatility(weights, cov_matrix)
    print(f"  波动率范围: [{portfolio_vols.min():.4f}, {portfolio_vols.max():.4f}]")
    print()
    print("  各风险等级平均波动率:")
    for level in range(1, 6):
        mask = risk_levels == level
        avg_vol = portfolio_vols[mask].mean()
        std_vol = portfolio_vols[mask].std()
        print(f"    C{level}: {avg_vol:.4f} ± {std_vol:.4f}")
    print()

    # 输出结果
    print("=" * 80)
    print("最终结果 | FINAL RESULTS")
    print("=" * 80)

    # 协方差矩阵
    print_matrix(cov_matrix, asset_names, "协方差矩阵 Σ (Covariance Matrix)")

    # 相关性矩阵
    print_matrix(corr_matrix, asset_names, "相关性矩阵 ρ (Correlation Matrix)")

    # 波动率向量
    print("波动率向量 σ (Volatility Vector)")
    print("=" * 80)
    for i, name in enumerate(asset_names):
        print(f"  {name:12s}: {volatility[i]:.6f}")
    print()

    # 保存结果
    print("步骤6: 保存结果...")

    # 保存协方差矩阵 (保留6位小数)
    df_cov = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
    df_cov = df_cov.round(6)
    df_cov.to_csv('reverse_covariance_matrix.csv')
    print("  ✓ 协方差矩阵保存至: reverse_covariance_matrix.csv")

    # 保存相关性矩阵 (保留6位小数)
    df_corr = pd.DataFrame(corr_matrix, index=asset_names, columns=asset_names)
    df_corr = df_corr.round(6)
    df_corr.to_csv('reverse_correlation_matrix.csv')
    print("  ✓ 相关性矩阵保存至: reverse_correlation_matrix.csv")

    # 保存波动率 (保留6位小数)
    df_vol = pd.DataFrame({
        'asset': asset_names,
        'volatility': volatility.round(6)
    })
    df_vol.to_csv('reverse_volatility.csv', index=False)
    print("  ✓ 波动率向量保存至: reverse_volatility.csv")

    # 保存组合波动率 (保留6位小数)
    df_portfolio_vol = pd.DataFrame({
        'portfolio_id': range(1, len(weights) + 1),
        'risk_level': [f'C{level}' for level in risk_levels],
        'volatility': portfolio_vols.round(6)
    })
    df_portfolio_vol.to_csv('reverse_portfolio_volatility.csv', index=False)
    print("  ✓ 组合波动率保存至: reverse_portfolio_volatility.csv")

    print()
    print("=" * 80)
    print("完成! | COMPLETED!")
    print("=" * 80)

    return cov_matrix, corr_matrix, volatility


if __name__ == '__main__':
    # 运行主函数
    cov_matrix, corr_matrix, volatility = main()

    # 可选: 在交互式环境中使用返回值
    # covariance_matrix = cov_matrix
    # correlation_matrix = corr_matrix
    # volatility_vector = volatility
