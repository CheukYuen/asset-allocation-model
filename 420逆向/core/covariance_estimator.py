"""
Reverse optimization to estimate covariance matrix from portfolio allocations.
从投资组合配置反向推导协方差矩阵的优化器。

Python 3.9 compatible - no scipy dependencies.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from utils import (
    nearest_psd_matrix,
    ensure_symmetry,
    cov_to_corr,
    is_positive_semidefinite
)


class CovarianceReverseOptimizer:
    """
    Reverse-engineer covariance matrix from portfolio allocations.
    从投资组合配置反向推导协方差矩阵。

    Assumes portfolios are approximately mean-variance optimal with
    varying risk aversion coefficients λ across risk levels.
    假设投资组合在不同风险等级下具有不同的风险厌恶系数λ,近似满足均值-方差最优化。
    """

    def __init__(self, portfolios_file: str):
        """
        Initialize optimizer with portfolio data.

        Parameters:
        -----------
        portfolios_file : str
            Path to 105.csv file containing portfolio allocations
        """
        self.portfolios_df = pd.read_csv(portfolios_file)
        self.n_portfolios = len(self.portfolios_df)
        self.asset_names = ['CASH', 'BOND', 'EQUITY', 'COMMODITY']
        self.n_assets = len(self.asset_names)

        # Extract portfolio weights (convert percentages to fractions)
        weight_cols = ['cash_pct', 'bond_pct', 'equity_pct', 'commodity_pct']
        self.weights = self.portfolios_df[weight_cols].values / 100.0

        # Reorder to match asset_names order (CASH, BOND, EQUITY, COMMODITY)
        # CSV order is: cash, bond, equity, commodity (already correct)
        # But we need alphabetical order for consistency: BOND, CASH, COMMODITY, EQUITY
        reorder_idx = [1, 0, 3, 2]  # cash, bond, equity, commodity → bond, cash, commodity, equity
        self.weights = self.weights[:, reorder_idx]
        self.asset_names_sorted = ['BOND', 'CASH', 'COMMODITY', 'EQUITY']

        # Extract risk levels (C1-C5)
        self.risk_levels = self.portfolios_df['risk_level'].apply(
            lambda x: int(x[1])  # Extract number from 'C1', 'C2', etc.
        ).values

        # Extract metadata
        self.lifecycle_stages = self.portfolios_df['lifecycle'].values
        self.demand_types = self.portfolios_df['demand'].values

    def compute_portfolio_volatility(self, covariance: np.ndarray) -> np.ndarray:
        """
        Compute annualized volatility for each portfolio given covariance matrix.
        给定协方差矩阵计算每个投资组合的年化波动率。

        Formula: σ_p = √(w^T Σ w)

        Parameters:
        -----------
        covariance : np.ndarray
            Covariance matrix (4x4)

        Returns:
        --------
        np.ndarray
            Portfolio volatilities (105,)
        """
        volatilities = np.zeros(self.n_portfolios)

        for i in range(self.n_portfolios):
            w = self.weights[i, :]
            var = w @ covariance @ w
            volatilities[i] = np.sqrt(var)

        return volatilities

    def estimate_risk_aversion(self, covariance: np.ndarray) -> Dict[int, float]:
        """
        Estimate risk aversion coefficient λ for each risk level.
        估计每个风险等级的风险厌恶系数λ。

        Uses existing covariance matrix to compute portfolio variances,
        then estimates λ assuming portfolios are on the efficient frontier.

        Higher risk level → lower λ (more risk tolerance)

        Parameters:
        -----------
        covariance : np.ndarray
            Existing covariance matrix (4x4)

        Returns:
        --------
        Dict[int, float]
            Risk aversion by level: {1: λ_C1, 2: λ_C2, ..., 5: λ_C5}
        """
        risk_aversion_by_level = {}

        for level in range(1, 6):  # C1 to C5
            # Extract portfolios at this risk level
            mask = self.risk_levels == level
            weights_level = self.weights[mask]

            # Compute average portfolio for this level
            w_avg = weights_level.mean(axis=0)

            # Compute portfolio variance
            var = w_avg @ covariance @ w_avg

            # Estimate λ using heuristic:
            # For mean-variance optimal: w* = (1/λ) Σ^(-1) μ
            # We assume μ scales with risk level (higher risk → higher return)
            # λ inversely scales with risk level
            # Use variance as proxy: λ ≈ 1 / var

            # Base lambda on portfolio variance (higher variance → lower λ)
            # Normalize so λ_C1 ≈ 5.0 and λ_C5 ≈ 1.0
            base_lambda = 1.0 / (var + 1e-6)

            # Scale λ to reasonable range
            risk_aversion_by_level[level] = base_lambda

        # Normalize so λ decreases monotonically with risk level
        # Force monotonic decrease: λ_C1 > λ_C2 > ... > λ_C5
        lambdas = [risk_aversion_by_level[i] for i in range(1, 6)]

        # If not monotonic, apply isotonic regression (simple averaging approach)
        for i in range(len(lambdas) - 1):
            if lambdas[i] < lambdas[i + 1]:
                # Average the two to enforce ordering
                avg = (lambdas[i] + lambdas[i + 1]) / 2
                lambdas[i] = avg * 1.1  # Slightly higher
                lambdas[i + 1] = avg * 0.9  # Slightly lower

        # Rescale to interpretable range (1-10)
        min_lambda = min(lambdas)
        max_lambda = max(lambdas)
        lambdas_scaled = [1 + 9 * (l - min_lambda) / (max_lambda - min_lambda + 1e-6)
                         for l in lambdas]

        # Reverse so C1 has highest lambda
        lambdas_scaled.reverse()
        risk_aversion_by_level = {i + 1: lambdas_scaled[i] for i in range(5)}

        return risk_aversion_by_level

    def reverse_optimize_covariance(
        self,
        existing_cov: np.ndarray,
        method: str = 'least_squares'
    ) -> np.ndarray:
        """
        Estimate covariance matrix by reverse optimization.
        通过反向优化估计协方差矩阵。

        Strategy:
        1. Compute target variances from portfolio weights and risk levels
        2. Solve for Σ that minimizes difference between target and computed variances
        3. Enforce PSD constraint

        Parameters:
        -----------
        existing_cov : np.ndarray
            Existing covariance matrix (for initialization)
        method : str
            Optimization method ('least_squares' or 'moment_matching')

        Returns:
        --------
        np.ndarray
            Estimated covariance matrix (4x4)
        """
        if method == 'least_squares':
            return self._reverse_optimize_least_squares(existing_cov)
        elif method == 'moment_matching':
            return self._reverse_optimize_moment_matching()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _reverse_optimize_least_squares(self, existing_cov: np.ndarray) -> np.ndarray:
        """
        Estimate Σ using least squares approach.
        使用最小二乘法估计协方差矩阵。

        Minimize: ||W^T Σ W - target_variances||²

        where target_variances are derived from risk level scaling.
        """
        # Compute current variances using existing covariance
        current_vols = self.compute_portfolio_volatility(existing_cov)

        # Define target variances based on risk level
        # C1: lowest variance, C5: highest variance
        # Scale variances proportionally
        target_vols = np.zeros(self.n_portfolios)

        for i in range(self.n_portfolios):
            level = self.risk_levels[i]
            # Target volatility scales with risk level (0.03 for C1, 0.15 for C5)
            target_vols[i] = 0.03 + (level - 1) * 0.03

        target_vars = target_vols ** 2

        # Construct design matrix for least squares
        # Each row i corresponds to portfolio i
        # We need to solve: Σ_elements that satisfy w_i^T Σ w_i = target_var_i

        # Since Σ is symmetric 4x4, we have 10 unique elements
        # Σ = [[σ11, σ12, σ13, σ14],
        #      [σ12, σ22, σ23, σ24],
        #      [σ13, σ23, σ33, σ34],
        #      [σ14, σ24, σ34, σ44]]

        # For each portfolio: w^T Σ w = Σ_ij w_i w_j
        # This is linear in Σ elements

        # Design matrix A: (n_portfolios x 10)
        A = np.zeros((self.n_portfolios, 10))

        for i in range(self.n_portfolios):
            w = self.weights[i, :]
            # Construct row for unique Σ elements
            idx = 0
            for j in range(self.n_assets):
                for k in range(j, self.n_assets):
                    if j == k:
                        A[i, idx] = w[j] ** 2
                    else:
                        A[i, idx] = 2 * w[j] * w[k]  # Off-diagonal appears twice
                    idx += 1

        # Solve least squares: A @ σ_vec = target_vars
        sigma_vec, residuals, rank, s = np.linalg.lstsq(A, target_vars, rcond=None)

        # Reconstruct covariance matrix from vector
        cov_estimated = np.zeros((self.n_assets, self.n_assets))
        idx = 0
        for j in range(self.n_assets):
            for k in range(j, self.n_assets):
                cov_estimated[j, k] = sigma_vec[idx]
                cov_estimated[k, j] = sigma_vec[idx]
                idx += 1

        # Enforce PSD constraint
        cov_estimated = nearest_psd_matrix(cov_estimated)

        return cov_estimated

    def _reverse_optimize_moment_matching(self) -> np.ndarray:
        """
        Estimate Σ using sample moment matching.
        使用样本矩匹配估计协方差矩阵。

        Strategy:
        1. Treat each portfolio weight as a "sample"
        2. Compute sample covariance of weights
        3. Scale by risk level
        """
        # Weight each portfolio by its risk level
        # Higher risk portfolios get more weight
        risk_weights = self.risk_levels / self.risk_levels.sum()

        # Compute weighted mean
        w_mean = (self.weights.T @ risk_weights).reshape(-1, 1)

        # Compute weighted covariance
        centered = self.weights - w_mean.T
        cov_estimated = (centered.T @ np.diag(risk_weights) @ centered)

        # Scale by overall variance to get reasonable magnitudes
        cov_estimated = cov_estimated * 0.01  # Scaling factor

        # Enforce PSD
        cov_estimated = nearest_psd_matrix(cov_estimated)

        return cov_estimated

    def get_portfolio_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of portfolio allocations.
        获取投资组合配置的汇总统计。

        Returns:
        --------
        pd.DataFrame
            Summary by risk level
        """
        summary = []

        for level in range(1, 6):
            mask = self.risk_levels == level
            weights_level = self.weights[mask]
            n_portfolios = weights_level.shape[0]

            summary.append({
                'risk_level': f'C{level}',
                'n_portfolios': n_portfolios,
                'mean_bond': weights_level[:, 0].mean(),
                'mean_cash': weights_level[:, 1].mean(),
                'mean_commodity': weights_level[:, 2].mean(),
                'mean_equity': weights_level[:, 3].mean(),
                'std_bond': weights_level[:, 0].std(),
                'std_cash': weights_level[:, 1].std(),
                'std_commodity': weights_level[:, 2].std(),
                'std_equity': weights_level[:, 3].std()
            })

        return pd.DataFrame(summary)
