"""
Validation metrics for comparing covariance matrices.
用于比较协方差矩阵的验证指标。

Python 3.9 compatible - no scipy dependencies.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from utils import cov_to_corr


def compare_covariance_matrices(
    cov_estimated: np.ndarray,
    cov_existing: np.ndarray,
    asset_names: list
) -> Dict[str, float]:
    """
    Compare estimated vs. existing covariance matrices.
    比较估计的协方差矩阵与现有协方差矩阵。

    Parameters:
    -----------
    cov_estimated : np.ndarray
        Estimated covariance matrix (4x4)
    cov_existing : np.ndarray
        Existing covariance matrix (4x4)
    asset_names : list
        Asset names for reporting

    Returns:
    --------
    Dict[str, float]
        Comparison metrics
    """
    # Element-wise difference
    diff = cov_estimated - cov_existing

    # Mean Squared Error
    mse = np.mean(diff ** 2)

    # Frobenius norm of difference
    frobenius_norm = np.linalg.norm(diff, 'fro')

    # Relative Frobenius norm
    frobenius_norm_existing = np.linalg.norm(cov_existing, 'fro')
    relative_frobenius = frobenius_norm / (frobenius_norm_existing + 1e-10)

    # Element-wise correlation
    # Flatten matrices and compute correlation
    vec_estimated = cov_estimated.flatten()
    vec_existing = cov_existing.flatten()

    correlation = np.corrcoef(vec_estimated, vec_existing)[0, 1]

    # Maximum absolute difference
    max_abs_diff = np.abs(diff).max()

    # Mean absolute percentage error (MAPE) for non-zero elements
    mask = np.abs(cov_existing) > 1e-10
    mape = np.mean(np.abs(diff[mask] / cov_existing[mask])) * 100

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'frobenius_norm': frobenius_norm,
        'relative_frobenius_norm': relative_frobenius,
        'element_correlation': correlation,
        'max_abs_diff': max_abs_diff,
        'mape_pct': mape
    }


def compare_volatilities(
    vols_estimated: np.ndarray,
    vols_existing: np.ndarray,
    portfolio_ids: np.ndarray
) -> Dict[str, float]:
    """
    Compare portfolio volatilities computed with different covariance matrices.
    比较使用不同协方差矩阵计算的投资组合波动率。

    Parameters:
    -----------
    vols_estimated : np.ndarray
        Volatilities using estimated covariance (105,)
    vols_existing : np.ndarray
        Volatilities using existing covariance (105,)
    portfolio_ids : np.ndarray
        Portfolio identifiers

    Returns:
    --------
    Dict[str, float]
        Comparison metrics
    """
    diff = vols_estimated - vols_existing

    # RMSE
    rmse = np.sqrt(np.mean(diff ** 2))

    # Mean absolute error
    mae = np.mean(np.abs(diff))

    # Mean absolute percentage error
    mape = np.mean(np.abs(diff / vols_existing)) * 100

    # Correlation
    correlation = np.corrcoef(vols_estimated, vols_existing)[0, 1]

    # Max absolute difference
    max_abs_diff = np.abs(diff).max()
    max_idx = np.abs(diff).argmax()

    return {
        'rmse': rmse,
        'mae': mae,
        'mape_pct': mape,
        'correlation': correlation,
        'max_abs_diff': max_abs_diff,
        'max_diff_portfolio_id': int(portfolio_ids[max_idx])
    }


def compare_correlations(
    corr_estimated: np.ndarray,
    corr_existing: np.ndarray,
    asset_names: list
) -> Dict[str, float]:
    """
    Compare correlation matrices.
    比较相关性矩阵。

    Parameters:
    -----------
    corr_estimated : np.ndarray
        Estimated correlation matrix (4x4)
    corr_existing : np.ndarray
        Existing correlation matrix (4x4)
    asset_names : list
        Asset names

    Returns:
    --------
    Dict[str, float]
        Comparison metrics
    """
    diff = corr_estimated - corr_existing

    # Mean Squared Error
    mse = np.mean(diff ** 2)

    # Frobenius norm
    frobenius_norm = np.linalg.norm(diff, 'fro')

    # Element-wise correlation (excluding diagonal)
    mask = ~np.eye(4, dtype=bool)
    vec_estimated = corr_estimated[mask]
    vec_existing = corr_existing[mask]

    correlation = np.corrcoef(vec_estimated, vec_existing)[0, 1]

    # Maximum absolute difference (off-diagonal)
    max_abs_diff = np.abs(diff[mask]).max()

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'frobenius_norm': frobenius_norm,
        'correlation': correlation,
        'max_abs_diff': max_abs_diff
    }


def eigenvalue_analysis(
    cov_estimated: np.ndarray,
    cov_existing: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compare eigenvalues (principal components) of covariance matrices.
    比较协方差矩阵的特征值(主成分)。

    Parameters:
    -----------
    cov_estimated : np.ndarray
        Estimated covariance matrix
    cov_existing : np.ndarray
        Existing covariance matrix

    Returns:
    --------
    Dict[str, np.ndarray]
        Eigenvalues and comparison metrics
    """
    eigvals_estimated = np.linalg.eigvalsh(cov_estimated)
    eigvals_existing = np.linalg.eigvalsh(cov_existing)

    # Sort in descending order
    eigvals_estimated = np.sort(eigvals_estimated)[::-1]
    eigvals_existing = np.sort(eigvals_existing)[::-1]

    # Variance explained
    var_explained_estimated = eigvals_estimated / eigvals_estimated.sum()
    var_explained_existing = eigvals_existing / eigvals_existing.sum()

    return {
        'eigvals_estimated': eigvals_estimated,
        'eigvals_existing': eigvals_existing,
        'var_explained_estimated': var_explained_estimated,
        'var_explained_existing': var_explained_existing
    }


def generate_validation_report(
    cov_estimated: np.ndarray,
    cov_existing: np.ndarray,
    vols_estimated: np.ndarray,
    vols_existing: np.ndarray,
    risk_levels: np.ndarray,
    asset_names: list,
    risk_aversion: Dict[int, float],
    output_file: str
) -> str:
    """
    Generate comprehensive validation report.
    生成综合验证报告。

    Parameters:
    -----------
    cov_estimated : np.ndarray
        Estimated covariance matrix
    cov_existing : np.ndarray
        Existing covariance matrix
    vols_estimated : np.ndarray
        Portfolio volatilities (estimated)
    vols_existing : np.ndarray
        Portfolio volatilities (existing)
    risk_levels : np.ndarray
        Risk levels (1-5)
    asset_names : list
        Asset names
    risk_aversion : Dict[int, float]
        Risk aversion by level
    output_file : str
        Output file path

    Returns:
    --------
    str
        Report text
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COVARIANCE MATRIX VALIDATION REPORT")
    report_lines.append("协方差矩阵验证报告")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1: Covariance Matrix Comparison
    report_lines.append("1. COVARIANCE MATRIX COMPARISON")
    report_lines.append("-" * 80)

    cov_metrics = compare_covariance_matrices(cov_estimated, cov_existing, asset_names)

    report_lines.append(f"Mean Squared Error (MSE):           {cov_metrics['mse']:.6e}")
    report_lines.append(f"Root Mean Squared Error (RMSE):     {cov_metrics['rmse']:.6e}")
    report_lines.append(f"Frobenius Norm of Difference:      {cov_metrics['frobenius_norm']:.6f}")
    report_lines.append(f"Relative Frobenius Norm:            {cov_metrics['relative_frobenius_norm']:.4f}")
    report_lines.append(f"Element-wise Correlation:           {cov_metrics['element_correlation']:.4f}")
    report_lines.append(f"Maximum Absolute Difference:        {cov_metrics['max_abs_diff']:.6f}")
    report_lines.append(f"Mean Absolute Percentage Error:     {cov_metrics['mape_pct']:.2f}%")
    report_lines.append("")

    # Interpretation
    report_lines.append("Interpretation:")
    if cov_metrics['relative_frobenius_norm'] < 0.1:
        report_lines.append("  ✓ Estimated covariance is very close to existing (< 10% difference)")
        recommendation = "KEEP EXISTING"
    elif cov_metrics['relative_frobenius_norm'] < 0.5:
        report_lines.append("  ≈ Moderate difference between matrices (10-50% difference)")
        recommendation = "CONSIDER HYBRID"
    else:
        report_lines.append("  ✗ Large difference between matrices (> 50% difference)")
        recommendation = "CONSIDER ESTIMATED"
    report_lines.append("")

    # Section 2: Correlation Matrix Comparison
    report_lines.append("2. CORRELATION STRUCTURE COMPARISON")
    report_lines.append("-" * 80)

    corr_estimated, vol_estimated_from_cov = cov_to_corr(cov_estimated)
    corr_existing, vol_existing_from_cov = cov_to_corr(cov_existing)

    corr_metrics = compare_correlations(corr_estimated, corr_existing, asset_names)

    report_lines.append(f"RMSE of Correlations:               {corr_metrics['rmse']:.6f}")
    report_lines.append(f"Correlation of Correlations:        {corr_metrics['correlation']:.4f}")
    report_lines.append(f"Max Absolute Difference:            {corr_metrics['max_abs_diff']:.6f}")
    report_lines.append("")

    # Section 3: Portfolio Volatility Comparison
    report_lines.append("3. PORTFOLIO VOLATILITY COMPARISON")
    report_lines.append("-" * 80)

    portfolio_ids = np.arange(1, len(vols_estimated) + 1)
    vol_metrics = compare_volatilities(vols_estimated, vols_existing, portfolio_ids)

    report_lines.append(f"RMSE of Volatilities:               {vol_metrics['rmse']:.6f}")
    report_lines.append(f"Mean Absolute Error:                {vol_metrics['mae']:.6f}")
    report_lines.append(f"Mean Absolute Percentage Error:     {vol_metrics['mape_pct']:.2f}%")
    report_lines.append(f"Correlation of Volatilities:        {vol_metrics['correlation']:.4f}")
    report_lines.append(f"Max Absolute Difference:            {vol_metrics['max_abs_diff']:.6f}")
    report_lines.append(f"Portfolio with Max Difference:      #{vol_metrics['max_diff_portfolio_id']}")
    report_lines.append("")

    # Volatility by risk level
    report_lines.append("Portfolio Volatility by Risk Level:")
    for level in range(1, 6):
        mask = risk_levels == level
        vol_est_mean = vols_estimated[mask].mean()
        vol_ext_mean = vols_existing[mask].mean()
        diff_pct = (vol_est_mean - vol_ext_mean) / vol_ext_mean * 100

        report_lines.append(
            f"  C{level}: Estimated={vol_est_mean:.4f}, "
            f"Existing={vol_ext_mean:.4f}, "
            f"Diff={diff_pct:+.2f}%"
        )
    report_lines.append("")

    # Section 4: Eigenvalue Analysis
    report_lines.append("4. EIGENVALUE ANALYSIS (PRINCIPAL COMPONENTS)")
    report_lines.append("-" * 80)

    eig_analysis = eigenvalue_analysis(cov_estimated, cov_existing)

    report_lines.append("Eigenvalues (Estimated):")
    for i, eigval in enumerate(eig_analysis['eigvals_estimated']):
        var_exp = eig_analysis['var_explained_estimated'][i]
        report_lines.append(f"  PC{i+1}: {eigval:.6f} ({var_exp*100:.2f}%)")
    report_lines.append("")

    report_lines.append("Eigenvalues (Existing):")
    for i, eigval in enumerate(eig_analysis['eigvals_existing']):
        var_exp = eig_analysis['var_explained_existing'][i]
        report_lines.append(f"  PC{i+1}: {eigval:.6f} ({var_exp*100:.2f}%)")
    report_lines.append("")

    # Section 5: Risk Aversion Estimates
    report_lines.append("5. RISK AVERSION COEFFICIENTS (λ)")
    report_lines.append("-" * 80)

    report_lines.append("Estimated Risk Aversion by Risk Level:")
    for level in range(1, 6):
        report_lines.append(f"  C{level}: λ = {risk_aversion[level]:.4f}")
    report_lines.append("")

    # Check monotonicity
    lambdas = [risk_aversion[i] for i in range(1, 6)]
    is_monotonic = all(lambdas[i] > lambdas[i+1] for i in range(len(lambdas)-1))
    if is_monotonic:
        report_lines.append("  ✓ Risk aversion decreases monotonically (C1 > C2 > ... > C5)")
    else:
        report_lines.append("  ✗ Risk aversion NOT monotonic - portfolios may not be mean-variance optimal")
    report_lines.append("")

    # Section 6: Final Recommendation
    report_lines.append("6. FINAL RECOMMENDATION")
    report_lines.append("-" * 80)

    report_lines.append(f"Primary Recommendation: {recommendation}")
    report_lines.append("")

    if recommendation == "KEEP EXISTING":
        report_lines.append("Rationale:")
        report_lines.append("  - Estimated covariance is very similar to existing")
        report_lines.append("  - Portfolios are approximately mean-variance optimal under existing Σ")
        report_lines.append("  - No strong evidence to change covariance matrix")
        report_lines.append("")
        report_lines.append("Action: Continue using existing covariance matrix from prompt.md")

    elif recommendation == "CONSIDER HYBRID":
        report_lines.append("Rationale:")
        report_lines.append("  - Moderate difference suggests both matrices have merit")
        report_lines.append("  - Portfolios partially deviate from mean-variance optimality")
        report_lines.append("  - Subjective factors (lifecycle, demand) influence allocations")
        report_lines.append("")
        report_lines.append("Action: Consider weighted average:")
        report_lines.append("  Σ_hybrid = α * Σ_existing + (1-α) * Σ_estimated")
        report_lines.append("  where α ∈ [0.5, 0.7] weights historical data more heavily")

    else:  # CONSIDER ESTIMATED
        report_lines.append("Rationale:")
        report_lines.append("  - Large difference indicates existing Σ may not reflect portfolio design")
        report_lines.append("  - Estimated Σ better fits the 105 portfolio allocations")
        report_lines.append("  - Financial planner intentions may differ from historical data")
        report_lines.append("")
        report_lines.append("Action: Consider replacing existing Σ with estimated Σ, or")
        report_lines.append("        Re-evaluate portfolio allocations for mean-variance consistency")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text
