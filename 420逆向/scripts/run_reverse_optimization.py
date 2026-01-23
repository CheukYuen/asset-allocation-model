"""
Main script to run reverse optimization and generate results.
运行反向优化并生成结果的主脚本。

Usage:
    python run_reverse_optimization.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import core modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
core_dir = project_dir / 'core'
sys.path.insert(0, str(core_dir))

import numpy as np
import pandas as pd

from covariance_estimator import CovarianceReverseOptimizer
from validation_metrics import generate_validation_report
from utils import (
    load_existing_covariance,
    save_matrix_to_csv,
    cov_to_corr,
    is_positive_semidefinite
)


def main():
    """Main execution function."""
    print("=" * 80)
    print("COVARIANCE REVERSE OPTIMIZATION")
    print("协方差矩阵反向优化")
    print("=" * 80)
    print()

    # Define file paths
    data_dir = project_dir / 'data' if (project_dir / 'data').exists() else project_dir
    portfolios_file = project_dir / '105.csv'
    prompt_file = project_dir / 'prompt.md'
    results_dir = project_dir / 'results'

    # Check if input files exist
    if not portfolios_file.exists():
        print(f"Error: Portfolio file not found: {portfolios_file}")
        return
    if not prompt_file.exists():
        print(f"Error: Prompt file not found: {prompt_file}")
        return

    print("Step 1: Loading portfolio data...")
    print(f"  Portfolio file: {portfolios_file}")
    optimizer = CovarianceReverseOptimizer(str(portfolios_file))
    print(f"  Loaded {optimizer.n_portfolios} portfolios")
    print(f"  Assets: {', '.join(optimizer.asset_names_sorted)}")
    print()

    # Display portfolio summary
    print("Portfolio Summary by Risk Level:")
    summary = optimizer.get_portfolio_summary()
    print(summary.to_string(index=False))
    print()

    print("Step 2: Loading existing covariance matrix...")
    print(f"  Prompt file: {prompt_file}")
    existing_data = load_existing_covariance(str(prompt_file))
    cov_existing = existing_data['covariance']
    vol_existing = existing_data['volatility']
    corr_existing = existing_data['correlation']
    asset_names = existing_data['asset_names']

    print("  Existing Covariance Matrix (Σ):")
    print(pd.DataFrame(cov_existing, index=asset_names, columns=asset_names).to_string())
    print()

    print("  Existing Volatility Vector (σ):")
    for i, name in enumerate(asset_names):
        print(f"    {name}: {vol_existing[i]:.6f}")
    print()

    # Check PSD
    is_psd = is_positive_semidefinite(cov_existing)
    print(f"  Is existing covariance PSD? {is_psd}")
    print()

    print("Step 3: Computing portfolio volatilities with existing Σ...")
    vols_existing = optimizer.compute_portfolio_volatility(cov_existing)
    print(f"  Portfolio volatility range: [{vols_existing.min():.6f}, {vols_existing.max():.6f}]")

    # Volatility by risk level
    print("  Average volatility by risk level:")
    for level in range(1, 6):
        mask = optimizer.risk_levels == level
        vol_mean = vols_existing[mask].mean()
        vol_std = vols_existing[mask].std()
        print(f"    C{level}: {vol_mean:.6f} ± {vol_std:.6f}")
    print()

    print("Step 4: Estimating risk aversion coefficients (λ)...")
    risk_aversion = optimizer.estimate_risk_aversion(cov_existing)
    print("  Risk aversion by level:")
    for level in range(1, 6):
        print(f"    C{level}: λ = {risk_aversion[level]:.6f}")
    print()

    print("Step 5: Running reverse optimization...")
    print("  Method: Least squares")
    cov_estimated = optimizer.reverse_optimize_covariance(
        cov_existing,
        method='least_squares'
    )

    print("  Estimated Covariance Matrix (Σ_estimated):")
    print(pd.DataFrame(cov_estimated, index=asset_names, columns=asset_names).to_string())
    print()

    # Check PSD
    is_psd_est = is_positive_semidefinite(cov_estimated)
    print(f"  Is estimated covariance PSD? {is_psd_est}")
    print()

    # Extract correlation and volatility from estimated covariance
    corr_estimated, vol_estimated_from_cov = cov_to_corr(cov_estimated)

    print("  Estimated Correlation Matrix (ρ_estimated):")
    print(pd.DataFrame(corr_estimated, index=asset_names, columns=asset_names).to_string())
    print()

    print("  Estimated Volatility Vector (σ_estimated):")
    for i, name in enumerate(asset_names):
        print(f"    {name}: {vol_estimated_from_cov[i]:.6f}")
    print()

    print("Step 6: Computing portfolio volatilities with estimated Σ...")
    vols_estimated = optimizer.compute_portfolio_volatility(cov_estimated)
    print(f"  Portfolio volatility range: [{vols_estimated.min():.6f}, {vols_estimated.max():.6f}]")

    # Volatility by risk level
    print("  Average volatility by risk level:")
    for level in range(1, 6):
        mask = optimizer.risk_levels == level
        vol_mean = vols_estimated[mask].mean()
        vol_std = vols_estimated[mask].std()
        print(f"    C{level}: {vol_mean:.6f} ± {vol_std:.6f}")
    print()

    print("Step 7: Generating validation report...")
    report_file = results_dir / 'validation_report.txt'
    report_text = generate_validation_report(
        cov_estimated=cov_estimated,
        cov_existing=cov_existing,
        vols_estimated=vols_estimated,
        vols_existing=vols_existing,
        risk_levels=optimizer.risk_levels,
        asset_names=asset_names,
        risk_aversion=risk_aversion,
        output_file=str(report_file)
    )

    print(f"  Validation report saved to: {report_file}")
    print()

    print("Step 8: Saving results...")

    # Save estimated covariance
    save_matrix_to_csv(
        cov_estimated,
        str(results_dir / 'estimated_covariance.csv'),
        row_names=asset_names,
        col_names=asset_names
    )
    print(f"  ✓ Saved: estimated_covariance.csv")

    # Save existing covariance (for comparison)
    save_matrix_to_csv(
        cov_existing,
        str(results_dir / 'existing_covariance.csv'),
        row_names=asset_names,
        col_names=asset_names
    )
    print(f"  ✓ Saved: existing_covariance.csv")

    # Save estimated correlation
    save_matrix_to_csv(
        corr_estimated,
        str(results_dir / 'estimated_correlation.csv'),
        row_names=asset_names,
        col_names=asset_names
    )
    print(f"  ✓ Saved: estimated_correlation.csv")

    # Save difference matrix
    diff_matrix = cov_estimated - cov_existing
    save_matrix_to_csv(
        diff_matrix,
        str(results_dir / 'difference_matrix.csv'),
        row_names=asset_names,
        col_names=asset_names
    )
    print(f"  ✓ Saved: difference_matrix.csv")

    # Save risk aversion coefficients
    risk_aversion_df = pd.DataFrame([
        {'risk_level': f'C{i}', 'lambda': risk_aversion[i]}
        for i in range(1, 6)
    ])
    risk_aversion_df.to_csv(results_dir / 'risk_aversion_by_level.csv', index=False)
    print(f"  ✓ Saved: risk_aversion_by_level.csv")

    # Save portfolio volatilities
    volatility_df = pd.DataFrame({
        'portfolio_id': range(1, optimizer.n_portfolios + 1),
        'risk_level': [f'C{level}' for level in optimizer.risk_levels],
        'lifecycle': optimizer.lifecycle_stages,
        'demand': optimizer.demand_types,
        'volatility_existing': vols_existing,
        'volatility_estimated': vols_estimated,
        'difference': vols_estimated - vols_existing,
        'pct_difference': (vols_estimated - vols_existing) / vols_existing * 100
    })
    volatility_df.to_csv(results_dir / 'portfolio_volatilities.csv', index=False)
    print(f"  ✓ Saved: portfolio_volatilities.csv")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Key Results:")
    print(f"  - Frobenius Norm Difference: {np.linalg.norm(diff_matrix, 'fro'):.6f}")
    print(f"  - Volatility RMSE: {np.sqrt(np.mean((vols_estimated - vols_existing)**2)):.6f}")
    print(f"  - Volatility Correlation: {np.corrcoef(vols_estimated, vols_existing)[0,1]:.6f}")
    print()
    print(f"Full validation report: {report_file}")
    print()


if __name__ == '__main__':
    main()
