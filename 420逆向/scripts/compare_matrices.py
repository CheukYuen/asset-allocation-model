"""
Script to display side-by-side comparison of covariance matrices.
显示协方差矩阵对比的脚本。

Usage:
    python compare_matrices.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
core_dir = project_dir / 'core'
sys.path.insert(0, str(core_dir))

import numpy as np
import pandas as pd

from utils import load_matrix_from_csv, load_existing_covariance


def format_matrix_comparison(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    asset_names: list,
    title1: str = "Matrix 1",
    title2: str = "Matrix 2"
) -> str:
    """
    Format side-by-side matrix comparison.
    格式化矩阵对比显示。

    Parameters:
    -----------
    matrix1 : np.ndarray
        First matrix
    matrix2 : np.ndarray
        Second matrix
    asset_names : list
        Asset names
    title1 : str
        Title for first matrix
    title2 : str
        Title for second matrix

    Returns:
    --------
    str
        Formatted comparison text
    """
    lines = []

    lines.append("=" * 120)
    lines.append(f"{title1:<50} | {title2}")
    lines.append("=" * 120)

    # Header row
    header = "       "
    for name in asset_names:
        header += f"{name:>12}"
    lines.append(f"{header:<50} | {header}")
    lines.append("-" * 120)

    # Data rows
    for i, row_name in enumerate(asset_names):
        row_str1 = f"{row_name:>6} "
        row_str2 = f"{row_name:>6} "

        for j in range(len(asset_names)):
            row_str1 += f"{matrix1[i, j]:>12.6f}"
            row_str2 += f"{matrix2[i, j]:>12.6f}"

        lines.append(f"{row_str1:<50} | {row_str2}")

    lines.append("=" * 120)

    return "\n".join(lines)


def format_difference_matrix(
    diff: np.ndarray,
    asset_names: list
) -> str:
    """
    Format difference matrix with highlighting.
    格式化差异矩阵并高亮显示。

    Parameters:
    -----------
    diff : np.ndarray
        Difference matrix
    asset_names : list
        Asset names

    Returns:
    --------
    str
        Formatted text
    """
    lines = []

    lines.append("=" * 80)
    lines.append("DIFFERENCE MATRIX (Estimated - Existing)")
    lines.append("差异矩阵 (估计 - 现有)")
    lines.append("=" * 80)

    # Header
    header = "       "
    for name in asset_names:
        header += f"{name:>12}"
    lines.append(header)
    lines.append("-" * 80)

    # Data rows
    for i, row_name in enumerate(asset_names):
        row_str = f"{row_name:>6} "

        for j in range(len(asset_names)):
            value = diff[i, j]
            # Highlight large differences
            if abs(value) > 0.01:
                row_str += f"{value:>12.6f}*"  # Mark with asterisk
            else:
                row_str += f"{value:>12.6f} "

        lines.append(row_str)

    lines.append("=" * 80)
    lines.append("* = absolute difference > 0.01")
    lines.append("")

    return "\n".join(lines)


def format_percentage_difference(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    asset_names: list
) -> str:
    """
    Format percentage difference matrix.
    格式化百分比差异矩阵。

    Parameters:
    -----------
    matrix1 : np.ndarray
        Estimated matrix
    matrix2 : np.ndarray
        Existing matrix (denominator)
    asset_names : list
        Asset names

    Returns:
    --------
    str
        Formatted text
    """
    lines = []

    lines.append("=" * 80)
    lines.append("PERCENTAGE DIFFERENCE (100 * (Estimated - Existing) / Existing)")
    lines.append("百分比差异 (100 * (估计 - 现有) / 现有)")
    lines.append("=" * 80)

    # Header
    header = "       "
    for name in asset_names:
        header += f"{name:>12}"
    lines.append(header)
    lines.append("-" * 80)

    # Data rows
    for i, row_name in enumerate(asset_names):
        row_str = f"{row_name:>6} "

        for j in range(len(asset_names)):
            if abs(matrix2[i, j]) > 1e-10:
                pct_diff = 100 * (matrix1[i, j] - matrix2[i, j]) / matrix2[i, j]
                row_str += f"{pct_diff:>11.2f}%"
            else:
                row_str += f"{'N/A':>12}"

        lines.append(row_str)

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    """Main execution function."""
    print("=" * 80)
    print("COVARIANCE MATRIX COMPARISON")
    print("协方差矩阵对比")
    print("=" * 80)
    print()

    # File paths
    results_dir = project_dir / 'results'

    estimated_file = results_dir / 'estimated_covariance.csv'
    existing_file = results_dir / 'existing_covariance.csv'
    diff_file = results_dir / 'difference_matrix.csv'

    # Check if files exist
    if not estimated_file.exists():
        print(f"Error: {estimated_file} not found.")
        print("Please run run_reverse_optimization.py first.")
        return

    if not existing_file.exists():
        print(f"Error: {existing_file} not found.")
        print("Please run run_reverse_optimization.py first.")
        return

    # Load matrices
    print("Loading matrices...")
    cov_estimated, row_names, col_names = load_matrix_from_csv(str(estimated_file))
    cov_existing, _, _ = load_matrix_from_csv(str(existing_file))

    asset_names = row_names

    print(f"  Assets: {', '.join(asset_names)}")
    print()

    # Display side-by-side comparison
    print(format_matrix_comparison(
        cov_existing,
        cov_estimated,
        asset_names,
        title1="EXISTING COVARIANCE (from prompt.md)",
        title2="ESTIMATED COVARIANCE (from portfolios)"
    ))
    print()

    # Display difference matrix
    diff = cov_estimated - cov_existing
    print(format_difference_matrix(diff, asset_names))
    print()

    # Display percentage difference
    print(format_percentage_difference(cov_estimated, cov_existing, asset_names))
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    frobenius_norm = np.linalg.norm(diff, 'fro')
    frobenius_norm_existing = np.linalg.norm(cov_existing, 'fro')
    relative_frobenius = frobenius_norm / frobenius_norm_existing

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs_diff = np.abs(diff).max()
    mean_abs_diff = np.mean(np.abs(diff))

    print(f"Frobenius Norm of Difference:     {frobenius_norm:.6f}")
    print(f"Relative Frobenius Norm:           {relative_frobenius:.4f}")
    print(f"Root Mean Squared Error:           {rmse:.6e}")
    print(f"Mean Absolute Difference:          {mean_abs_diff:.6e}")
    print(f"Maximum Absolute Difference:       {max_abs_diff:.6f}")
    print()

    # Element-wise correlation
    vec_estimated = cov_estimated.flatten()
    vec_existing = cov_existing.flatten()
    correlation = np.corrcoef(vec_estimated, vec_existing)[0, 1]

    print(f"Element-wise Correlation:          {correlation:.6f}")
    print()

    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
