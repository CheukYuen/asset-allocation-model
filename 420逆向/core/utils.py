"""
Utility functions for covariance matrix operations.
协方差矩阵操作的工具函数。

Python 3.9 compatible - no scipy dependencies.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import re


def load_existing_covariance(prompt_file: str) -> Dict[str, np.ndarray]:
    """
    Load existing covariance matrix, volatility vector, and correlation matrix from prompt.md.
    从 prompt.md 文件加载现有的协方差矩阵、波动率向量和相关性矩阵。

    Parameters:
    -----------
    prompt_file : str
        Path to prompt.md file

    Returns:
    --------
    Dict with keys:
        - 'covariance': 4x4 covariance matrix (Σ)
        - 'volatility': 4x1 volatility vector (σ)
        - 'correlation': 4x4 correlation matrix (ρ)
        - 'asset_names': List of asset names in order
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract volatility vector (σ)
    sigma_pattern = r'asset_class,sigma_ann\n((?:.*\n)*?)```'
    sigma_match = re.search(sigma_pattern, content)

    if sigma_match:
        sigma_text = sigma_match.group(1).strip()
        sigma_lines = [line.split(',') for line in sigma_text.split('\n')]
        sigma_dict = {line[0]: float(line[1]) for line in sigma_lines}
    else:
        raise ValueError("Could not find volatility vector in prompt.md")

    # Extract covariance matrix (Σ)
    cov_pattern = r'## 年化协方差矩阵.*?\n```csv\n(.*?)\n```'
    cov_match = re.search(cov_pattern, content, re.DOTALL)

    if cov_match:
        cov_text = cov_match.group(1).strip()
        cov_lines = cov_text.split('\n')
        header = cov_lines[0].split(',')[1:]  # Asset names

        cov_matrix = []
        for line in cov_lines[1:]:
            parts = line.split(',')
            values = [float(x) for x in parts[1:]]
            cov_matrix.append(values)

        cov_matrix = np.array(cov_matrix)
    else:
        raise ValueError("Could not find covariance matrix in prompt.md")

    # Extract correlation matrix (ρ)
    corr_pattern = r'## 长期相关性矩阵.*?\n```csv\n(.*?)\n```'
    corr_match = re.search(corr_pattern, content, re.DOTALL)

    if corr_match:
        corr_text = corr_match.group(1).strip()
        corr_lines = corr_text.split('\n')

        corr_matrix = []
        for line in corr_lines[1:]:
            parts = line.split(',')
            values = [float(x) for x in parts[1:]]
            corr_matrix.append(values)

        corr_matrix = np.array(corr_matrix)
    else:
        raise ValueError("Could not find correlation matrix in prompt.md")

    # Map asset names: bond, cash, commodity, equity (sorted alphabetically in header)
    # Create volatility vector in same order
    asset_order = ['BOND', 'CASH', 'COMMODITY', 'EQUITY']
    sigma_vector = np.array([sigma_dict[asset] for asset in asset_order])

    return {
        'covariance': cov_matrix,
        'volatility': sigma_vector,
        'correlation': corr_matrix,
        'asset_names': asset_order
    }


def nearest_psd_matrix(sigma: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Project a matrix to the nearest positive semi-definite matrix.
    将矩阵投影到最近的半正定矩阵。

    Uses eigenvalue decomposition to clip negative eigenvalues.
    使用特征值分解裁剪负特征值。

    Parameters:
    -----------
    sigma : np.ndarray
        Input matrix (may not be PSD)
    epsilon : float
        Minimum eigenvalue threshold (default: 1e-8)

    Returns:
    --------
    np.ndarray
        Nearest PSD matrix
    """
    # Ensure symmetry
    sigma_sym = ensure_symmetry(sigma)

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(sigma_sym)

    # Clip negative eigenvalues
    eigvals_pos = np.maximum(eigvals, epsilon)

    # Reconstruct matrix
    sigma_psd = eigvecs @ np.diag(eigvals_pos) @ eigvecs.T

    # Ensure symmetry again (numerical stability)
    sigma_psd = ensure_symmetry(sigma_psd)

    return sigma_psd


def ensure_symmetry(matrix: np.ndarray) -> np.ndarray:
    """
    Ensure matrix is symmetric.
    确保矩阵对称。

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix

    Returns:
    --------
    np.ndarray
        Symmetric matrix: (A + A^T) / 2
    """
    return (matrix + matrix.T) / 2


def cov_to_corr(covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert covariance matrix to correlation matrix and volatility vector.
    将协方差矩阵转换为相关性矩阵和波动率向量。

    Formula: ρ_ij = Σ_ij / (σ_i * σ_j)

    Parameters:
    -----------
    covariance : np.ndarray
        Covariance matrix (nxn)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (correlation_matrix, volatility_vector)
    """
    # Extract volatilities (standard deviations) from diagonal
    volatility = np.sqrt(np.diag(covariance))

    # Compute correlation matrix
    # ρ = D^(-1) Σ D^(-1) where D = diag(σ)
    inv_vol = 1.0 / volatility
    correlation = (covariance * inv_vol[:, np.newaxis]) * inv_vol[np.newaxis, :]

    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(correlation, 1.0)

    return correlation, volatility


def corr_to_cov(correlation: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix and volatility vector to covariance matrix.
    将相关性矩阵和波动率向量转换为协方差矩阵。

    Formula: Σ_ij = ρ_ij * σ_i * σ_j

    Parameters:
    -----------
    correlation : np.ndarray
        Correlation matrix (nxn)
    volatility : np.ndarray
        Volatility vector (n,)

    Returns:
    --------
    np.ndarray
        Covariance matrix (nxn)
    """
    # Σ = D ρ D where D = diag(σ)
    covariance = (correlation * volatility[:, np.newaxis]) * volatility[np.newaxis, :]

    return covariance


def is_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is positive semi-definite.
    检查矩阵是否为半正定。

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix
    tol : float
        Tolerance for negative eigenvalues

    Returns:
    --------
    bool
        True if all eigenvalues >= -tol
    """
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= -tol)


def save_matrix_to_csv(matrix: np.ndarray, filepath: str,
                       row_names: Optional[list] = None,
                       col_names: Optional[list] = None) -> None:
    """
    Save a matrix to CSV with optional row and column names.
    将矩阵保存为CSV文件,可选行列名称。

    Parameters:
    -----------
    matrix : np.ndarray
        Matrix to save
    filepath : str
        Output file path
    row_names : list, optional
        Row labels
    col_names : list, optional
        Column labels
    """
    df = pd.DataFrame(matrix, index=row_names, columns=col_names)
    df.to_csv(filepath)


def load_matrix_from_csv(filepath: str) -> Tuple[np.ndarray, list, list]:
    """
    Load a matrix from CSV with row and column names.
    从CSV文件加载矩阵及行列名称。

    Parameters:
    -----------
    filepath : str
        Input file path

    Returns:
    --------
    Tuple[np.ndarray, list, list]
        (matrix, row_names, col_names)
    """
    df = pd.read_csv(filepath, index_col=0)
    matrix = df.values
    row_names = list(df.index)
    col_names = list(df.columns)

    return matrix, row_names, col_names
