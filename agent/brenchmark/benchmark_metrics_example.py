"""
Example implementation for computing Median and σ-based benchmarks
on 1Y / 2Y / 3Y returns. Compatible with Python 3.9, pandas 2.2.3,
numpy 1.26.4, and no SciPy dependencies.

Steps:
1) Read XLS source data (columns: product_id, product_name, category,
   ret_1y, ret_2y, ret_3y). Percentage values like "6.2%" are allowed.
2) Compute per-category medians, means, and population std (ddof=0)
   for each return column.
3) Flag products that beat the median on each horizon and pass the
   "at least 2 horizons beat median" rule.
4) Assign σ-based levels on 1Y (Top / Neutral / Lagging).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

RET_COLS: List[str] = ["ret_1y", "ret_2y", "ret_3y"]


def _parse_return(value) -> float:
    """Convert values like '6.2%' or 0.062 to float (decimal)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
        return float(text) / 100.0
    return float(text)


def load_benchmark_data(path: Path) -> pd.DataFrame:
    """Load XLS data and normalize return columns to decimals."""
    df = pd.read_excel(path, engine="xlrd")
    missing = [col for col in ["product_id", "product_name", "category"] if col not in df.columns]
    missing_returns = [col for col in RET_COLS if col not in df.columns]
    if missing or missing_returns:
        raise ValueError("Missing required columns: %s %s" % (missing, missing_returns))

    for col in RET_COLS:
        df[col] = df[col].apply(_parse_return)
    return df


def compute_group_stats(df: pd.DataFrame, group_col: str = "category") -> pd.DataFrame:
    """Compute median, mean, and population std per group for each return column."""
    grouped = df.groupby(group_col)
    frames = []
    for col in RET_COLS:
        med = grouped[col].median().rename(col + "_median")
        mean = grouped[col].mean().rename(col + "_mean")
        std_pop = grouped[col].apply(lambda x: float(np.std(x.dropna(), ddof=0))).rename(col + "_std")
        frames.extend([med, mean, std_pop])
    return pd.concat(frames, axis=1).reset_index()


def classify_sigma_level(ret_value: float, mu: float, sigma: float) -> str:
    """Return σ-based level for one value."""
    if pd.isna(ret_value) or pd.isna(mu) or pd.isna(sigma):
        return "Unknown"
    if ret_value < mu:
        return "Lagging"
    if ret_value < mu + sigma:
        return "Neutral"
    return "Top"


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Add median beat flags, pass_screen, and sigma_level_1y."""
    for horizon in ["1y", "2y", "3y"]:
        df["beat_median_" + horizon] = df["ret_" + horizon] > df["ret_" + horizon + "_median"]

    beat_cols = ["beat_median_1y", "beat_median_2y", "beat_median_3y"]
    df["pass_screen"] = df[beat_cols].sum(axis=1) >= 2

    df["sigma_level_1y"] = df.apply(
        lambda row: classify_sigma_level(row["ret_1y"], row["ret_1y_mean"], row["ret_1y_std"]),
        axis=1,
    )
    return df


def process_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load, compute metrics, and save results."""
    raw = load_benchmark_data(input_path)
    stats = compute_group_stats(raw, group_col="category")
    enriched = raw.merge(stats, on="category", how="left")
    enriched = apply_rules(enriched)
    enriched.to_csv(output_path, index=False)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute benchmark metrics for products.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "mock_products.xls",
        help="Path to XLS input data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "mock_products_processed.csv",
        help="Path to write processed CSV.",
    )
    args = parser.parse_args()

    result = process_file(args.input, args.output)
    print("Processed rows:", len(result))
    print("Saved to:", args.output)
    print("\nPreview (top 5 rows):")
    print(result.head())


if __name__ == "__main__":
    main()

