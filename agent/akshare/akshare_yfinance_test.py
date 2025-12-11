"""
Quick sanity tests for AkShare (CBCFPI) and yfinance (CSI 300 TR).

Notes:
- Python 3.9 compatible (no 3.10+ syntax).
- Requires: akshare, yfinance, pandas, numpy.
- Does not rely on SciPy (not available in prod).
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import yfinance as yf
except ImportError:
    yf = None


def _standardize_date_value(
    df: pd.DataFrame,
    date_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
) -> pd.DataFrame:
    """Align date/value columns regardless of field names."""
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    for col in date_keys:
        if col in df.columns:
            date_col = col
            break
    for col in value_keys:
        if col in df.columns:
            value_col = col
            break
    if date_col is None or value_col is None:
        missing = {
            "date_found": date_col is not None,
            "value_found": value_col is not None,
            "columns": list(df.columns),
        }
        raise ValueError(f"Cannot locate date/value columns: {missing}")

    out = (
        df[[date_col, value_col]]
        .rename(columns={date_col: "date", value_col: "value"})
        .copy()
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["date", "value"]).set_index("date").sort_index()
    return out


def fetch_cbcfpi(indicator: str = "全价", period: str = "总值") -> pd.DataFrame:
    """Fetch CBCFPI (total price) via AkShare and compute returns."""
    if ak is None:
        raise ImportError("akshare is required for CBCFPI test")

    raw = ak.bond_composite_index_cbond(indicator=indicator, period=period)
    aligned = _standardize_date_value(
        raw,
        date_keys=("date", "日期"),
        value_keys=("value", "全价指数", "全价指数(总值)", "全价指数-总值"),
    )
    aligned["ret_d"] = np.log(aligned["value"] / aligned["value"].shift(1))
    aligned["ret_m"] = aligned["ret_d"].resample("ME").sum()
    return aligned


def fetch_csi300_tr(
    start: str = "2010-01-01",
    ticker: str = "H00300.SS",
    fallback_tickers: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Fetch CSI 300 Total Return via yfinance and compute returns."""
    if yf is None:
        raise ImportError("yfinance is required for CSI 300 TR test")

    tickers_to_try = [ticker]
    if fallback_tickers:
        tickers_to_try.extend(list(fallback_tickers))

    last_err = None
    for tk in tickers_to_try:
        ticker_obj = yf.Ticker(tk)

        def _history_with_fallback() -> pd.DataFrame:
            # Use explicit start to avoid period warnings.
            return ticker_obj.history(start=start, interval="1d", auto_adjust=False)

        hist = _history_with_fallback()
        try:
            if "Close" not in hist.columns:
                raise ValueError(
                    "Missing Close column; columns=%s" % list(hist.columns)
                )
            df = hist.copy()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            if len(df) <= 1:
                raise ValueError("Not enough rows (<=1) for %s" % tk)
            df["ret_d"] = np.log(df["Close"] / df["Close"].shift(1))
            df["ret_m"] = df["ret_d"].resample("ME").sum()
            df.attrs["ticker_used"] = tk
            return df
        except Exception as exc:
            last_err = exc
            continue

    raise ValueError("All tickers failed: %s" % last_err)


def _print_summary(name: str, df: pd.DataFrame, value_col: str) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"rows: {len(df)}")
    ticker_used = df.attrs.get("ticker_used")
    if ticker_used:
        print(f"ticker_used: {ticker_used}")
    print("recent values:")
    print(df[[value_col]].tail(3))
    print("recent monthly log returns:")
    print(df["ret_m"].dropna().tail(3))


def _save_csv(df: pd.DataFrame, filename: str) -> None:
    path = Path(__file__).parent / filename
    df.to_csv(path, index_label="date")
    print(f"saved: {path.name} (rows={len(df)})")


def main() -> int:
    # AkShare CBCFPI
    try:
        cbcfpi = fetch_cbcfpi()
        _print_summary("CBCFPI (AkShare, 全价/总值)", cbcfpi, "value")
        _save_csv(cbcfpi, "cbcfpi_history.csv")
    except Exception as err:
        print(f"[cbcfi error] {err}", file=sys.stderr)

    # yfinance CSI300 TR
    try:
        csi300_tr = fetch_csi300_tr(
            fallback_tickers=("N00300.SS", "000300.SS", "510300.SS"),
        )
        _print_summary("CSI 300 TR (yfinance)", csi300_tr, "Close")
        _save_csv(csi300_tr, "csi300_tr_history.csv")
    except Exception as err:
        print(f"[csi300_tr error] {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

