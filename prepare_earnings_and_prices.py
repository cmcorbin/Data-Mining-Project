"""
This script reads the raw earnings, stock-price, and dataset summary files and then cleans the data. 
It then identifies the eligble stocks that have sufficient earnings-history(30 dates) and price data. 
Determines whether to consider the close price on the release date based on pre/post market announcement.  
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

EARNINGS_FILE = RAW_DIR / "earnings_latest.csv"
PRICES_FILE = RAW_DIR / "stock_prices_latest.csv"
SUMMARY_FILE = RAW_DIR / "dataset_summary.csv"

MIN_EARNINGS_REPORTS = 30
MAX_PRICE_GAP_DAYS = 7


# ------------------------------------------------------------
# Load and clean the CSV files
# ------------------------------------------------------------

def read_csv_with_nulls(path):
    return pd.read_csv(
        path,
        na_values=["NULL", "null", "Null", ""],
        keep_default_na=True
    )


def clean_earnings(path):
    earnings = read_csv_with_nulls(path)

    required_columns = {"symbol", "date", "eps_est", "eps"}
    missing_columns = required_columns - set(earnings.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing_columns)}"
        )

    earnings = earnings.copy()
    earnings["symbol"] = (
        earnings["symbol"].astype("string").str.strip().str.upper()
    )
    earnings["earnings_date"] = pd.to_datetime(
        earnings["date"], errors="coerce"
    )
    earnings["eps_est"] = pd.to_numeric(earnings["eps_est"], errors="coerce")
    earnings["eps"] = pd.to_numeric(earnings["eps"], errors="coerce")

    if "release_time" not in earnings.columns:
        earnings["release_time"] = pd.NA

    earnings["release_time"] = (
        earnings["release_time"].astype("string").str.strip().str.lower()
    )
    earnings.loc[
        ~earnings["release_time"].isin(["pre", "post"]), "release_time"
    ] = pd.NA

    if "qtr" not in earnings.columns:
        earnings["qtr"] = pd.NA

    earnings = earnings.dropna(subset=["symbol", "earnings_date"])
    earnings = earnings.drop_duplicates(
        subset=["symbol", "earnings_date"], keep="first"
    )
    earnings = earnings.sort_values(
        ["symbol", "earnings_date"]
    ).reset_index(drop=True)

    return earnings[
        ["symbol", "earnings_date", "qtr", "eps_est", "eps", "release_time"]
    ]


def clean_prices(path):
    prices = read_csv_with_nulls(path)

    required_columns = {"symbol", "date", "close_adjusted"}
    missing_columns = required_columns - set(prices.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing_columns)}"
        )

    prices = prices.copy()
    prices["symbol"] = (
        prices["symbol"].astype("string").str.strip().str.upper()
    )
    prices["price_date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close_adjusted"] = pd.to_numeric(
        prices["close_adjusted"], errors="coerce"
    )

    prices = prices.dropna(
        subset=["symbol", "price_date", "close_adjusted"]
    )
    prices = prices[prices["close_adjusted"] > 0]
    prices = prices.drop_duplicates(
        subset=["symbol", "price_date"], keep="first"
    )
    prices = prices.sort_values(
        ["symbol", "price_date"]
    ).reset_index(drop=True)

    return prices[["symbol", "price_date", "close_adjusted"]]


# ------------------------------------------------------------
# Identify eligible symbols
# ------------------------------------------------------------

def get_eligible_symbols(earnings, prices, summary_path):
    earnings_counts = earnings.groupby("symbol")["earnings_date"].nunique()
    earnings_eligible = set(
        earnings_counts[earnings_counts >= MIN_EARNINGS_REPORTS].index
    )

    if summary_path.exists():
        summary = read_csv_with_nulls(summary_path)
        if {"symbol", "total_earnings"}.issubset(summary.columns):
            summary["symbol"] = (
                summary["symbol"].astype("string").str.strip().str.upper()
            )
            summary["total_earnings"] = pd.to_numeric(
                summary["total_earnings"], errors="coerce"
            )
            summary_eligible = set(
                summary.loc[
                    summary["total_earnings"] >= MIN_EARNINGS_REPORTS, "symbol"
                ].dropna()
            )
            earnings_eligible = earnings_eligible.intersection(summary_eligible)

    price_symbols = set(prices["symbol"].dropna().unique())
    return sorted(earnings_eligible.intersection(price_symbols))


# ------------------------------------------------------------
# Match earnings events to prices 
# ------------------------------------------------------------

def match_earnings_to_prices(earnings, prices):
    """
    For each event, use binary search within that symbol's sorted price dates.
    For 'post' earnings announcemnts same-day close is used, while for 'pre' or unknown same day closed the previous trading date is used. 
    """
    prices_by_symbol = prices.groupby("symbol", sort=False)

    # Add output columns once rather than repeatedly concatenating DataFrames.
    matched = earnings.copy()
    matched["matched_price_date"] = pd.NaT
    matched["matched_close_adjusted"] = np.nan

    # Work on each ticker's relatively small event list.
    for symbol, event_index in matched.groupby("symbol", sort=False).groups.items():
        if symbol not in prices_by_symbol.indices:
            continue

        stock_prices = prices_by_symbol.get_group(symbol)
        price_dates = stock_prices["price_date"].to_numpy(dtype="datetime64[ns]")
        price_values = stock_prices["close_adjusted"].to_numpy(dtype=float)

        event_dates = matched.loc[event_index, "earnings_date"].to_numpy(
            dtype="datetime64[ns]"
        )
        release_times = (
            matched.loc[event_index, "release_time"]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        # Default to strictly earlier price dates.
        positions = np.searchsorted(price_dates, event_dates, side="left") - 1

        # For after-market announcements, same-day closing price is valid.
        post_mask = release_times == "post"
        if post_mask.any():
            positions[post_mask] = (
                np.searchsorted(
                    price_dates, event_dates[post_mask], side="right"
                ) - 1
            )

        valid = positions >= 0
        valid_event_indices = np.asarray(list(event_index))[valid]

        if len(valid_event_indices) > 0:
            valid_positions = positions[valid]
            matched.loc[valid_event_indices, "matched_price_date"] = (
                price_dates[valid_positions]
            )
            matched.loc[valid_event_indices, "matched_close_adjusted"] = (
                price_values[valid_positions]
            )

    matched["price_gap_days"] = (
        matched["earnings_date"] - matched["matched_price_date"]
    ).dt.days
    matched["has_recent_price_match"] = (
        matched["price_gap_days"].between(0, MAX_PRICE_GAP_DAYS)
    )

    return matched.sort_values(
        ["symbol", "earnings_date"]
    ).reset_index(drop=True)


# ------------------------------------------------------------
# Run the pipeline and save outputs
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [EARNINGS_FILE, PRICES_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. Put the CSV in data/raw/."
            )

    print("Loading and cleaning earnings data...", flush=True)
    earnings = clean_earnings(EARNINGS_FILE)

    print("Loading and cleaning stock-price data...", flush=True)
    prices = clean_prices(PRICES_FILE)

    print(f"Clean earnings rows: {len(earnings):,}", flush=True)
    print(f"Clean price rows:    {len(prices):,}", flush=True)

    eligible_symbols = get_eligible_symbols(
        earnings, prices, SUMMARY_FILE
    )
    print(
        f"Eligible symbols (at least {MIN_EARNINGS_REPORTS} earnings records "
        f"and price data): {len(eligible_symbols):,}",
        flush=True
    )

    eligible_earnings = earnings[
        earnings["symbol"].isin(eligible_symbols)
    ].copy()
    eligible_prices = prices[
        prices["symbol"].isin(eligible_symbols)
    ].copy()

    # Save the eligble earnings and prices before matching
    earnings.to_csv(OUTPUT_DIR / "cleaned_earnings.csv", index=False)
    prices.to_csv(OUTPUT_DIR / "cleaned_prices.csv", index=False)
    eligible_earnings.to_csv(
        OUTPUT_DIR / "eligible_earnings.csv", index=False
    )
    eligible_prices.to_csv(
        OUTPUT_DIR / "eligible_prices.csv", index=False
    )

    print("Matching earnings events to stock prices...", flush=True)
    matched = match_earnings_to_prices(
        eligible_earnings, eligible_prices
    )
    matched.to_csv(
        OUTPUT_DIR / "earnings_with_price_match.csv", index=False
    )

    print("\nOutput files saved to:", OUTPUT_DIR, flush=True)
    print(f"Events matched to a price row: "
          f"{matched['matched_price_date'].notna().sum():,}", flush=True)
    print(f"Events with recent price match (<= {MAX_PRICE_GAP_DAYS} days): "
          f"{matched['has_recent_price_match'].sum():,}", flush=True)
    print(f"Events with no price match: "
          f"{matched['matched_price_date'].isna().sum():,}", flush=True)
    print(
        "\nThis script covers loading/cleaning and event-to-price matching. "
        "It does not yet calculate earnings-beat labels, previous-quarter "
        "features, or 10-trading-day momentum.",
        flush=True
    )


if __name__ == "__main__":
    main()
