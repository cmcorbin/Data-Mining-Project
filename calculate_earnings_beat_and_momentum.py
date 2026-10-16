
"""
This script calculates the earnings suprise and beat/miss labels.  
Adds the previous available earnings event labels for beats and misses. 
Calculate momentum for a 10 day period pre earnings
Flag and remove extreme price movements due to bad data
"""
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

EARNINGS_FILE = (
    BASE_DIR / "data" / "processed" / "earnings_with_price_match.csv"
)

PRICES_FILE = (
    BASE_DIR / "data" / "processed" / "eligible_prices.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "final_earnings_event_dataset.csv"
)

MOMENTUM_TRADING_DAYS = 10
EXTREME_DAILY_MOVE_THRESHOLD = 0.50


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

# Calculate adjusted-price momentum over 10 trading-day intervals
# Flag a window if an adjusted daily return exceeds 50% either direction
def calculate_event_momentum(event, price_lookup):
    symbol = event["symbol"]
    matched_date = event["_matched_price_date"]

    result = {
        "momentum_start_date": pd.NaT,
        "momentum_end_price_date": pd.NaT,
        "momentum_start_close": np.nan,
        "momentum_end_close": np.nan,
        "momentum_10_trading_days": np.nan,
        "momentum_window_extreme_daily_move": pd.NA,
        "momentum_window_valid": False,
    }

    if pd.isna(matched_date) or symbol not in price_lookup:
        return result

    prices = price_lookup[symbol]

    # Find the last trading date on or before the matched price date.
    end_position = (
        prices["price_date"].searchsorted(matched_date, side="right") - 1
    )

    if end_position < MOMENTUM_TRADING_DAYS:
        return result

    start_position = end_position - MOMENTUM_TRADING_DAYS

    window = prices.iloc[start_position:end_position + 1].copy()

    if len(window) != MOMENTUM_TRADING_DAYS + 1:
        return result

    closes = window["close_adjusted"].to_numpy(dtype=float)

    if not np.isfinite(closes).all() or (closes <= 0).any():
        return result

    daily_returns = closes[1:] / closes[:-1] - 1.0

    #Flag large daily moves
    extreme_daily_move = bool(
        np.any(np.abs(daily_returns) > EXTREME_DAILY_MOVE_THRESHOLD)
    )

    momentum = closes[-1] / closes[0] - 1.0

    result.update({
        "momentum_start_date": window["price_date"].iloc[0],
        "momentum_end_price_date": window["price_date"].iloc[-1],
        "momentum_start_close": closes[0],
        "momentum_end_close": closes[-1],
        "momentum_10_trading_days": momentum,
        "momentum_window_extreme_daily_move": extreme_daily_move,
        "momentum_window_valid": True,
    })

    return result


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def main():
    print("Loading matched earnings events...")

    #read in and initalize data 
    earnings = pd.read_csv(EARNINGS_FILE)

    required_columns = {
        "symbol",
        "earnings_date",
        "eps_est",
        "eps",
        "matched_price_date",
    }

    missing_columns = required_columns - set(earnings.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required earnings columns: {sorted(missing_columns)}"
        )

    earnings["symbol"] = earnings["symbol"].astype(str).str.strip()
    earnings["earnings_date"] = pd.to_datetime(
        earnings["earnings_date"], errors="coerce"
    )
    earnings["_matched_price_date"] = pd.to_datetime(
        earnings["matched_price_date"], errors="coerce"
    )

    earnings["eps_est"] = to_numeric(earnings["eps_est"])
    earnings["eps"] = to_numeric(earnings["eps"])

    print("Loading adjusted stock prices...")

    prices = pd.read_csv(
        PRICES_FILE,
        usecols=["symbol", "price_date", "close_adjusted"],
    )

    prices["symbol"] = prices["symbol"].astype(str).str.strip()
    prices["price_date"] = pd.to_datetime(
        prices["price_date"], errors="coerce"
    )
    prices["close_adjusted"] = to_numeric(prices["close_adjusted"])

    prices = prices.dropna(
        subset=["symbol", "price_date", "close_adjusted"]
    )

    prices = prices[prices["close_adjusted"] > 0].copy()

    prices = (
        prices.sort_values(["symbol", "price_date"])
        .drop_duplicates(["symbol", "price_date"], keep="last")
        .reset_index(drop=True)
    )

    price_lookup = {
        symbol: group.reset_index(drop=True)
        for symbol, group in prices.groupby("symbol", sort=False)
    }

    print("Calculating EPS beat/miss labels...")

    earnings["eps_surprise"] = earnings["eps"] - earnings["eps_est"]

    # Find the valid earnings
    earnings["eps_surprise_pct"] = np.where(
        earnings["eps"].notna()
        & earnings["eps_est"].notna()
        & (earnings["eps_est"] != 0),
        earnings["eps_surprise"] / earnings["eps_est"].abs(),
        np.nan,
    )

    usable_eps = earnings["eps"].notna() & earnings["eps_est"].notna()

    earnings["eps_beat"] = pd.Series(
        pd.NA, index=earnings.index, dtype="Int64"
    )

    earnings.loc[usable_eps, "eps_beat"] = (
        earnings.loc[usable_eps, "eps"]
        > earnings.loc[usable_eps, "eps_est"]
    ).astype("int64")

    print("Calculating previous-event beat/miss labels...")

    # Sort earnings values
    earnings = (
        earnings.sort_values(
            ["symbol", "earnings_date", "_matched_price_date"],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # Grab the previous earnings label
    earnings["previous_quarter_beat"] = (
        earnings.groupby("symbol", sort=False)["eps_beat"]
        .shift(1)
        .astype("Int64")
    )

    print("Calculating adjusted-price momentum...")

    momentum_results = []

    for event in earnings.to_dict("records"):
        momentum_results.append(
            calculate_event_momentum(event, price_lookup)
        )

    momentum_df = pd.DataFrame(momentum_results)

    output = pd.concat(
        [
            earnings.drop(columns=["_matched_price_date"]),
            momentum_df,
        ],
        axis=1,
    )

    for column in [
        "momentum_window_extreme_daily_move",
        "momentum_window_valid",
    ]:
        output[column] = output[column].astype("boolean")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Summary of results
    # --------------------------------------------------------

    momentum_valid = output["momentum_window_valid"].fillna(False)
    extreme_flags = (
        output["momentum_window_extreme_daily_move"].fillna(False)
    )

    usable_eps_count = int(usable_eps.sum())
    beat_count = int((output["eps_beat"] == 1).sum())
    nonbeat_count = int((output["eps_beat"] == 0).sum())

    print(f"\nFeature dataset saved to:\n{OUTPUT_FILE}")
    print(f"\nTotal earnings events: {len(output):,}")
    print(f"Events with usable actual and estimated EPS: {usable_eps_count:,}")
    print(f"Earnings beats: {beat_count:,}")
    print(f"Non-beats: {nonbeat_count:,}")

    print(
        "Events with previous-event beat/miss label: "
        f"{output['previous_quarter_beat'].notna().sum():,}"
    )

    print(
        "Events with valid 10-trading-day momentum: "
        f"{momentum_valid.sum():,}"
    )

    print(
        "Momentum windows with extreme adjusted daily move: "
        f"{(extreme_flags & momentum_valid).sum():,}"
    )

    if momentum_valid.any():
        momentum = output.loc[
            momentum_valid, "momentum_10_trading_days"
        ]

        print("\n10-trading-day momentum summary:")
        print(
            momentum.describe(
                percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
            ).to_string()
        )

        columns_to_show = [
            "symbol",
            "earnings_date",
            "momentum_start_date",
            "momentum_end_price_date",
            "momentum_10_trading_days",
            "momentum_window_extreme_daily_move",
        ]

        print("\nLargest positive momentum observations:")
        print(
            output.loc[momentum_valid]
            .nlargest(15, "momentum_10_trading_days")[columns_to_show]
            .to_string(index=False)
        )

    print(
        "\nNote: extreme-move flags identify windows that need review. "
        "They do not automatically remove observations."
    )


if __name__ == "__main__":
    main()
