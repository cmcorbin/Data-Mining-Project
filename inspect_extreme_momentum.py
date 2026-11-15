
import pandas as pd

price_file = "data/raw/stock_prices_latest.csv"
event_file = "data/processed/final_earnings_event_dataset.csv"

# Load the events with the most extreme momentum values
events = pd.read_csv(event_file)

extreme_events = (
    events.dropna(subset=["momentum_10_trading_days"])
    .nlargest(15, "momentum_10_trading_days")
)

# Only load price histories for the symbols we need to inspect
symbols = set(extreme_events["symbol"].astype(str))

price_chunks = []

for chunk in pd.read_csv(
    price_file,
    usecols=[
        "symbol", "date", "close",
        "close_adjusted", "split_coefficient"
    ],
    chunksize=1_000_000
):
    matching = chunk[chunk["symbol"].astype(str).isin(symbols)]
    if not matching.empty:
        price_chunks.append(matching)

prices = pd.concat(price_chunks, ignore_index=True)
prices["date"] = pd.to_datetime(prices["date"])

prices = prices.sort_values(["symbol", "date"])
prices = prices.drop_duplicates(["symbol", "date"], keep="last")

for _, event in extreme_events.iterrows():
    symbol = str(event["symbol"])
    event_date = pd.to_datetime(event["earnings_date"])
    matched_date = pd.to_datetime(event["matched_price_date"])

    history = prices[
        (prices["symbol"].astype(str) == symbol)
        & (prices["date"] <= matched_date)
    ].tail(11)

    print("\n" + "=" * 75)
    print(
        f"{symbol} | earnings date: {event_date.date()} "
        f"| matched date: {matched_date.date()}"
    )
    print(
        f"Stored momentum: "
        f"{event['momentum_10_trading_days']:.2f}%"
    )

    if len(history) < 11:
        print("Insufficient price history for a 10-session comparison.")
        continue

    raw_start = history["close"].iloc[0]
    raw_end = history["close"].iloc[-1]
    adj_start = history["close_adjusted"].iloc[0]
    adj_end = history["close_adjusted"].iloc[-1]

    raw_return = (raw_end / raw_start - 1) * 100
    adj_return = (adj_end / adj_start - 1) * 100

    print(f"Raw-price 10-session return: {raw_return:.2f}%")
    print(f"Adjusted-price 10-session return: {adj_return:.2f}%")
    print("\nPrice history:")
    print(history.to_string(index=False))
