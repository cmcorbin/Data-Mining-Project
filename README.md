# Earnings Surprise and Stock Momentum Analysis

## Project Overview

This project investigates whether a company's previous earnings performance and recent stock-price momentum can help predict whether its next earnings announcement will beat analyst EPS estimates.

Each row in the final dataset represents an individual earnings announcement event. The dataset combines historical earnings information with adjusted stock prices to create features for later statistical analysis and machine learning.

## Research Question

Do a company's previous earnings beat/miss result and its 10-trading-day stock-price momentum help predict whether its next earnings announcement will beat analyst EPS estimates?

## Project Structure

```text
GroupProject/
├── data/                         # Download separately (see Data Access)
│   ├── raw/
│   └── processed/
├── prepare_earnings_and_prices.py
├── calculate_earnings_beat_and_momentum.py
└── README.md
```

The data files are not included in this repository and must be obtained separately.

## Data Source and Citation

The project uses earnings and historical stock-price data from the Kaggle dataset [US Historical Stock Prices with Earnings Data](https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data), uploaded by `tsaustin`.

The dataset includes historical stock prices, earnings announcement dates, estimated EPS, reported EPS, and a summary of available data by stock symbol. Its original description identifies Nasdaq, Yahoo Finance, Zacks, and Alpha Vantage as data sources.

**Dataset citation:**

> tsaustin. (n.d.). *US historical stock prices with earnings data* [Data set]. Kaggle. https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data

**License and redistribution:** The Kaggle dataset page lists its license as unknown. Users should review the current dataset terms and applicable source restrictions before redistributing the original files. This project does not assume that publicly accessible data is automatically available for unrestricted redistribution.

## Data Access and Setup

Download the original dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data).

Place the required CSV files in the project directory using the following structure:

```text
data/
├── raw/
│   ├── earnings_latest.csv
│   ├── stock_prices_latest.csv
│   └── dataset_summary.csv
└── processed/
```

The `data/raw/` directory must contain the original input CSV files. The scripts generate the processed datasets in `data/processed/`.

### Requirements

- Python 3
- The Python packages required by the scripts

Use the project's existing virtual environment and install any required packages before running the pipeline.

## Data Processing Pipeline

### 1. Prepare Earnings and Stock-Price Data

**Script:** `prepare_earnings_and_prices.py`

This script loads the raw earnings and stock-price CSV files, cleans the records, identifies eligible stock symbols, and matches earnings announcements to relevant stock-price observations.

#### Earnings Data Cleaning

- Standardizes stock symbols by trimming whitespace and converting them to uppercase.
- Converts earnings dates to datetime values.
- Converts estimated and actual EPS values to numeric values.
- Normalizes release-time values to `pre` or `post`; other values are treated as missing.
- Removes records without a valid stock symbol or earnings date.
- Removes duplicate earnings events for the same symbol and date.

Missing actual or estimated EPS values do not automatically disqualify an earnings event at this stage.

#### Stock-Price Data Cleaning

- Standardizes stock symbols.
- Converts price dates and adjusted closing prices to appropriate data types.
- Removes records with missing symbols, dates, or adjusted closing prices.
- Removes nonpositive adjusted closing prices.
- Removes duplicate records for the same symbol and date.

#### Stock Eligibility

A stock symbol is eligible when:

1. It has at least 30 unique earnings dates in the cleaned earnings data.
2. It passes the additional 30-earnings threshold in `dataset_summary.csv` if that file exists and contains the required columns.
3. It has at least one valid stock-price record.

#### Earnings-to-Price Matching

Each earnings event is matched to an appropriate adjusted closing price:

- For after-market announcements (`post`), the same day's closing price is allowed.
- For before-market announcements (`pre`) or unknown release times, the script uses a strictly earlier trading date.
- The script calculates the calendar-day gap between the earnings date and matched price date.
- Matches within seven calendar days are flagged as recent.

Events without a price match or with an older match are retained in the output rather than automatically deleted.

#### Outputs

| File | Description |
|---|---|
| `cleaned_earnings.csv` | Cleaned earnings records |
| `cleaned_prices.csv` | Cleaned adjusted-price records |
| `eligible_earnings.csv` | Earnings records for eligible symbols |
| `eligible_prices.csv` | Price records for eligible symbols |
| `earnings_with_price_match.csv` | Earnings events with matched-price information |

### 2. Engineer Earnings and Momentum Features

**Script:** `calculate_earnings_beat_and_momentum.py`

This script uses the prepared earnings events and adjusted stock prices to calculate the features needed for the research question.

#### Earnings Features

- `eps_surprise`: Actual EPS minus estimated EPS.
- `eps_surprise_pct`: EPS surprise expressed relative to estimated EPS, subject to the calculation's handling of missing or zero estimates.
- `eps_beat`: Binary target indicating whether actual EPS exceeds estimated EPS. A value of `1` represents a beat; `0` represents a miss or tie. Missing actual or estimated EPS produces a missing label.
- `previous_quarter_beat`: The beat/miss label from the previous available earnings event for the same company.

**Note:** The current implementation uses the previous available earnings event, which may not always correspond to the immediately preceding fiscal quarter.

#### Stock Momentum

`momentum_10_trading_days` measures the adjusted stock-price return over a 10-trading-day interval ending on or before the matched price date.

The calculation uses 11 adjusted closing-price observations: one starting price and one ending price separated by 10 trading-day intervals.

\[
\text{Momentum}=\frac{P_{\text{end}}}{P_{\text{start}}}-1
\]

Adjusted closing prices are used directly; split adjustments are not applied a second time.

The dataset also records the start and end dates and prices used in the calculation.

#### Momentum Quality Checks

The script identifies potentially suspicious momentum windows:

- `momentum_window_extreme_daily_move`: Flags windows containing at least one absolute daily adjusted-price return greater than 50%.
- `momentum_window_valid`: Indicates whether sufficient price observations were available to calculate momentum.

These flags identify observations for review. They do not prove that an observation is erroneous, and the feature-engineering script retains flagged observations in the main dataset.

#### Main Output

`data/processed/final_earnings_event_dataset.csv`

This is the primary Checkpoint 1 output. It contains the earnings information, matched-price information, engineered features, and quality flags.

The dataset is retained before any optional modeling filters are applied.

## Checkpoint 1 Results

The latest reported feature-engineering run produced the following results:

| Metric | Count |
|---|---:|
| Total earnings events | 133,755 |
| Events with valid 10-trading-day momentum | 133,483 |
| Events with usable actual and estimated EPS | 83,166 |
| Events with a previous-event beat/miss label | 80,498 |
| Momentum windows flagged for extreme daily moves | 216 |

These counts describe the reported runs and should be rechecked if the raw data or scripts change.

## Running the Pipeline

Run the commands from the project root directory with the Python virtual environment activated.

```bash
python prepare_earnings_and_prices.py
python calculate_earnings_beat_and_momentum.py
```

The second script should run after the first because it depends on the processed earnings and price data.

## Important Methodological Considerations

- Use adjusted closing prices consistently for stock-return calculations.
- Preserve the main feature-engineered dataset before applying modeling-specific filters.
- Treat extreme-momentum flags as screening indicators rather than proof of erroneous data.
- Verify that previous-event earnings labels align with the intended previous-quarter definition.
- Document exclusions and compare modeling results with and without flagged observations.
- Avoid data leakage when splitting observations into training and test sets, especially because each company can contribute multiple earnings events.

## Acknowledgments

The original earnings and stock-price data are attributed to the Kaggle dataset cited above. The cleaning, matching, feature engineering, and subsequent analysis are performed as part of this project.