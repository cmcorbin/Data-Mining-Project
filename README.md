# Earnings Surprise and Stock Momentum Analysis

## Project Overview

This project investigates whether a company's previous earnings performance and recent stock-price momentum can help predict whether its next earnings announcement will beat analyst EPS estimates.

Each row in the final dataset represents an individual earnings announcement event. The dataset combines historical earnings information with adjusted stock prices to create features for statistical analysis and machine learning.

## Research Question

Do a company's previous earnings beat/miss result and its 10-trading-day stock-price momentum help predict whether its next earnings announcement will beat analyst EPS estimates?

## Project Structure

```text
GroupProject/
├── data/                              # Download separately (see Data Access)
│   ├── raw/
│   └── processed/
│       ├── final_earnings_event_dataset.csv
│       └── checkpoint2_results/
├── prepare_earnings_and_prices.py
├── calculate_earnings_beat_and_momentum.py
├── analyze_checkpoint2_results.py
├── analyze_company_bootstrap.py
├── analyze_missing_data.py
├── analyze_extreme_momentum.py
└── README.md
```

The data files are not included in this repository and must be obtained separately. Aggregate analysis results are saved in `data/processed/checkpoint2_results/`.

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

The reported feature-engineering run produced the following results:

| Metric | Count |
|---|---:|
| Total earnings events | 133,755 |
| Events with valid 10-trading-day momentum | 133,483 |
| Events with usable actual and estimated EPS | 83,166 |
| Events with a previous-event beat/miss label | 80,498 |
| Momentum windows flagged for extreme daily moves | 216 |

These counts describe the reported runs and should be rechecked if the raw data or scripts change.

## Checkpoint 2 Analysis

Checkpoint 2 extends the initial data preparation and feature-engineering work by comparing earnings beat rates across the four combinations of previous earnings performance and 10-trading-day momentum.

The primary analysis uses earnings events with complete values for the current EPS beat indicator, previous earnings beat indicator, and momentum. Each observation represents an earnings announcement event.

### 1. Four-Group Earnings Beat Comparison

**Script:** `analyze_checkpoint2_results.py`

This script analyzes the complete-case dataset and compares current earnings beat rates across four groups:

1. Previous earnings did not beat, non-positive momentum.
2. Previous earnings did not beat, positive momentum.
3. Previous earnings beat, non-positive momentum.
4. Previous earnings beat, positive momentum.

Momentum is classified as positive when the 10-trading-day return is greater than zero. Values at or below zero are classified as non-positive.

The earnings beat rate is calculated as:

\[
\text{Beat Rate}=\frac{\text{Number of EPS Beats}}{\text{Total Earnings Events}}\times100
\]

#### Preliminary Group Results

The complete-case analysis contains 78,774 earnings events.

| Previous earnings result | Momentum | Events | EPS Beats | Beat Rate |
|---|---|---:|---:|---:|
| Did not beat | Non-positive | 14,781 | 6,352 | 42.97% |
| Did not beat | Positive | 16,947 | 8,262 | 48.75% |
| Beat | Non-positive | 20,903 | 14,103 | 67.47% |
| Beat | Positive | 26,143 | 18,727 | 71.63% |

The highest observed beat rate occurs when the previous earnings event beat and momentum is positive (71.63%). The lowest occurs when the previous earnings event did not beat and momentum is non-positive (42.97%).

Within both previous earnings groups, positive momentum is associated with a higher observed beat rate.

### 2. Confidence Interval Analysis

The analysis calculates Wilson 95% confidence intervals for each group's earnings beat rate.

Newcombe-Wilson confidence intervals are also calculated for differences between group beat rates. These intervals provide an estimate of uncertainty around the observed percentage-point differences.

The four comparisons are:

- Positive versus non-positive momentum when the previous earnings event did not beat.
- Positive versus non-positive momentum when the previous earnings event beat.
- Previous earnings beat versus non-beat among positive-momentum events.
- Previous earnings beat versus non-beat among non-positive-momentum events.

These comparisons help evaluate whether the observed differences are distinguishable from zero under the interval method's assumptions.

### 3. Company-Level Bootstrap Analysis

**Script:** `analyze_company_bootstrap.py`

Because individual companies can contribute multiple earnings announcements, treating every event as fully independent may understate uncertainty.

To address repeated observations from the same company, a company-level bootstrap procedure is performed.

The procedure:

1. Samples entire companies with replacement.
2. Retains all eligible complete-case earnings events belonging to each sampled company.
3. Recalculates the four group beat rates and their differences.
4. Repeats the process 5,000 times.
5. Uses the 2.5th and 97.5th percentiles of the bootstrap distributions to calculate 95% confidence intervals.

A fixed random seed is used to make the analysis reproducible. Companies selected more than once contribute their earnings events multiple times within that bootstrap iteration.

The complete-case sample contains 78,774 events from 2,784 companies.

#### Company-Level Bootstrap Results

| Comparison | Observed Difference | 95% Bootstrap Confidence Interval |
|---|---:|---:|
| Momentum effect when previous earnings did not beat | +5.78 pp | [4.66, 6.88] pp |
| Momentum effect when previous earnings beat | +4.16 pp | [3.33, 4.98] pp |
| Previous earnings effect with positive momentum | +22.88 pp | [21.67, 24.10] pp |
| Previous earnings effect with non-positive momentum | +24.49 pp | [23.21, 25.76] pp |

All four bootstrap confidence intervals are above zero. All 5,000 iterations produced positive differences for each comparison.

This provides additional evidence that the observed group differences remain positive when repeated observations from the same company are accounted for through company-level resampling.

The bootstrap does not establish causation or eliminate possible selection bias from missing observations.

### 4. Missing-Data Analysis

**Script:** `analyze_missing_data.py`

The missing-data analysis examines the availability of the three variables required for the four-group comparison across all 133,755 earnings events.

It calculates:

- Available and missing observations for each required variable.
- The number of events missing exactly one, two, or all three variables.
- The specific combinations of variables that are missing.
- The overall number and percentage of complete and incomplete observations.

#### Missing Data by Variable

| Variable | Available Events | Missing Events |
|---|---:|---:|
| Current EPS beat | 83,166 (62.18%) | 50,589 (37.82%) |
| Previous earnings beat | 80,498 (60.18%) | 53,257 (39.82%) |
| 10-day momentum | 133,483 (99.80%) | 272 (0.20%) |

#### Missing Data by Number of Variables

| Category | Events | Percentage |
|---|---:|---:|
| No missing variables | 78,774 | 58.89% |
| Missing exactly one variable | 5,938 | 4.44% |
| Missing exactly two variables | 48,949 | 36.60% |
| Missing all three variables | 94 | 0.07% |
| Total incomplete observations | 54,981 | 41.11% |

The largest missing-value combination consists of events missing both the current EPS beat and previous earnings beat indicators. This combination accounts for 48,919 events (36.57% of the full dataset).

The results indicate that missing earnings information is the primary reason observations are excluded from the four-group comparison. Historical momentum information is available for nearly all events.

Because the primary analysis uses complete observations only, the results may not represent all earnings events in the original dataset. Potential selection bias should be considered when interpreting the observed relationships.

### 5. Extreme-Momentum Sensitivity Analysis

**Script:** `analyze_extreme_momentum.py`

An additional sensitivity analysis examines whether the four-group results are substantially affected by observations flagged for extreme daily price movements within the ten-trading-day momentum windows.

The original complete-case sample retains all flagged observations. For the sensitivity analysis, complete-case events explicitly flagged for extreme daily price movements are excluded, and the four-group beat rates are recalculated.

The original complete-case sample contains 78,774 events. Excluding the 107 complete-case events flagged for extreme daily price movements produces a filtered sample of 78,667 events.

#### Original and Filtered Beat Rates

| Previous earnings result | Momentum | Original Beat Rate | Filtered Beat Rate |
|---|---|---:|---:|
| Did not beat | Non-positive | 42.97% | 42.97% |
| Did not beat | Positive | 48.75% | 48.77% |
| Beat | Non-positive | 67.47% | 67.49% |
| Beat | Positive | 71.63% | 71.67% |

#### Comparison of Group Differences

| Comparison | Original Difference | Filtered Difference |
|---|---:|---:|
| Momentum effect when previous earnings did not beat | +5.78 pp | +5.80 pp |
| Momentum effect when previous earnings beat | +4.16 pp | +4.17 pp |
| Previous earnings effect with positive momentum | +22.88 pp | +22.90 pp |
| Previous earnings effect with non-positive momentum | +24.49 pp | +24.52 pp |

The results remain nearly identical after excluding extreme-momentum observations. The small changes suggest that these flagged observations do not materially drive the observed relationships.

The original complete-case sample is retained for the primary analysis, while the filtered results serve as an additional robustness check.

### 6. Interpretation of Checkpoint 2 Results

The four-group comparison shows that previous earnings performance and positive pre-announcement momentum are both associated with higher observed EPS beat rates.

The group with a previous earnings beat and positive momentum has the highest observed beat rate (71.63%), while the group with a previous earnings non-beat and non-positive momentum has the lowest (42.97%).

The confidence interval and company-level bootstrap analyses show that all four observed differences remain positive under the respective uncertainty calculations. The sensitivity analysis also indicates that excluding extreme-momentum observations has very little effect on the results.

These findings are consistent with the original hypothesis. However, the analysis is observational and does not establish that previous earnings performance or stock momentum causes a subsequent earnings beat. Missing earnings information, repeated observations, and the use of the previous available earnings event as a proxy for the previous fiscal quarter remain important methodological considerations.

## Analysis Output Files

The Checkpoint 2 scripts save aggregate results in `data/processed/checkpoint2_results/`.

| File | Description |
|---|---|
| `beat_rate_confidence_intervals.csv` | Wilson confidence intervals for the four group beat rates |
| `beat_rate_difference_confidence_intervals.csv` | Newcombe-Wilson confidence intervals for group differences |
| `company_bootstrap_results.csv` | Summary of company-level bootstrap results |
| `company_bootstrap_iterations.csv` | Results from individual bootstrap iterations |
| `missing_data_by_variable.csv` | Available and missing counts for each variable |
| `missing_data_by_category.csv` | Counts by number of missing variables |
| `missing_data_combinations.csv` | Specific combinations of missing variables |
| `missing_data_overall_summary.csv` | Overall complete and incomplete observation counts |
| `extreme_momentum_group_results_original.csv` | Original complete-case group results |
| `extreme_momentum_group_results.csv` | Group results after excluding flagged events |
| `extreme_momentum_rate_comparison.csv` | Original versus filtered beat-rate comparison |
| `extreme_momentum_difference_comparison.csv` | Original versus filtered group-difference comparison |

## Running the Pipeline

Run the commands from the project root directory with the Python virtual environment activated.

**Data preparation and feature engineering:**

```bash
python prepare_earnings_and_prices.py
python calculate_earnings_beat_and_momentum.py
```

The second script should run after the first because it depends on the processed earnings and price data.

**Checkpoint 2 analysis:**

```bash
python analyze_checkpoint2_results.py
python analyze_company_bootstrap.py
python analyze_missing_data.py
python analyze_extreme_momentum.py
```

The Checkpoint 2 analysis scripts depend on `data/processed/final_earnings_event_dataset.csv`, which must be generated by the data-processing pipeline.

The aggregate output files are saved in `data/processed/checkpoint2_results/`.

## Important Methodological Considerations

- Use adjusted closing prices consistently for stock-return calculations.
- Preserve the main feature-engineered dataset before applying modeling-specific filters.
- Treat extreme-momentum flags as screening indicators rather than proof of erroneous data.
- Verify that previous-event earnings labels align with the intended previous-quarter definition.
- Document exclusions and compare modeling results with and without flagged observations.
- Avoid data leakage when splitting observations into training and test sets, especially because each company can contribute multiple earnings events.
- Interpret the confidence intervals and bootstrap results as measures of uncertainty around observed associations, not proof of causation.
- Recognize that the complete-case analysis excludes events with missing earnings information, which may introduce selection bias.

## Acknowledgments

The original earnings and stock-price data are attributed to the Kaggle dataset cited above. The cleaning, matching, feature engineering, statistical comparisons, and subsequent analysis are performed as part of this project.