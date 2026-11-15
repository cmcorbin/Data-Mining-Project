import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/final_earnings_event_dataset.csv"
OUTPUT_DIRECTORY = "data/processed/checkpoint2_results"

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42

# Expected complete-case sample from Checkpoint 1.
EXPECTED_EVENTS = 78774
EXPECTED_COMPANIES = 2784


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def calculate_rate(beats, total):
    """
    Calculate an earnings beat rate.
    Return NaN if a group has no observations.
    """
    if total == 0:
        return np.nan

    return beats / total


def calculate_difference(rate_1, rate_2):
    """
    Calculate the difference between two beat rates.
    """
    if np.isnan(rate_1) or np.isnan(rate_2):
        return np.nan

    return rate_1 - rate_2


# ---------------------------------------------------------
# Load and prepare the dataset
# ---------------------------------------------------------

print("Loading earnings event dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Convert the variables used in the analysis to numeric values.
df["eps_beat"] = pd.to_numeric(df["eps_beat"], errors="coerce")
df["previous_quarter_beat"] = pd.to_numeric(
    df["previous_quarter_beat"], errors="coerce"
)
df["momentum_10_trading_days"] = pd.to_numeric(
    df["momentum_10_trading_days"], errors="coerce"
)

# Remove observations missing any of the required variables.
df = df.dropna(
    subset=[
        "symbol",
        "eps_beat",
        "previous_quarter_beat",
        "momentum_10_trading_days"
    ]
).copy()

# Standardize ticker symbols.
df["symbol"] = df["symbol"].astype(str).str.strip()

# Keep only valid binary outcome and previous earnings values.
df = df[
    df["eps_beat"].isin([0, 1])
    & df["previous_quarter_beat"].isin([0, 1])
].copy()

# Convert the binary variables to integers.
df["eps_beat"] = df["eps_beat"].astype(int)
df["previous_quarter_beat"] = df["previous_quarter_beat"].astype(int)

# Create the momentum category.
# Positive momentum means a return greater than zero.
df["momentum_group"] = np.where(
    df["momentum_10_trading_days"] > 0,
    "Positive",
    "Non-positive"
)

print("\nDataset preparation complete.")
print("Complete-case earnings events:", len(df))
print("Unique companies:", df["symbol"].nunique())

# Check whether the resulting sample matches Checkpoint 1.
if len(df) != EXPECTED_EVENTS:
    print(
        "\nWARNING: Event count differs from the Checkpoint 1 reference "
        f"of {EXPECTED_EVENTS:,}."
    )

if df["symbol"].nunique() != EXPECTED_COMPANIES:
    print(
        "\nWARNING: Company count differs from the Checkpoint 1 reference "
        f"of {EXPECTED_COMPANIES:,}."
    )


# ---------------------------------------------------------
# Define the four comparison groups
# ---------------------------------------------------------

# Each group is identified by:
# (previous earnings beat, positive momentum)

groups = [
    (0, "Non-positive"),
    (0, "Positive"),
    (1, "Non-positive"),
    (1, "Positive")
]

group_names = [
    "Previous did not beat, non-positive momentum",
    "Previous did not beat, positive momentum",
    "Previous beat, non-positive momentum",
    "Previous beat, positive momentum"
]

group_lookup = {
    group: index
    for index, group in enumerate(groups)
}

# Assign each event to one of the four groups.
df["group_index"] = [
    group_lookup[(previous, momentum)]
    for previous, momentum in zip(
        df["previous_quarter_beat"],
        df["momentum_group"]
    )
]


# ---------------------------------------------------------
# Aggregate event counts by company and group
# ---------------------------------------------------------

print("\nAggregating earnings events by company...")

company_symbols = sorted(df["symbol"].unique())

number_of_companies = len(company_symbols)
number_of_groups = len(groups)

# Each row represents one company.
# Each column represents one of the four comparison groups.
#
# The matrices store the number of events and number of beats
# for each company within each group.

company_index = {
    symbol: index
    for index, symbol in enumerate(company_symbols)
}

events_matrix = np.zeros(
    (number_of_companies, number_of_groups),
    dtype=int
)

beats_matrix = np.zeros(
    (number_of_companies, number_of_groups),
    dtype=int
)

# Convert company and group labels into array indices.
company_indices = df["symbol"].map(company_index).to_numpy()
group_indices = df["group_index"].to_numpy()
beat_values = df["eps_beat"].to_numpy()

# Count all events for each company and group.
np.add.at(
    events_matrix,
    (company_indices, group_indices),
    1
)

# Count earnings beats for each company and group.
np.add.at(
    beats_matrix,
    (company_indices, group_indices),
    beat_values
)

# Calculate the original group totals and beat rates.
observed_events = events_matrix.sum(axis=0)
observed_beats = beats_matrix.sum(axis=0)

observed_rates = np.array([
    calculate_rate(beats, total)
    for beats, total in zip(observed_beats, observed_events)
])

print("\nOriginal group beat rates:")

for index, name in enumerate(group_names):
    print(
        f"{name}: "
        f"{observed_rates[index] * 100:.2f}% "
        f"({observed_beats[index]:,}/{observed_events[index]:,})"
    )


# ---------------------------------------------------------
# Define the four comparisons
# ---------------------------------------------------------

# Each comparison calculates:
# first group's beat rate - second group's beat rate.

comparisons = [
    (
        "Momentum effect when previous earnings did not beat",
        1,
        0
    ),
    (
        "Momentum effect when previous earnings beat",
        3,
        2
    ),
    (
        "Previous earnings effect with positive momentum",
        3,
        1
    ),
    (
        "Previous earnings effect with non-positive momentum",
        2,
        0
    )
]

observed_differences = np.array([
    calculate_difference(
        observed_rates[first],
        observed_rates[second]
    )
    for _, first, second in comparisons
])


# ---------------------------------------------------------
# Company-level bootstrap
# ---------------------------------------------------------

print("\nStarting company-level bootstrap...")
print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)
print("Companies sampled per iteration:", number_of_companies)

rng = np.random.default_rng(RANDOM_SEED)

# Store the beat rates and differences from every iteration.
bootstrap_rates = np.full(
    (BOOTSTRAP_ITERATIONS, number_of_groups),
    np.nan
)

bootstrap_differences = np.full(
    (BOOTSTRAP_ITERATIONS, len(comparisons)),
    np.nan
)

for iteration in range(BOOTSTRAP_ITERATIONS):

    # Randomly select entire companies with replacement.
    #
    # A company may be selected more than once. If that happens,
    # all of its earnings events are represented multiple times.
    sampled_company_indices = rng.integers(
        0,
        number_of_companies,
        size=number_of_companies
    )

    # Sum all events and beats belonging to the sampled companies.
    # Repeated indices naturally count a company multiple times.
    sampled_events = events_matrix[sampled_company_indices].sum(axis=0)
    sampled_beats = beats_matrix[sampled_company_indices].sum(axis=0)

    # Calculate the four beat rates for this iteration.
    rates = np.array([
        calculate_rate(beats, total)
        for beats, total in zip(sampled_beats, sampled_events)
    ])

    bootstrap_rates[iteration] = rates

    # Calculate the four differences for this iteration.
    differences = np.array([
        calculate_difference(
            rates[first],
            rates[second]
        )
        for _, first, second in comparisons
    ])

    bootstrap_differences[iteration] = differences

    # Display progress every 500 iterations.
    if (iteration + 1) % 500 == 0:
        print(
            f"Completed {iteration + 1:,} "
            f"of {BOOTSTRAP_ITERATIONS:,} iterations."
        )


# ---------------------------------------------------------
# Calculate bootstrap confidence intervals
# ---------------------------------------------------------

print("\nCalculating bootstrap confidence intervals...")

results = []

# Summarize the four group beat rates.
for index, name in enumerate(group_names):

    valid_values = bootstrap_rates[:, index]
    valid_values = valid_values[~np.isnan(valid_values)]

    lower_bound, upper_bound = np.percentile(
        valid_values,
        [2.5, 97.5]
    )

    results.append({
        "analysis": "Group beat rate",
        "comparison": name,
        "observed_difference_or_rate": observed_rates[index],
        "bootstrap_mean": np.mean(valid_values),
        "ci_lower_95": lower_bound,
        "ci_upper_95": upper_bound,
        "bootstrap_probability_positive": np.nan,
        "valid_iterations": len(valid_values)
    })


# Summarize the four differences between groups.
for index, (name, first, second) in enumerate(comparisons):

    valid_values = bootstrap_differences[:, index]
    valid_values = valid_values[~np.isnan(valid_values)]

    lower_bound, upper_bound = np.percentile(
        valid_values,
        [2.5, 97.5]
    )

    probability_positive = np.mean(valid_values > 0)

    results.append({
        "analysis": "Difference in beat rates",
        "comparison": name,
        "observed_difference_or_rate": observed_differences[index],
        "bootstrap_mean": np.mean(valid_values),
        "ci_lower_95": lower_bound,
        "ci_upper_95": upper_bound,
        "bootstrap_probability_positive": probability_positive,
        "valid_iterations": len(valid_values)
    })


results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

summary_file = os.path.join(
    OUTPUT_DIRECTORY,
    "company_bootstrap_results.csv"
)

iterations_file = os.path.join(
    OUTPUT_DIRECTORY,
    "company_bootstrap_iterations.csv"
)

# Save the summary table.
results_df.to_csv(summary_file, index=False)

# Save the individual bootstrap results.
iterations_df = pd.DataFrame(
    bootstrap_rates,
    columns=[
        "rate_previous_no_beat_nonpositive",
        "rate_previous_no_beat_positive",
        "rate_previous_beat_nonpositive",
        "rate_previous_beat_positive"
    ]
)

for index, (name, _, _) in enumerate(comparisons):
    column_name = f"difference_{index + 1}"
    iterations_df[column_name] = bootstrap_differences[:, index]

iterations_df.to_csv(iterations_file, index=False)


# ---------------------------------------------------------
# Display final results
# ---------------------------------------------------------

print("\nCompany-level bootstrap results:")

for _, row in results_df.iterrows():

    if row["analysis"] == "Group beat rate":
        print(f"\n{row['comparison']}")
        print(
            f"Observed beat rate: "
            f"{row['observed_difference_or_rate'] * 100:.2f}%"
        )

    else:
        print(f"\n{row['comparison']}")
        print(
            f"Observed difference: "
            f"{row['observed_difference_or_rate'] * 100:.2f} "
            f"percentage points"
        )

        print(
            f"95% Bootstrap CI: "
            f"[{row['ci_lower_95'] * 100:.2f}, "
            f"{row['ci_upper_95'] * 100:.2f}] percentage points"
        )

        print(
            f"Proportion of bootstrap differences above zero: "
            f"{row['bootstrap_probability_positive'] * 100:.2f}%"
        )

print("\nResults saved to:")
print(summary_file)
print(iterations_file)

print("\nCompany-level bootstrap analysis complete.")
