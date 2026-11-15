import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/final_earnings_event_dataset.csv"

OUTPUT_DIRECTORY = "data/processed/checkpoint2_results"

EXTREME_FLAG_COLUMN = "momentum_window_extreme_daily_move"

# Original complete-case group results from Checkpoint 1.
ORIGINAL_RESULTS = {
    (0, "Non-positive"): {"events": 14781, "beats": 6352},
    (0, "Positive"): {"events": 16947, "beats": 8262},
    (1, "Non-positive"): {"events": 20903, "beats": 14103},
    (1, "Positive"): {"events": 26143, "beats": 18727}
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def calculate_rate(beats, total):
    """Calculate the percentage of earnings events that beat."""
    if total == 0:
        return np.nan

    return (beats / total) * 100


def parse_extreme_flag(value):
    """
    Convert the extreme-momentum flag into True, False, or missing.

    Unknown values remain missing instead of being assumed
    to represent non-extreme observations.
    """
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    value = str(value).strip().lower()

    if value in ["true", "1", "yes", "y", "t"]:
        return True

    if value in ["false", "0", "no", "n", "f"]:
        return False

    return pd.NA


# ---------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------

print("Loading earnings event dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\nDataset loaded successfully.")
print("Total earnings events:", len(df))


# ---------------------------------------------------------
# Check required columns
# ---------------------------------------------------------

required_columns = [
    "symbol",
    "eps_beat",
    "previous_quarter_beat",
    "momentum_10_trading_days",
    EXTREME_FLAG_COLUMN
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found."
        )


# ---------------------------------------------------------
# Prepare the analysis variables
# ---------------------------------------------------------

print("\nPreparing analysis variables...")

df["eps_beat"] = pd.to_numeric(
    df["eps_beat"], errors="coerce"
)

df["previous_quarter_beat"] = pd.to_numeric(
    df["previous_quarter_beat"], errors="coerce"
)

df["momentum_10_trading_days"] = pd.to_numeric(
    df["momentum_10_trading_days"], errors="coerce"
)

# Convert the extreme-momentum flag.
df["extreme_flag"] = df[EXTREME_FLAG_COLUMN].apply(
    parse_extreme_flag
).astype("boolean")

# Count the original flag values.
extreme_count = (df["extreme_flag"] == True).sum()
non_extreme_count = (df["extreme_flag"] == False).sum()
unknown_count = df["extreme_flag"].isna().sum()

print("\nExtreme-momentum flag summary:")
print("Flagged extreme events:", extreme_count)
print("Not flagged:", non_extreme_count)
print("Unknown flag values:", unknown_count)


# ---------------------------------------------------------
# Create the original complete-case sample
# ---------------------------------------------------------

complete_df = df.dropna(
    subset=[
        "eps_beat",
        "previous_quarter_beat",
        "momentum_10_trading_days"
    ]
).copy()

complete_df = complete_df[
    complete_df["eps_beat"].isin([0, 1])
    & complete_df["previous_quarter_beat"].isin([0, 1])
].copy()

complete_df["eps_beat"] = complete_df["eps_beat"].astype(int)
complete_df["previous_quarter_beat"] = (
    complete_df["previous_quarter_beat"].astype(int)
)

complete_df["momentum_group"] = np.where(
    complete_df["momentum_10_trading_days"] > 0,
    "Positive",
    "Non-positive"
)

print("\nOriginal complete-case sample:", len(complete_df))


# ---------------------------------------------------------
# Create the filtered sample
# ---------------------------------------------------------

# Retain only events explicitly identified as not extreme.
# Events with unknown flag values are excluded from this
# sensitivity analysis rather than assumed to be non-extreme.

filtered_df = complete_df[
    complete_df["extreme_flag"] == False
].copy()

excluded_extreme = (
    complete_df["extreme_flag"] == True
).sum()

excluded_unknown = (
    complete_df["extreme_flag"].isna()
).sum()

print("\nSensitivity analysis sample:")
print("Original complete-case events:", len(complete_df))
print("Excluded flagged extreme events:", excluded_extreme)
print("Excluded events with unknown flags:", excluded_unknown)
print("Remaining events:", len(filtered_df))


# ---------------------------------------------------------
# Calculate four-group results
# ---------------------------------------------------------

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


def calculate_group_results(data, sample_name):
    """Calculate sample sizes and beat rates for all four groups."""

    results = []

    for index, (previous, momentum) in enumerate(groups):

        group = data[
            (data["previous_quarter_beat"] == previous)
            & (data["momentum_group"] == momentum)
        ]

        total = len(group)
        beats = group["eps_beat"].sum()

        rate = calculate_rate(beats, total)

        results.append({
            "sample": sample_name,
            "group": group_names[index],
            "previous_quarter_beat": previous,
            "momentum_group": momentum,
            "events": total,
            "beats": beats,
            "beat_rate_percent": rate
        })

    return pd.DataFrame(results)


original_results = calculate_group_results(
    complete_df,
    "Original complete-case sample"
)

filtered_results = calculate_group_results(
    filtered_df,
    "Excluding extreme-momentum events"
)


# ---------------------------------------------------------
# Compare original and filtered results
# ---------------------------------------------------------

comparison = original_results[
    ["group", "events", "beats", "beat_rate_percent"]
].copy()

comparison = comparison.rename(columns={
    "events": "original_events",
    "beats": "original_beats",
    "beat_rate_percent": "original_beat_rate"
})

filtered_comparison = filtered_results[
    ["group", "events", "beats", "beat_rate_percent"]
].rename(columns={
    "events": "filtered_events",
    "beats": "filtered_beats",
    "beat_rate_percent": "filtered_beat_rate"
})

comparison = comparison.merge(
    filtered_comparison,
    on="group"
)

comparison["change_in_events"] = (
    comparison["filtered_events"]
    - comparison["original_events"]
)

comparison["change_in_beat_rate_pp"] = (
    comparison["filtered_beat_rate"]
    - comparison["original_beat_rate"]
)


# ---------------------------------------------------------
# Calculate the four group differences
# ---------------------------------------------------------

def calculate_group_differences(results, sample_name):
    """Calculate the same four comparisons used previously."""

    rates = results.set_index("group")["beat_rate_percent"]

    differences = [
        (
            "Momentum effect when previous earnings did not beat",
            rates[group_names[1]] - rates[group_names[0]]
        ),
        (
            "Momentum effect when previous earnings beat",
            rates[group_names[3]] - rates[group_names[2]]
        ),
        (
            "Previous earnings effect with positive momentum",
            rates[group_names[3]] - rates[group_names[1]]
        ),
        (
            "Previous earnings effect with non-positive momentum",
            rates[group_names[2]] - rates[group_names[0]]
        )
    ]

    return pd.DataFrame([
        {
            "sample": sample_name,
            "comparison": name,
            "difference_percentage_points": difference
        }
        for name, difference in differences
    ])


original_differences = calculate_group_differences(
    original_results,
    "Original complete-case sample"
)

filtered_differences = calculate_group_differences(
    filtered_results,
    "Excluding extreme-momentum events"
)

difference_comparison = original_differences[
    ["comparison", "difference_percentage_points"]
].rename(columns={
    "difference_percentage_points": "original_difference"
})

difference_comparison = difference_comparison.merge(
    filtered_differences[
        ["comparison", "difference_percentage_points"]
    ].rename(columns={
        "difference_percentage_points": "filtered_difference"
    }),
    on="comparison"
)

difference_comparison["change_in_difference_pp"] = (
    difference_comparison["filtered_difference"]
    - difference_comparison["original_difference"]
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n========================================")
print("ORIGINAL FOUR-GROUP RESULTS")
print("========================================")

print(
    original_results[
        ["group", "events", "beats", "beat_rate_percent"]
    ].to_string(
        index=False,
        formatters={
            "beat_rate_percent": "{:.2f}%".format
        }
    )
)


print("\n========================================")
print("FILTERED FOUR-GROUP RESULTS")
print("========================================")

print(
    filtered_results[
        ["group", "events", "beats", "beat_rate_percent"]
    ].to_string(
        index=False,
        formatters={
            "beat_rate_percent": "{:.2f}%".format
        }
    )
)


print("\n========================================")
print("BEAT RATE COMPARISON")
print("========================================")

print(
    comparison.to_string(
        index=False,
        formatters={
            "original_beat_rate": "{:.2f}%".format,
            "filtered_beat_rate": "{:.2f}%".format,
            "change_in_beat_rate_pp": "{:+.2f}".format
        }
    )
)


print("\n========================================")
print("COMPARISON OF GROUP DIFFERENCES")
print("========================================")

print(
    difference_comparison.to_string(
        index=False,
        formatters={
            "original_difference": "{:.2f}".format,
            "filtered_difference": "{:.2f}".format,
            "change_in_difference_pp": "{:+.2f}".format
        }
    )
)


# ---------------------------------------------------------
# Validate original results
# ---------------------------------------------------------

print("\n========================================")
print("VALIDATION")
print("========================================")

validation_passed = True

for _, row in original_results.iterrows():

    original_group = row["group"]

    expected_index = group_names.index(original_group)
    previous, momentum = groups[expected_index]

    expected = ORIGINAL_RESULTS[(previous, momentum)]

    if (
        row["events"] != expected["events"]
        or row["beats"] != expected["beats"]
    ):
        validation_passed = False

        print(
            f"WARNING: Original results do not match "
            f"the Checkpoint 1 reference for {original_group}."
        )

if validation_passed:
    print("Original group counts match Checkpoint 1.")

print("Filtered sample size:", len(filtered_df))
print("Sensitivity analysis complete.")


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

group_file = os.path.join(
    OUTPUT_DIRECTORY,
    "extreme_momentum_group_results.csv"
)

comparison_file = os.path.join(
    OUTPUT_DIRECTORY,
    "extreme_momentum_rate_comparison.csv"
)

difference_file = os.path.join(
    OUTPUT_DIRECTORY,
    "extreme_momentum_difference_comparison.csv"
)

original_results.to_csv(
    group_file.replace(".csv", "_original.csv"),
    index=False
)

filtered_results.to_csv(
    group_file,
    index=False
)

comparison.to_csv(comparison_file, index=False)

difference_comparison.to_csv(
    difference_file,
    index=False
)

print("\nResults saved to:")
print(group_file.replace(".csv", "_original.csv"))
print(group_file)
print(comparison_file)
print(difference_file)
