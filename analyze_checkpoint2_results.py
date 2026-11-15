import pandas as pd
import numpy as np
import os


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/processed/final_earnings_event_dataset.csv"

OUTPUT_DIR = "data/processed/checkpoint2_results"

CONFIDENCE_LEVEL = 0.95


# --------------------------------------------------
# Wilson Confidence Interval
# --------------------------------------------------

def calculate_wilson_interval(beats, total, confidence=0.95):

    if total == 0:
        return np.nan, np.nan

    p = beats / total

    # Critical value for a two-sided confidence interval
    z = 1.959963984540054

    if confidence != 0.95:
        raise ValueError("This function currently supports 95% confidence intervals.")

    denominator = 1 + (z ** 2 / total)

    center = (
        p + (z ** 2 / (2 * total))
    ) / denominator

    margin = (
        z / denominator
    ) * np.sqrt(
        (p * (1 - p) / total) +
        (z ** 2 / (4 * total ** 2))
    )

    lower = center - margin
    upper = center + margin

    return lower, upper


# --------------------------------------------------
# Confidence Interval for Difference in Proportions
# --------------------------------------------------

def calculate_difference_interval(
    beats_1,
    total_1,
    beats_2,
    total_2
):

    p1 = beats_1 / total_1
    p2 = beats_2 / total_2

    # Calculate Wilson intervals for both groups
    lower_1, upper_1 = calculate_wilson_interval(
        beats_1,
        total_1
    )

    lower_2, upper_2 = calculate_wilson_interval(
        beats_2,
        total_2
    )

    # Difference between the two observed proportions
    difference = p1 - p2

    # Newcombe-Wilson confidence interval
    lower = difference - np.sqrt(
        (p1 - lower_1) ** 2 +
        (upper_2 - p2) ** 2
    )

    upper = difference + np.sqrt(
        (upper_1 - p1) ** 2 +
        (p2 - lower_2) ** 2
    )

    return difference, lower, upper



# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Prepare Analysis Variables
# --------------------------------------------------

# Keep only events with all three required variables
df = df.dropna(
    subset=[
        "eps_beat",
        "previous_quarter_beat",
        "momentum_10_trading_days"
    ]
).copy()

# Previous earnings beat or did not beat
df["previous_beat_group"] = np.where(
    df["previous_quarter_beat"] == 1,
    "Beat",
    "Did not beat"
)

# Positive or non-positive momentum
df["momentum_group"] = np.where(
    df["momentum_10_trading_days"] > 0,
    "Positive",
    "Non-positive"
)


# --------------------------------------------------
# Calculate Beat Rates and Confidence Intervals
# --------------------------------------------------

results = []

groups = df.groupby(
    ["previous_beat_group", "momentum_group"]
)

for (previous, momentum), group in groups:

    total = len(group)

    beats = group["eps_beat"].sum()

    beat_rate = beats / total

    lower, upper = calculate_wilson_interval(
        beats,
        total,
        CONFIDENCE_LEVEL
    )

    results.append({
        "previous_earnings": previous,
        "momentum": momentum,
        "total_events": total,
        "eps_beats": int(beats),
        "beat_rate": beat_rate,
        "confidence_lower": lower,
        "confidence_upper": upper
    })


# --------------------------------------------------
# Export Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

# --------------------------------------------------
# Compare Beat Rates Between Groups
# --------------------------------------------------

comparisons = [

    {
        "comparison": "Momentum effect when previous report did not beat",
        "group_1": {
            "previous_beat_group": "Did not beat",
            "momentum_group": "Positive"
        },
        "group_2": {
            "previous_beat_group": "Did not beat",
            "momentum_group": "Non-positive"
        }
    },

    {
        "comparison": "Momentum effect when previous report beat",
        "group_1": {
            "previous_beat_group": "Beat",
            "momentum_group": "Positive"
        },
        "group_2": {
            "previous_beat_group": "Beat",
            "momentum_group": "Non-positive"
        }
    },

    {
        "comparison": "Previous earnings effect with positive momentum",
        "group_1": {
            "previous_beat_group": "Beat",
            "momentum_group": "Positive"
        },
        "group_2": {
            "previous_beat_group": "Did not beat",
            "momentum_group": "Positive"
        }
    },

    {
        "comparison": "Previous earnings effect with non-positive momentum",
        "group_1": {
            "previous_beat_group": "Beat",
            "momentum_group": "Non-positive"
        },
        "group_2": {
            "previous_beat_group": "Did not beat",
            "momentum_group": "Non-positive"
        }
    }
]

difference_results = []

for comparison in comparisons:

    # Select the first group
    group_1 = df[
        (df["previous_beat_group"] ==
         comparison["group_1"]["previous_beat_group"]) &
        (df["momentum_group"] ==
         comparison["group_1"]["momentum_group"])
    ]

    # Select the second group
    group_2 = df[
        (df["previous_beat_group"] ==
         comparison["group_2"]["previous_beat_group"]) &
        (df["momentum_group"] ==
         comparison["group_2"]["momentum_group"])
    ]

    total_1 = len(group_1)
    total_2 = len(group_2)

    beats_1 = group_1["eps_beat"].sum()
    beats_2 = group_2["eps_beat"].sum()

    rate_1 = beats_1 / total_1
    rate_2 = beats_2 / total_2

    difference, lower, upper = calculate_difference_interval(
        beats_1,
        total_1,
        beats_2,
        total_2
    )

    difference_results.append({
        "comparison": comparison["comparison"],
        "group_1_events": total_1,
        "group_2_events": total_2,
        "group_1_beat_rate": rate_1,
        "group_2_beat_rate": rate_2,
        "difference": difference,
        "confidence_lower": lower,
        "confidence_upper": upper
    })


# --------------------------------------------------
# Export Difference Results
# --------------------------------------------------

difference_df = pd.DataFrame(difference_results)

difference_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "beat_rate_difference_confidence_intervals.csv"
    ),
    index=False
)

print("\nBeat Rate Difference Comparisons")
print("--------------------------------")

print(
    difference_df.to_string(
        index=False,
        formatters={
            "group_1_beat_rate": "{:.2%}".format,
            "group_2_beat_rate": "{:.2%}".format,
            "difference": "{:.2%}".format,
            "confidence_lower": "{:.2%}".format,
            "confidence_upper": "{:.2%}".format
        }
    )
)


results_df.to_csv(
    os.path.join(OUTPUT_DIR, "beat_rate_confidence_intervals.csv"),
    index=False
)

print("\nFour-Group Beat Rate Comparison")
print("--------------------------------")

print(
    results_df.to_string(
        index=False,
        formatters={
            "beat_rate": "{:.2%}".format,
            "confidence_lower": "{:.2%}".format,
            "confidence_upper": "{:.2%}".format
        }
    )
)

print("\nResults saved successfully.")
