import os
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/final_earnings_event_dataset.csv"

OUTPUT_DIRECTORY = "data/processed/checkpoint2_results"

TOTAL_EXPECTED_EVENTS = 133755

# Variables required for the four-group analysis.
REQUIRED_COLUMNS = [
    "eps_beat",
    "previous_quarter_beat",
    "momentum_10_trading_days"
]


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

for column in REQUIRED_COLUMNS:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )


# ---------------------------------------------------------
# Identify missing values
# ---------------------------------------------------------

print("\nAnalyzing missing values...")

# Convert analysis variables to numeric values.
# Values that cannot be interpreted numerically become missing.
for column in REQUIRED_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Create a Boolean missing-value indicator for each variable.
missing_data = df[REQUIRED_COLUMNS].isna()

# Count available and missing observations for each variable.
variable_results = []

for column in REQUIRED_COLUMNS:

    missing_count = missing_data[column].sum()
    available_count = len(df) - missing_count

    missing_percentage = (missing_count / len(df)) * 100
    available_percentage = (available_count / len(df)) * 100

    variable_results.append({
        "variable": column,
        "available_count": available_count,
        "available_percentage": available_percentage,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

variable_results_df = pd.DataFrame(variable_results)


# ---------------------------------------------------------
# Analyze overlapping missing values
# ---------------------------------------------------------

# Count how many of the three required variables are missing
# for each earnings event.

df["missing_variable_count"] = missing_data.sum(axis=1)

# Categorize each event based on the number of missing variables.
df["missing_category"] = df["missing_variable_count"].map({
    0: "Complete - no missing variables",
    1: "Missing exactly one variable",
    2: "Missing exactly two variables",
    3: "Missing all three variables"
})

missing_category_order = [
    "Complete - no missing variables",
    "Missing exactly one variable",
    "Missing exactly two variables",
    "Missing all three variables"
]

# Count observations in each category.
category_results = (
    df["missing_category"]
    .value_counts()
    .reindex(missing_category_order, fill_value=0)
    .reset_index()
)

category_results.columns = [
    "missing_category",
    "event_count"
]

category_results["percentage"] = (
    category_results["event_count"] / len(df)
) * 100


# ---------------------------------------------------------
# Analyze exact missing-value combinations
# ---------------------------------------------------------

# This identifies which specific variables are missing together.

df["missing_combination"] = missing_data.apply(
    lambda row: ", ".join(
        column
        for column in REQUIRED_COLUMNS
        if row[column]
    ) if row.any() else "No missing variables",
    axis=1
)

combination_results = (
    df["missing_combination"]
    .value_counts()
    .reset_index()
)

combination_results.columns = [
    "missing_combination",
    "event_count"
]

combination_results["percentage"] = (
    combination_results["event_count"] / len(df)
) * 100


# ---------------------------------------------------------
# Calculate overall completeness
# ---------------------------------------------------------

complete_count = (df["missing_variable_count"] == 0).sum()
incomplete_count = (df["missing_variable_count"] > 0).sum()

complete_percentage = (complete_count / len(df)) * 100
incomplete_percentage = (incomplete_count / len(df)) * 100

overall_results = pd.DataFrame([
    {
        "category": "Complete observations",
        "event_count": complete_count,
        "percentage": complete_percentage
    },
    {
        "category": "Incomplete observations",
        "event_count": incomplete_count,
        "percentage": incomplete_percentage
    }
])


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n========================================")
print("MISSING DATA BY VARIABLE")
print("========================================")

print(
    variable_results_df.to_string(
        index=False,
        formatters={
            "available_percentage": "{:.2f}%".format,
            "missing_percentage": "{:.2f}%".format
        }
    )
)


print("\n========================================")
print("MISSING DATA BY NUMBER OF VARIABLES")
print("========================================")

print(
    category_results.to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


print("\n========================================")
print("SPECIFIC MISSING-VALUE COMBINATIONS")
print("========================================")

print(
    combination_results.to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


print("\n========================================")
print("OVERALL DATASET COMPLETENESS")
print("========================================")

print(
    overall_results.to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


# ---------------------------------------------------------
# Validate results
# ---------------------------------------------------------

print("\n========================================")
print("VALIDATION")
print("========================================")

print("Total events:", len(df))
print("Complete events:", complete_count)
print("Incomplete events:", incomplete_count)

print(
    "Complete + incomplete:",
    complete_count + incomplete_count
)

if complete_count + incomplete_count != len(df):
    raise ValueError("Missing-data counts do not add up to total events.")

if len(df) != TOTAL_EXPECTED_EVENTS:
    print(
        "\nWARNING: Dataset size differs from the expected "
        f"{TOTAL_EXPECTED_EVENTS:,} events."
    )

print("\nMissing-data analysis completed successfully.")


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

variable_file = os.path.join(
    OUTPUT_DIRECTORY,
    "missing_data_by_variable.csv"
)

category_file = os.path.join(
    OUTPUT_DIRECTORY,
    "missing_data_by_category.csv"
)

combination_file = os.path.join(
    OUTPUT_DIRECTORY,
    "missing_data_combinations.csv"
)

overall_file = os.path.join(
    OUTPUT_DIRECTORY,
    "missing_data_overall_summary.csv"
)

variable_results_df.to_csv(variable_file, index=False)
category_results.to_csv(category_file, index=False)
combination_results.to_csv(combination_file, index=False)
overall_results.to_csv(overall_file, index=False)

print("\nResults saved to:")
print(variable_file)
print(category_file)
print(combination_file)
print(overall_file)
