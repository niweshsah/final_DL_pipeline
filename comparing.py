import pandas as pd

# Load the two CSVs
csv1_path = "/DATA/yolo_on_brain/evaluation_results_less_threshold.csv"  # First CSV path
csv2_path = "/DATA/yolo_on_brain/evaluation_results.csv"                # Second CSV path

df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Rename columns in df2 to distinguish them (add suffix "_2")
df2_renamed = df2.add_suffix("_2")

# Merge on Case column (inner join to keep only common cases)
merged = pd.merge(df1, df2_renamed, left_on="Case", right_on="Case_2", how="inner")

# Drop duplicate Case_2 column (same as Case)
merged.drop(columns=["Case_2"], inplace=True)

# Calculate differences for key metrics: Dice, FP%, FN%, TP%
merged["Dice_diff"] = merged["Dice"] - merged["Dice_2"]
merged["FP (%)_diff"] = merged["FP (%)"] - merged["FP (%)_2"]
merged["FN (%)_diff"] = merged["FN (%)"] - merged["FN (%)_2"]
merged["TP (%)_diff"] = merged["TP (%)"] - merged["TP (%)_2"]

# Determine which CSV is better per case for FN% (lower FN% is better)
def better_fn_label(row):
    if row["FN (%)"] == row["FN (%)_2"]:
        return "equal"
    return "csv1" if row["FN (%)"] < row["FN (%)_2"] else "csv2"

merged["Better_FN%"] = merged.apply(better_fn_label, axis=1)

# Print summary of differences (mean, median, min, max)
print("=== Overall Comparison Summary ===\n")
for metric in ["Dice_diff", "FP (%)_diff", "FN (%)_diff", "TP (%)_diff"]:
    print(f"{metric}:")
    print(f"  Mean:   {merged[metric].mean():.4f}")
    print(f"  Median: {merged[metric].median():.4f}")
    print(f"  Min:    {merged[metric].min():.4f}")
    print(f"  Max:    {merged[metric].max():.4f}")
    print()

# Print FN% better counts summary
fn_better_counts = merged["Better_FN%"].value_counts()
print("FN% Comparison (Lower FN% is better):")
print(f"  Cases where csv1 has better (lower) FN%: {fn_better_counts.get('csv1', 0)}")
print(f"  Cases where csv2 has better (lower) FN%: {fn_better_counts.get('csv2', 0)}")
print(f"  Cases where FN% is equal: {fn_better_counts.get('equal', 0)}\n")

# Top 5 cases where csv1 improves FN% the most (FN% lower in csv1)
print("Top 5 cases with biggest FN% improvement (csv1 better than csv2):")
print(
    merged[merged["Better_FN%"] == "csv1"]
    .sort_values(by="FN (%)_diff")
    .head(5)[["Case", "FN (%)", "FN (%)_2", "FN (%)_diff", "Dice", "Dice_2"]]
)
print()

# Top 5 cases where csv2 has better FN% (csv1 worse)
print("Top 5 cases with biggest FN% drop (csv2 better than csv1):")
print(
    merged[merged["Better_FN%"] == "csv2"]
    .sort_values(by="FN (%)_diff", ascending=False)
    .head(5)[["Case", "FN (%)", "FN (%)_2", "FN (%)_diff", "Dice", "Dice_2"]]
)
print()

# Also keep your Dice improvement/drop summary for quick reference
print("Top 5 cases with biggest Dice improvement (csv1 better than csv2):")
print(merged.sort_values(by="Dice_diff", ascending=False)[["Case", "Dice", "Dice_2", "Dice_diff"]].head(5))
print()

print("Top 5 cases with biggest Dice drop (csv2 better than csv1):")
print(merged.sort_values(by="Dice_diff", ascending=True)[["Case", "Dice", "Dice_2", "Dice_diff"]].head(5))

# Save merged comparison to CSV
merged.to_csv("comparison_results_with_fn_focus.csv", index=False)
print("\nMerged comparison saved to 'comparison_results_with_fn_focus.csv'")



