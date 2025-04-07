import pandas as pd

# File paths
cmv_file = "./data/CMV_July_2022.csv"
delta_ids_file = "./data/all_delta_ids.csv"
output_file = "./data/CMV_July_2022_with_deltas.csv"

# Load the CMV dataset
cmv_df = pd.read_csv(cmv_file)

# Load the external set of IDs (assuming it has a column named 'comment_id')
delta_ids_df = pd.read_csv(delta_ids_file)

# Extract the set of valid IDs from only_OP_delta_ids_output.csv
delta_ids_set = set(delta_ids_df['comment_id'])

# Ensure CMV dataset has an ID column (assuming it's named 'id')
if 'id' not in cmv_df.columns:
    raise ValueError("The CMV dataset does not contain an 'id' column.")

# Create a new column 'has_delta_from_OP' based on whether the ID is in the delta set
cmv_df['has_delta_from_OP'] = cmv_df['id'].astype(str).isin(delta_ids_set)

# Save the updated CSV file
cmv_df.to_csv(output_file, index=False)

print(f"Updated file saved as: {output_file}")

# Count True/False values in "has_delta_from_OP" (only in the merged file)
if "has_delta_from_OP" in cmv_df.columns:
    true_count = (cmv_df["has_delta_from_OP"] == True).sum()
    false_count = (cmv_df["has_delta_from_OP"] == False).sum()
    print(f"📊 In merged file: {true_count} rows where has_delta_from_OP = True")
    print(f"📊 In merged file: {false_count} rows where has_delta_from_OP = False")

