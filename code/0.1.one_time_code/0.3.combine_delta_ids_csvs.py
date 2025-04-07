import pandas as pd

# Define file paths directly
file1 = "./data/delta_ids_after_json.csv"  # Replace with actual file name
file2 = "./data/delta_ids_pre_json.csv"
output_file = "./data/all_delta_ids.csv"

# Load the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Print row counts before merging
print(f"📊 Row count in {file1}: {len(df1)}")
print(f"📊 Row count in {file2}: {len(df2)}")

# Check for duplicates based on all columns
duplicates = pd.merge(df1, df2, how="inner")

if not duplicates.empty:
    print("❌ Duplicates found! Resolve duplicates before merging.")
    print(duplicates)
else:
    print("✅ No duplicates found. Merging files...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    # Print row count after merging
    print(f"📊 Row count in merged file: {len(merged_df)}")

    merged_df.to_csv(output_file, index=False)
    print(f"✅ Merged file saved as {output_file}")