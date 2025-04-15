import pandas as pd
import os

ALLOWED_COLUMNS = ['author', 'body', 'created_utc', 'is_submitter', 'id', 'link_id', 'parent_id', 'score', 'subreddit_id']

def purge_csv_columns(filepath):
    """Reads the CSV file and rewrites it with only the allowed columns."""
    # Add low_memory=False and escapechar
    df = pd.read_csv(filepath, low_memory=False, escapechar='\\')
    allowed_cols = [col for col in df.columns if col in ALLOWED_COLUMNS]
    df = df[allowed_cols]
    # Add explicit quoting parameters and escapechar
    df.to_csv(filepath, index=False, quoting=1, escapechar='\\', doublequote=True)

# Input folder with monthly chunks (relative to where this script is run from)
input_dir = os.path.join("..", "data", "monthly_chunks")
output_dir = input_dir  # Overwrite in place
os.makedirs(output_dir, exist_ok=True)

# Delta ID file
delta_ids_file = os.path.join("..", "data", "all_delta_ids.csv")
delta_ids_df = pd.read_csv(delta_ids_file)
delta_ids_set = set(delta_ids_df['comment_id'].astype(str))

# Loop over all CSV files in the monthly chunks folder
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"🔄 Processing {filename}...")
        purge_csv_columns(input_path)

        # Use improved reading parameters
        cmv_df = pd.read_csv(input_path, low_memory=False, escapechar='\\')
        print(f"Initial row count for {filename}: {len(cmv_df)}")
        if 'id' not in cmv_df.columns:
            raise ValueError(f"The file {filename} does not contain an 'id' column.")

        cmv_df['has_delta_from_OP'] = cmv_df['id'].astype(str).isin(delta_ids_set)
        # Custom helper function to check if a value is blank (either NaN or an empty string/whitespace)
        def is_blank(x):
            return pd.isna(x) or (isinstance(x, str) and x.strip() == '')
        
        # Remove rows with any blank cell in the DataFrame
        blank_mask = cmv_df.apply(lambda col: col.map(is_blank)).any(axis=1)
        if blank_mask.any():
            print(f"Diagnostic: Removing {blank_mask.sum()} rows with at least one blank cell for {filename}.")
            cmv_df = cmv_df[~blank_mask]
        print(f"Row count after adding delta column for {filename}: {len(cmv_df)}")

        # Use improved writing parameters
        cmv_df.to_csv(output_path, index=False, quoting=1, escapechar='\\', doublequote=True)

        true_count = cmv_df['has_delta_from_OP'].sum()
        false_count = len(cmv_df) - true_count
        print(f"✅ {filename}: {true_count} True, {false_count} False")

print("🎉 Done processing all monthly files.")