import pandas as pd
import os

input_dir = os.path.join("..", "data", "monthly_chunks")
output_dir = os.path.join("..", "data", "monthly_chunks")
columns_to_keep = [
    "parent_id",
    "score",
    "created_utc",
    "author",
    "subreddit_id",
    "body",
    "id",
    "link_id"
]
removed_values = ['[removed]', '[deleted]']

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        df = pd.read_csv(input_path)
        print(len(df), "rows loaded from", input_path)

        # Determine which columns to keep that are present in the DataFrame
        available_columns = [col for col in columns_to_keep if col in df.columns]
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            print(f"Note: The following expected columns are missing in {input_path}: {missing_columns}")

        # Select only the available columns
        df_cleaned = df[available_columns]

        # Remove rows where 'author' or 'body' contain removed values (if these columns exist)
        if 'author' in available_columns and 'body' in available_columns:
            df_cleaned = df_cleaned[~df_cleaned['author'].isin(removed_values) & ~df_cleaned['body'].isin(removed_values)]

        df_cleaned.to_csv(input_path, index=False)
        print(f"Cleaned dataset saved to {input_path}")
        print(f"Removed {len(df) - len(df_cleaned)} rows with '[removed]' or '[deleted]' values in 'author' or 'body' columns.")