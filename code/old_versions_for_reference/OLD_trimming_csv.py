import pandas as pd

# Load the CSV file
input_file = "../data/CMV_July_2022_with_deltas.csv"  # Update this path if needed
output_file = "../data/CMV_purged_columns.csv"

df = pd.read_csv(input_file)

print(len(df), "rows loaded from", input_file)
# List of columns to drop
columns_to_drop = [
    "score", "edited", "time_edited",
    "subreddit_type", "subreddit_id", "stickied", "error"
]

# Drop the columns if they exist in the DataFrame
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Filter out rows where 'author' or 'body' is '[removed]' or '[deleted]'
removed_values = ['[removed]', '[deleted]']
df_cleaned = df_cleaned[~df_cleaned['author'].isin(removed_values) & ~df_cleaned['body'].isin(removed_values)]

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")
print(f"Removed {len(df) - len(df_cleaned)} rows with '[removed]' or '[deleted]' values in 'author' or 'body' columns.")