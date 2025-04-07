import pandas as pd

def extract_first_1000_rows(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Select the first 1000 rows
    df_subset = df.head(1000)
    
    # Save to a new CSV file
    df_subset.to_csv(output_csv, index=False)
    
    print(f"First 1000 rows saved to {output_csv}")

# Example usage
input_file = "./data/CMV_purged_columns.csv"  # Change this to your actual file
output_file = "./data/1000_rows_cmv_sample.csv"
extract_first_1000_rows(input_file, output_file)