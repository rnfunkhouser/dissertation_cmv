import pandas as pd
import os
import csv
import sys

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)

def process_csv_file(input_path, output_path, delta_ids_set):
    print(f"Processing {os.path.basename(input_path)}...")
    
    # Use newline='' to ensure embedded line breaks in fields (like \r or \n in 'body') are handled properly
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        df = pd.read_csv(f)
    print(f"Initial row count: {len(df)}")
    
    # Add delta information: mark True if the 'id' is in delta_ids_set, otherwise False
    df['has_delta_from_OP'] = df['id'].astype(str).isin(delta_ids_set)
    
    # Clean the body column since a few comments have \r in them and break the CSV
    df['body'] = df['body'].astype(str).str.replace('\r', ' ', regex=False)
    
    # Use newline='' to prevent extra line breaks in the output, especially with embedded \r or \n in 'body'
    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        # Convert DataFrame to a list of dicts to write with custom quoting for 'body'
        rows = df.to_dict(orient='records')
        fieldnames = df.columns.tolist()
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, escapechar='\\')

        writer.writeheader()
        for row in rows:
            if 'body' in row and row['body'] is not None:
                row['body'] = f'"{row["body"]}"' if not (str(row['body']).startswith('"') and str(row['body']).endswith('"')) else row['body']
            writer.writerow(row)
    
    # Calculate and print statistics
    true_count = df['has_delta_from_OP'].sum()
    false_count = len(df) - true_count
    print(f"✅ {os.path.basename(input_path)}: {true_count} True, {false_count} False")
    
    return len(df)

# Main script
input_dir = os.path.join("..", "data", "monthly_chunks")
output_dir = os.path.join("..", "data", "processed_chunks")
os.makedirs(output_dir, exist_ok=True)

# Delta ID file
delta_ids_file = os.path.join("..", "data", "all_delta_ids.csv")
delta_ids_df = pd.read_csv(delta_ids_file)
delta_ids_set = set(delta_ids_df['comment_id'].astype(str))

# Process all CSV files
total_processed = 0
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            rows_processed = process_csv_file(input_path, output_path, delta_ids_set)
            total_processed += rows_processed
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"🎉 Done processing all monthly files. Total rows processed: {total_processed}")