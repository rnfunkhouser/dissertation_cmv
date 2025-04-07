import pandas as pd
import os
import csv
import sys

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)

ALLOWED_COLUMNS = ['author', 'body', 'created_utc', 'is_submitter', 'id', 'link_id', 'parent_id', 'score', 'subreddit_id']

def process_csv_file(input_path, output_path, delta_ids_set):
    """Process CSV file by reading with csv module and writing with pandas."""
    print(f"Processing {os.path.basename(input_path)}...")
    
    # Read header to determine column indices
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        
    # Get indices of columns we want to keep
    keep_indices = []
    column_map = {}  # Maps file indices to our column names
    
    for i, col_name in enumerate(header):
        if col_name in ALLOWED_COLUMNS:
            keep_indices.append(i)
            column_map[i] = col_name
    
    # Get the index of the 'id' column
    if 'id' not in column_map.values():
        raise ValueError(f"The file {os.path.basename(input_path)} does not contain an 'id' column.")
    id_index = [i for i, col in column_map.items() if col == 'id'][0]
    
    # Prepare data structure for the filtered rows
    filtered_data = {col: [] for col in ALLOWED_COLUMNS if col in column_map.values()}
    
    # Read and filter the data
    row_count = 0
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            row_count += 1
            if len(row) >= max(keep_indices) + 1:  # Make sure row has enough columns
                # Check for blank cells
                has_blank = False
                for idx in keep_indices:
                    if idx < len(row) and (not row[idx] or row[idx].strip() == ''):
                        has_blank = True
                        break
                
                if not has_blank:
                    for idx in keep_indices:
                        if idx < len(row):
                            filtered_data[column_map[idx]].append(row[idx])
                        else:
                            filtered_data[column_map[idx]].append("")  # Fill with empty if column missing
    
    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)
    print(f"Initial row count: {row_count}, Filtered rows: {len(df)}")
    
    # Add delta information
    df['has_delta_from_OP'] = df['id'].astype(str).isin(delta_ids_set)
    
    # Write to output file - use a specialized approach for handling the body field
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Create new columns list with has_delta_from_OP added
        output_columns = list(df.columns)
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        
        # Write header
        writer.writerow(output_columns)
        
        # Write data rows, handling each row directly
        for _, row in df.iterrows():
            writer.writerow([row[col] for col in output_columns])
    
    # Calculate statistics
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