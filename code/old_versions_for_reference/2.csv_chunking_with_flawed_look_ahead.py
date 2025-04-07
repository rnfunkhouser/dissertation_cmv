# Configuration - Set your input and output paths here
INPUT_FILE_PATH = '../data/filtered_sorted_comments.json'
OUTPUT_DIRECTORY = '../data/monthly_chunks'

import json
import csv
import os
import datetime
from collections import defaultdict

def convert_utc_to_datetime(utc_timestamp):
    """Convert UTC timestamp to datetime object."""
    return datetime.datetime.utcfromtimestamp(int(utc_timestamp))

def get_month_key(dt):
    """Return a string key in format YYYY-MM for a datetime object."""
    return f"{dt.year}-{dt.month:02d}"

def process_reddit_comments(input_file, output_dir):
    """Process Reddit comments NDJSON file into monthly CSV files."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Scanning file to identify all months of data...")
    # First pass: Identify all months in the dataset
    all_months = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(f"Scanning line {line_num}...")
                
            try:
                comment = json.loads(line.strip())
                if 'created_utc' not in comment:
                    continue
                    
                dt = convert_utc_to_datetime(comment['created_utc'])
                month_key = get_month_key(dt)
                all_months.add(month_key)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Sort months chronologically
    all_months = sorted(all_months)
    print(f"Found data for {len(all_months)} months: {', '.join(all_months)}")
    
    # Process each month
    for i, current_month in enumerate(all_months):
        print(f"\nProcessing month: {current_month} ({i+1}/{len(all_months)})")
        
        # Get the next month for look-ahead (if available)
        next_month = all_months[i+1] if i < len(all_months) - 1 else None
        
        # First collect all post IDs from the current month
        current_month_post_ids = set()
        skipped_entries = 0
        
        print(f"Pass 1: Collecting post IDs for {current_month}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    comment = json.loads(line.strip())
                    
                    # Check for required fields
                    if 'created_utc' not in comment:
                        skipped_entries += 1
                        if skipped_entries <= 10:  # Limit the number of warnings to avoid flooding
                            print(f"Warning: Skipping entry at line {line_num} - missing 'created_utc' field")
                        continue
                        
                    if 'link_id' not in comment:
                        skipped_entries += 1
                        if skipped_entries <= 10:  # Limit the number of warnings to avoid flooding
                            print(f"Warning: Skipping entry at line {line_num} - missing 'link_id' field")
                        continue
                    
                    dt = convert_utc_to_datetime(comment['created_utc'])
                    month_key = get_month_key(dt)
                    
                    if month_key == current_month:
                        current_month_post_ids.add(comment['link_id'])
                        
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        if skipped_entries > 10:
            print(f"...and {skipped_entries - 10} more entries with missing fields")
        
        print(f"Found {len(current_month_post_ids)} unique post IDs for {current_month}")
        
        # Now collect all comments for this month, plus spillovers from next month
        current_month_data = []
        spillover_count = 0
        
        print(f"Pass 2: Collecting comments for {current_month} and spillover comments...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    comment = json.loads(line.strip())
                    
                    # Skip if missing required fields
                    if 'created_utc' not in comment or 'link_id' not in comment:
                        continue
                        
                    dt = convert_utc_to_datetime(comment['created_utc'])
                    month_key = get_month_key(dt)
                    
                    # Add current month's comments
                    if month_key == current_month:
                        current_month_data.append(comment)
                    
                    # Add next month's comments that belong to current month's posts
                    elif next_month and month_key == next_month and comment['link_id'] in current_month_post_ids:
                        current_month_data.append(comment)
                        spillover_count += 1
                        
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        # Write current month's data to CSV
        if current_month_data:
            write_to_csv(current_month_data, current_month, output_dir)
            print(f"Wrote {len(current_month_data)} comments to {current_month}.csv")
            print(f"Including {spillover_count} spillover comments from {next_month if next_month else 'N/A'}")
        else:
            print(f"No data to write for month {current_month}")
        
        # Clear data to free memory
        current_month_data = None
        current_month_post_ids = None

def write_to_csv(comments, month_key, output_dir):
    """Write comments to a CSV file."""
    output_file = os.path.join(output_dir, f"{month_key}.csv")
    
    # Get all possible fields from the first comment
    if not comments:
        return
        
    fieldnames = list(comments[0].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for comment in comments:
            # Ensure all fields exist (use None for missing fields)
            row = {field: comment.get(field, None) for field in fieldnames}
            writer.writerow(row)

if __name__ == '__main__':
    print(f"STARTING PROCESSING")
    print(f"Input file: {INPUT_FILE_PATH}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    
    process_reddit_comments(INPUT_FILE_PATH, OUTPUT_DIRECTORY)
    print("Processing complete!")