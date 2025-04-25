"""
This script processes a JSON-lines file in two passes.
The first pass builds a "first appearance" dictionary mapping each link_id
to the month (YYYY-MM) of its earliest occurrence. The second pass streams
through the file again, unescapes HTML, converts the UTC timestamp into a 
human-readable datetime, and then writes each record to a CSV file 
corresponding to the month of its first appearance.
"""

import os
import json
import csv
import datetime
import html

ALLOWED_COLUMNS = ['author', 'body', 'created_utc', 'is_submitter', 'id', 'link_id', 'parent_id', 'score', 'subreddit_id', 'datetime']

def parse_record(line):
    """Parse a JSON string to a Python dict."""
    try:
        record = json.loads(line)
        return record
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None

def get_month_from_utc(utc_value):
    """
    Convert the UTC timestamp (assumed to be Unix timestamp in seconds)
    to a datetime object and a string representing the month (YYYY-MM).
    """
    try:
        # Convert to int in case it is a string.
        timestamp = int(utc_value)
        dt = datetime.datetime.utcfromtimestamp(timestamp)
        month_str = dt.strftime("%Y-%m")
        return month_str, dt
    except Exception as e:
        print(f"Error converting utc value {utc_value}: {e}")
        return None, None

def first_pass(input_file):
    """
    First pass: Build a dictionary mapping each link_id to the month of its
    first appearance. Assumes that if a link_id appears more than once,
    the earliest month is retained.
    """
    first_appearance = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = parse_record(line)
            if not record:
                continue
            # Require the 'utc' and 'link_id' keys.
            utc_value = record.get("created_utc")
            link_id = record.get("link_id")
            if utc_value is None or link_id is None:
                continue

            month_str, _ = get_month_from_utc(utc_value)
            if month_str is None:
                continue

            # If this is the first time we see the link_id, or if the current month
            # is earlier (using string comparison on YYYY-MM works correctly), update it.
            if (link_id not in first_appearance) or (month_str < first_appearance[link_id]):
                first_appearance[link_id] = month_str

            if i % 100000 == 0 and i > 0:
                print(f"First pass processed {i} records")
    print(f"Completed first pass. Unique link_ids: {len(first_appearance)}")
    return first_appearance

def second_pass(input_file, first_appearance, output_dir):
    """
    Second pass: Open one CSV for each unique month as determined by first_pass.
    For every record, look up the link_id to decide which CSV file to write the record to.
    Additionally, unescape HTML fields and convert the utc timestamp to a new datetime field.
    """
    # Determine unique months from the first appearance dictionary.
    months = set(first_appearance.values())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare CSV writers and file handles for each month.
    csv_writers = {}
    file_handles = {}
    header_written = {}
    for month in months:
        output_path = os.path.join(output_dir, f"data_{month}.csv")
        # Use newline='' to prevent extra blank lines or newline misinterpretation on different OSes,
        # especially important when fields like 'body' contain \r or \n characters.
        f_out = open(output_path, 'w', newline='', encoding='utf-8')
        file_handles[month] = f_out
        writer = csv.DictWriter(f_out, fieldnames=ALLOWED_COLUMNS, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        csv_writers[month] = writer
        header_written[month] = False

    # Process the file line by line.
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = parse_record(line)
            if record is None:
                continue
            
            # Convert the utc field into a datetime field (string format).
            utc_value = record.get("created_utc")
            month_str_current, dt = get_month_from_utc(utc_value)
            if dt:
                record["datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Determine the correct month to assign this record based on its link_id.
            link_id = record.get("link_id")
            if link_id is None:
                continue
            first_month = first_appearance.get(link_id)
            if not first_month:
                continue
            
            writer = csv_writers[first_month]
            # Write header if not already written
            if not header_written[first_month]:
                writer.writeheader()
                header_written[first_month] = True
            
            # Process the "body" field for proper CSV handling: unescape HTML only
            if 'body' in record and record['body'] is not None:
                record['body'] = html.unescape(record['body'])
            
            # Create a filtered record that contains only the allowed columns
            filtered_record = {key: record.get(key, "") for key in ALLOWED_COLUMNS}
            
            # Write the filtered record using DictWriter
            writer.writerow(filtered_record)

            if i % 100000 == 0 and i > 0:
                print(f"Second pass processed {i} records")

    # Close all open CSV files.
    for f_out in file_handles.values():
        f_out.close()
    print("Completed second pass.")

if __name__ == "__main__":
    # Set your input file and output directory here (update paths as needed)
    input_file = "../data/filtered_cmv_comments.json"  
    output_dir = "../data/monthly_chunks/"   

    print("Starting first pass to build first-appearance dictionary...")
    first_appearance = first_pass(input_file)
    
    print("Starting second pass to stream records into month-specific CSVs...")
    second_pass(input_file, first_appearance, output_dir)
    
    print("Processing complete.")
