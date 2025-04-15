import json
import os
from datetime import datetime, timezone


def scan_for_is_submitter_start(json_path, scan_lines=10000, threshold=0.9):
    """
    Scans from the beginning of the JSON file to find the first record where 'author_fullname' is present.
    Then, it verifies that 'author_fullname' appears consistently in the next `scan_lines` records.
    If at least `threshold` fraction of records in that batch have 'is_submitter', it prints the UTC timestamp
    of the first occurrence.
    """
    print("Starting scan for consistent 'author_fullname' appearance from the beginning of the file.")
    found = False
    first_timestamp = None
    first_line = None

    # Open the file in text mode
    with open(json_path, 'r', encoding='utf-8') as f:
        line_num = 0
        # Search for the first record with a non-null 'is_submitter'
        for line in f:
            line_num += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "author_fullname" in record and record["author_fullname"] is not None:
                first_line = line_num
                # Try to extract and convert the 'created_utc' value
                if 'created_utc' in record:
                    try:
                        # If created_utc is a string of digits, convert it
                        if isinstance(record['created_utc'], str) and record['created_utc'].isdigit():
                            ts = int(record['created_utc'])
                        else:
                            ts = record['created_utc']
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    except Exception as e:
                        dt = None
                else:
                    dt = None
                first_timestamp = dt
                print(f"Found first record with 'author_fullname' at line {line_num}.")
                found = True
                break
        
        if not found:
            print("No record with 'author_fullname' found in the file.")
            return
        
        # Now, verify that 'is_submitter' appears consistently in the next scan_lines records
        count_with_is_submitter = 0
        total = 0
        for i in range(scan_lines):
            line = f.readline()
            if not line:
                print("Reached end of file before scanning all lines in the verification batch.")
                break
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "author_fullname" in record and record["author_fullname"] is not None:
                count_with_is_submitter += 1

    fraction = count_with_is_submitter / total if total > 0 else 0
    if fraction >= threshold:
        print(f"'author_fullname' appears consistently in the next {total} records ({fraction*100:.2f}% of records).")
        if first_timestamp:
            print(f"First consistent 'author_fullname' appearance at UTC: {first_timestamp.isoformat()} (line {first_line}).")
        else:
            print("No 'created_utc' value available for the first record with 'is_submitter'.")
    else:
        print(f"'author_fullname' does not appear consistently in the next {total} records ({fraction*100:.2f}% of records).")


if __name__ == "__main__":
    json_path = os.path.join("..", "data", "changemyview_comments.json")
    scan_for_is_submitter_start(json_path)