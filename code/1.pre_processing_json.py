#this file sorts the json by date and filters out records before a certain date

import json
import csv
import datetime
import os
import heapq
import time

# File paths
input_file = '../data/changemyview_comments.json'  # Input file with all subreddit comments
output_file = '../data/filtered_cmv_comments.json' # New output file for filtered & sorted results
double_objects_csv = '../data/double_objects.csv'       # CSV file to store lines with two JSON objects. There seemed to be a few glitchy ones like this.

# Parameters
CHUNK_SIZE = 100000  # Process this many records at a time before sorting and writing a chunk

# Counters for date field usage
count_created_at = 0
count_created_utc = 0

# Define cutoff date: Only keep records after July 4, 2017 (i.e. July 5th onward)
cutoff_date = datetime.datetime(2017, 7, 4)

# List to store temporary chunk file names
chunk_files = []
chunk_index = 0
current_chunk = []  # Will hold tuples of (record_date, record)

def parse_line(line):
    """
    Decode one or more JSON objects from a line.
    Returns a list of decoded objects.
    """
    results = []
    line = line.strip()
    if not line:
        return results
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(line):
        try:
            obj, idx = decoder.raw_decode(line[pos:])
            results.append(obj)
            pos += idx
            while pos < len(line) and line[pos].isspace():
                pos += 1
        except json.JSONDecodeError:
            break
    return results

def get_record_date(record):
    """
    Extract a datetime object from the record.
    - If 'created_utc' is available, it's assumed to be a Unix timestamp.
    - If 'CreatedAt' is available, it's parsed as an ISO formatted string
      (with a fallback format if needed).
    Returns None if no valid date can be parsed.
    """
    if 'created_utc' in record:
        try:
            ts = float(record['created_utc'])
            # Use fromtimestamp with timezone then remove tzinfo to mimic utcfromtimestamp
            return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    elif 'CreatedAt' in record:
        try:
            return datetime.datetime.fromisoformat(record['CreatedAt'])
        except Exception:
            try:
                return datetime.datetime.strptime(record['CreatedAt'], '%Y-%m-%d %H:%M:%S')
            except Exception:
                return None
    else:
        print("Skipping record: no valid date field found (neither 'created_utc' nor 'CreatedAt').")
        return None

# Open CSV file for writing double object records immediately
with open(double_objects_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Object1", "Object2"])

    # Process the input file line by line
    with open(input_file, 'r', encoding='utf-8') as infile:
        line_count = 0
        last_print_time = time.time()
        
        for line in infile:
            line_count += 1
            if time.time() - last_print_time > 5:
                total_records = chunk_index * CHUNK_SIZE + len(current_chunk)
                print(f"Processed {line_count} lines. Total records kept: {total_records}. Chunks written: {chunk_index}.")
                last_print_time = time.time()
                
            objs = parse_line(line)
            if not objs:
                continue

            # Check for double objects in one line
            if len(objs) > 1:
                if len(objs) >= 2:
                    csv_writer.writerow([json.dumps(objs[0]), json.dumps(objs[1])])
                else:
                    csv_writer.writerow([json.dumps(objs[0]), ""])
                # Skip further processing of this line
                continue

            # Single JSON object on the line
            record = objs[0]
            # Update field usage counts
            if 'created_utc' in record:
                count_created_utc += 1
            if 'CreatedAt' in record:
                count_created_at += 1

            rec_date = get_record_date(record)
            if rec_date is None:
                continue  # Skip record if no valid date is found

            # Only keep records with date strictly after July 4, 2017
            if rec_date > cutoff_date:
                current_chunk.append((rec_date, record))
            
            # If the chunk is big enough, sort and write it to a temporary file
            if len(current_chunk) >= CHUNK_SIZE:
                try:
                    current_chunk.sort(key=lambda x: x[0])
                    chunk_filename = f'chunk_{chunk_index}.json'
                    with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
                        for dt, rec in current_chunk:
                            json.dump({'date': dt.isoformat(), 'record': rec}, chunk_file)
                            chunk_file.write('\n')
                    chunk_files.append(chunk_filename)
                    print(f"Chunk {chunk_index} written with {len(current_chunk)} records.")
                except Exception as e:
                    print(f"Error writing chunk {chunk_index}: {e}")
                finally:
                    chunk_index += 1
                    current_chunk = []

    # After processing all lines, write any remaining records to a final chunk file.
    if current_chunk:
        current_chunk.sort(key=lambda x: x[0])
        chunk_filename = f'chunk_{chunk_index}.json'
        with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
            for dt, rec in current_chunk:
                json.dump({'date': dt.isoformat(), 'record': rec}, chunk_file)
                chunk_file.write('\n')
        chunk_files.append(chunk_filename)
        chunk_index += 1
        current_chunk = []

print(f"Finished processing input file. Total lines processed: {line_count}")

# Function to iterate through a sorted chunk file.
def chunk_iterator(chunk_file):
    try:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    dt = datetime.datetime.fromisoformat(data['date'])
                    yield (dt, data['record'])
                except Exception as e:
                    print(f"Error processing line in {chunk_file}: {e}")
    except Exception as e:
        print(f"Error opening chunk file {chunk_file}: {e}")

# Merge sorted chunks using heapq.merge, which efficiently merges multiple sorted inputs.
try:
    print("Starting merge process...")
    iterators = [chunk_iterator(cf) for cf in chunk_files]
    merged_iterator = heapq.merge(*iterators, key=lambda x: x[0])

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for dt, rec in merged_iterator:
            json.dump(rec, outfile)
            outfile.write('\n')
    print("Merge process completed successfully.")
except Exception as e:
    print(f"Error during merging process: {e}")

# Cleanup: remove temporary chunk files
for cf in chunk_files:
    os.remove(cf)

# Output summary information
print("Processing complete.")
print("Field usage:")
print("created_utc:", count_created_utc)
print("CreatedAt:", count_created_at)
print(f"Filtered and sorted records have been written to: {output_file}")
print(f"Double objects were exported to: {double_objects_csv}")