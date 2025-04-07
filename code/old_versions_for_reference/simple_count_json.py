import os

def count_ndjson_entries(filename):
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                count += 1
    return count

if __name__ == '__main__':
    # Set the path to your NDJSON file here.
    filename = "/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/filtered_sorted_comments.json"  # Replace with your file path
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
    else:
        total_entries = count_ndjson_entries(filename)
        print(f"Total entries: {total_entries}")