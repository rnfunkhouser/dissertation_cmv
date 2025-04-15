import json
import re
import csv

def extract_ids_from_selftext(text):
    """
    Extracts comment IDs from the approved section of selftext.
    Approved section is defined as the text between:
      "# Deltas from OP" and "#Deltas from Other Users"
    
    Within that section, any URL that contains the string 
    "]/r/changemyview/comments/" is searched. The comment ID is 
    assumed to be the segment between the last "/" and "/?context=3)".
    
    Returns a list of found comment IDs.
    """
    start_marker = "# Deltas from OP"
    end_marker = "#Deltas from Other Users"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    
    # Check if both markers exist and are in the proper order
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return []
    
    # Extract only the approved portion of the text
    approved_text = text[start_index:end_index]
    
    # Define regex pattern:
    # - Look for the literal string "]/r/changemyview/comments/"
    # - Then lazily match any characters until the next "/" 
    # - Then capture the comment ID (any characters not a slash)
    # - Followed by the literal string "/?context=3)"
    pattern = re.compile(r'\]\(/r/changemyview/comments/.*?/([^/]+)/\?context=3\)')
    
    # Find all occurrences of the pattern and return the list of comment IDs
    comment_ids = pattern.findall(approved_text)
    return comment_ids

def process_ndjson(input_file, output_file):
    """
    Processes an NDJSON file to extract comment IDs from the 'selftext' field
    and writes the results to a CSV file.
    
    Each JSON object is expected to contain a 'selftext' field. The function
    extracts text between "# Deltas from OP" and "#Deltas from Other Users",
    finds all matching comment IDs, and writes each found ID as a separate row
    in the output CSV.
    """
    extracted_ids = []
    
    # Open and process the NDJSON file line by line
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: Error decoding JSON: {e}")
                continue

            selftext = data.get('selftext', '')
            if selftext:
                ids = extract_ids_from_selftext(selftext)
                extracted_ids.extend(ids)
    
    # Write the extracted IDs to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['comment_id'])  # Write header row
        for cid in extracted_ids:
            writer.writerow([cid])
    
    print(f"Extraction complete. {len(extracted_ids)} comment IDs written to {output_file}")

if __name__ == '__main__':
    # Hardcode the input and output file paths
    input_file = '/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/deltalog_entries.json'  # Replace with your NDJSON file path
    output_file = '../data/deltalog_output_new.csv'             # Replace with your desired CSV file path
    
    process_ndjson(input_file, output_file)