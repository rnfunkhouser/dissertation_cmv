import json

def search_ndjson(file_path, target_id):
    """
    Search through an NDJSON file for a JSON object where the "id" field equals target_id.
    
    Args:
        file_path (str): Path to the NDJSON file.
        target_id (str): The id value to search for.
    
    Returns:
        dict or None: The JSON object if found, otherwise None.
    """
    with open(file_path, 'r') as f:
        # Process the file one line at a time to handle large files.
        for line in f:
            try:
                # Parse the JSON object from the current line.
                record = json.loads(line)
            except json.JSONDecodeError as e:
                # Log the error and skip the problematic line.
                print(f"Error parsing line: {e}")
                continue
            
            # Check if the "id" field exists and matches the target_id.
            if record.get("id") == target_id:
                return record
    return None

if __name__ == "__main__":
    # Manually set the file path to your NDJSON file
    file_path = "/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/changemyview_comments.json"  # Update this with your actual file path
    target_id = "jl1ybbi"
    
    result = search_ndjson(file_path, target_id)
    
    if result:
        print("Found object:")
        print(json.dumps(result, indent=2))
    else:
        print(f"No object found with id: {target_id}")