import pandas as pd
import json
import re

def extract_json_from_text(text):
    """Extracts JSON from text, first trying DB3PARAMSSTART, then looking for any JSON block."""
    
    # Try to find JSON with DB3PARAMSSTART
    json_match = re.search(r"DB3PARAMSSTART\s*(\{.*?\})\s*DB3PARAMSEND", text, re.DOTALL)
    
    if json_match:
        return json_match.group(1)  # Return JSON if found with markers
    
    # Fallback: Look for any JSON block
    json_matches = re.findall(r'(\{.*?\})', text, re.DOTALL)
    
    if json_matches:
        return json_matches[0]  # Return first valid JSON block found
    
    return None  # No JSON found

def extract_comment_id_from_url(url):
    """Extracts only the Reddit comment ID from the awardedLink URL."""
    if isinstance(url, str):
        # Use regex to match a Reddit comment ID at the end of the URL (6-7 alphanumeric characters)
        match = re.search(r'/([a-z0-9]{6,7})$', url)
        if match:
            return match.group(1)  # Extract and return just the comment ID
    return None  # Return None if no valid ID is found

def extract_comment_ids(input_csv):
    """Reads a CSV, extracts comment IDs where the awarding user is the OP from JSON in the 'selftext' column."""
    df = pd.read_csv(input_csv)
    comment_ids = []
    
    for index, text in df["selftext"].dropna().items():  # Drop NaN values to avoid errors
        json_str = extract_json_from_text(text)
        if json_str:
            try:
                json_data = json.loads(json_str)  # Convert to dictionary
                op_username = json_data.get("opUsername", "")  # Get OP username
                
                for comment in json_data.get("comments", []):
                    if comment.get("awardingUsername") == op_username:
                        awarded_link = comment.get("awardedLink")
                        comment_id = extract_comment_id_from_url(awarded_link)
                        if comment_id:
                            comment_ids.append(comment_id)
                            print(f"Match found! Index: {index}, Extracted Comment ID: {comment_id}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at index {index}")
                continue  # Skip invalid JSON entries
    
    print(f"\nTotal comment IDs extracted: {len(comment_ids)}")
    return comment_ids

# File paths
input_csv = "./data/deltalog_submissions.csv"

# Run the extraction function
extracted_comment_ids = extract_comment_ids(input_csv)

# Save to a new CSV file
output_csv = "./data/delta_ids_pre_json.csv"
pd.DataFrame(extracted_comment_ids, columns=["comment_id"]).to_csv(output_csv, index=False)

print(f"Extracted comment IDs saved to {output_csv}")