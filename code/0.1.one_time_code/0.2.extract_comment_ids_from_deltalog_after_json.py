import pandas as pd
import re

def extract_comment_ids_from_text(text):
    """Extracts comment IDs from text between '# Deltas from OP' and '#Deltas from Other Users'."""
    extracted_ids = []
    
    # Find the relevant section of the text
    match = re.search(r"# Deltas from OP(.*?)#Deltas from Other Users", text, re.DOTALL)
    if match:
        relevant_text = match.group(1)
        
        # Find all URLs starting with the specified pattern
        urls = re.findall(r"\]\(/r/changemyview/comments/[^)]+", relevant_text)

        for url in urls:
            # Extract the comment ID that precedes "/?context=3"
            parts = url.split("/")
            if len(parts) > 2 and "?context=3" in parts[-1]:
                comment_id = parts[-2]  # Extract the correct ID
                extracted_ids.append(comment_id)

    return extracted_ids

def extract_comment_ids(input_csv):
    """Reads a CSV, extracts comment IDs using the new method, and saves them to a CSV."""
    df = pd.read_csv(input_csv)
    all_comment_ids = []

    for index, text in df["selftext"].dropna().items():  # Drop NaN values to avoid errors
        comment_ids = extract_comment_ids_from_text(text)
        if comment_ids:
            all_comment_ids.extend(comment_ids)
            print(f"Index {index}: Extracted {len(comment_ids)} comment IDs")  # Debugging

    print(f"\nTotal extracted comment IDs: {len(all_comment_ids)}")  # Debugging summary
    return all_comment_ids

# File paths
input_csv = "./data/deltalog_submissions.csv"

# Run the extraction function
extracted_comment_ids = extract_comment_ids(input_csv)

# Save to a new CSV file
output_csv = "./data/delta_ids_after_json.csv"
pd.DataFrame(extracted_comment_ids, columns=["comment_id"]).to_csv(output_csv, index=False)

print(f"Extracted comment IDs saved to {output_csv}")