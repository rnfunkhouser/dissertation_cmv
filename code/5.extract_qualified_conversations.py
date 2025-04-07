#!/usr/bin/env python3
"""
Script to extract qualified conversations from a series of CSV files.

For each CSV file, it:
  - Loads the CSV file into a pandas DataFrame.
  - Converts string boolean values in the columns 'is_tlc', 'is_tlc_author', and 'is_submitter' to actual booleans.
  - Sorts by the 'created_utc' column.
  - Groups comments by the top-level comment id (tlc_id) to process each conversation thread.
  - Filters for conversation threads that have exactly 2 unique authors and where at least one comment has is_submitter==True and at least one has is_tlc_author==True.
  - For qualifying conversations:
      - Updates the top-level comment’s children_count to the total number of comments in the thread.
      - Concatenates all comments by the TLC author (in chronological order) into the top-level comment's body.
  - Returns the updated top-level comment (as a dictionary) for each qualifying conversation.

After processing all CSV files, the script combines the results and exports them to a single CSV file named "qualified_conversations_combined.csv".
"""

import pandas as pd
import glob
import os

def process_csv_file(input_csv):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total rows loaded: {len(df)}")
    
    # Convert potential string boolean values to actual booleans.
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() == 'true'
        return False
    
    df['is_tlc'] = df['is_tlc'].apply(to_bool)
    df['is_tlc_author'] = df['is_tlc_author'].apply(to_bool)
    df['is_submitter'] = df['is_submitter'].apply(to_bool)
    
    qualified_conversations = []  # List to store each qualifying conversation (as a dict).
    
    # Group the DataFrame by tlc_id.
    grouped = df.groupby('tlc_id')
    print("Processing groups by tlc_id...")
    
    for tlc_id, group in grouped:
        # Filter: conversation must have exactly 2 unique authors.
        unique_authors = group['author'].unique()
        if len(unique_authors) != 2:
            continue
        
        # Filter: at least one comment must have is_submitter==True and one with is_tlc_author==True.
        if not group['is_submitter'].any() or not group['is_tlc_author'].any():
            continue
        
        total_comments = len(group)
        
        # Identify the top-level comment row (where is_tlc is True).
        tlc_rows = group[group['is_tlc']]
        if tlc_rows.empty:
            continue
        tlc_row = tlc_rows.iloc[0].copy()
        
        # Update the children_count to be the total number of comments in this conversation.
        tlc_row['children_count'] = total_comments
        
        # Get all comments from the TLC author and sort them by the created_utc timestamp.
        tlc_author_comments = group[group['is_tlc_author']].copy()
        tlc_author_comments.sort_values('created_utc', inplace=True)
        
        # Build the concatenated body text.
        concatenated_body = tlc_row['body'] if pd.notnull(tlc_row['body']) else ""
        for _, row in tlc_author_comments.iterrows():
            # Skip the top-level comment since it's already included.
            if row['id'] == tlc_row['id']:
                continue
            comment_body = row['body'] if pd.notnull(row['body']) else ""
            concatenated_body += "\n\n-------\n\n" + comment_body
        
        # Update the top-level comment's body with the concatenated text.
        tlc_row['body'] = concatenated_body
        
        # Append the updated top-level comment (as a dict) to the qualified conversations list.
        qualified_conversations.append(tlc_row.to_dict())
        print(f"Qualified conversation found for TLC id: {tlc_id} with {total_comments} comments.")
    
    print(f"Total qualified conversations in {input_csv}: {len(qualified_conversations)}")
    return qualified_conversations

def process_all_files():
    all_qualified = []
    csv_files = glob.glob(os.path.join("..", "data", "processed_chunks", "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")
    for input_csv in csv_files:
        print(f"Processing file: {input_csv}")
        qualified = process_csv_file(input_csv)
        all_qualified.extend(qualified)
    return all_qualified

def main():
    all_qualified = process_all_files()
    print(f"Total qualified conversations from all files: {len(all_qualified)}")
    qualified_df = pd.DataFrame(all_qualified)
    output_csv = os.path.join("..", "data", "qualified_conversations.csv")
    qualified_df.to_csv(output_csv, index=False)
    print(f"Combined qualified conversations exported to {output_csv}")

if __name__ == "__main__":
    main()