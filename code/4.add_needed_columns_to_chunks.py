#adds data needed to reconstruct trees, such as top level comment data. 

import csv
import os
from datetime import datetime, timezone


# Define the RedditComment class with new attributes and methods
class RedditComment:
    def __init__(self, row):
        # Store the original row
        self.original = row.copy()
        # Original fields
        self.id = row['id']
        self.author = row['author']
        self.parent_id = row['parent_id']
        self.body = row.get('body', '').strip()
        # Store the original created_utc value
        self.raw_created_utc = row.get('created_utc', '')
        # Convert the raw_created_utc to a datetime object (posted_at)
        utc_val = self.raw_created_utc
        if utc_val:
            try:
                if isinstance(utc_val, str) and utc_val.isdigit():
                    ts = int(utc_val)
                else:
                    ts = utc_val
                self.posted_at = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception as e:
                self.posted_at = None
        else:
            self.posted_at = None
        self.is_submitter = str(row.get('is_submitter', '')).strip().lower() in ["true"]
        # New columns/attributes
        self.is_tlc = True if self.parent_id.startswith("t3_") else False
        self.tlc_id = None
        self.is_tlc_author = False
        self.children_count = None
        # For building the comment tree
        self.children = []
    
    def assign_tlc_info(self, tlc_id, tlc_author, visited=None):
        if visited is None:
            visited = set()
        # If this comment was already processed, skip to avoid cycles
        if self.id in visited:
            return
        visited.add(self.id)
        self.tlc_id = tlc_id
        self.is_tlc_author = (self.author == tlc_author)
        for child in self.children:
            child.assign_tlc_info(tlc_id, tlc_author, visited)
    
    def total_children(self, visited=None):
        if visited is None:
            visited = set()
        # If this comment was already processed, return 0 to avoid cycles
        if self.id in visited:
            return 0
        visited.add(self.id)
        count = len(self.children)
        for child in self.children:
            count += child.total_children(visited)
        return count

def process_chunk(filepath):
    """
    Reads a CSV chunk file, creates RedditComment objects, and rebuilds the comment tree.
    Returns a dictionary of all comments.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        all_comments_dict = {}
        top_level_comments = []
        
        for comment in reader:
            reddit_comment = RedditComment(comment)
            all_comments_dict[reddit_comment.id] = reddit_comment
            if reddit_comment.is_tlc:
                top_level_comments.append(reddit_comment)
            else:
                # For child comments, the parent's id is in parent_id with a "t1_" prefix removed.
                parent_comment_id = reddit_comment.parent_id[3:]
                parent_comment = all_comments_dict.get(parent_comment_id)
                if parent_comment:
                    parent_comment.children.append(reddit_comment)
                else:
                    # Parent is missing; skip linking this comment.
                    pass
        
        # For each top-level comment, assign its own id as the TLC id,
        # mark it as the TLC author, and propagate to all its children.
        for tlc in top_level_comments:
            tlc.tlc_id = tlc.id
            tlc.is_tlc_author = True
            for child in tlc.children:
                child.assign_tlc_info(tlc.id, tlc.author)
        
        # Compute the total number of descendant comments for each comment.
        for comment in all_comments_dict.values():
            comment.children_count = comment.total_children()
        
        return all_comments_dict

def update_csv_file(filepath):
    """
    Processes the CSV file to add new columns (is_tlc, tlc_id, is_tlc_author, children_count)
    and overwrites the original CSV with the updated data.
    """
    
    all_comments = process_chunk(filepath)
    
    # Create a new list of dictionaries from all_comments with the desired fields.
    updated_rows = []
    for comment in all_comments.values():
        # Start with the original row data
        row = comment.original.copy()
        
        # Update or add new columns
        row['is_tlc'] = comment.is_tlc
        row['tlc_id'] = comment.tlc_id
        row['is_tlc_author'] = comment.is_tlc_author
        row['children_count'] = comment.children_count
        
        # Also add the human-readable time column, posted_at
        row['posted_at'] = comment.posted_at.isoformat() if comment.posted_at else ""
        
        # Ensure the raw created_utc remains (if present)
        row['created_utc'] = comment.raw_created_utc
        
        updated_rows.append(row)
    
    if updated_rows:
        # Get the fieldnames from the keys of the first row (preserving original order if possible)
        fieldnames = list(updated_rows[0].keys())
    else:
        fieldnames = []
    
    # Overwrite the original CSV file with updated data.
    with open(filepath, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"Updated CSV file saved: {filepath}")

if __name__ == "__main__":
    input_dir = os.path.join("..", "data", "processed_chunks")
    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(input_dir, filename)
        print(f"Processing file: {filepath}")
        update_csv_file(filepath)
    print("✅ All files updated.")