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
        self.body = (row.get('body') or '').strip()
        self.is_submitter = str(row.get('is_submitter', '')).strip().lower() in ["true"]
        if not self.parent_id:
            print(f"Warning: Missing parent_id in row: {row}")
        self.is_tlc = str(self.parent_id).startswith("t3_")
        self.tlc_id = None
        self.is_tlc_author = False
        self.children_count = None
        self.actually_has_delta = False  
        #Be aware that the column name "has_delta_from_OP" is a misnomer since that field is 
        #actually True for the OP comment awarding the delta (starting June 23, 2018, which is why we begin July 2018) 
        self.op_delta = str(row.get("has_delta_from_OP", "")).strip().lower() in ["true"] 
        self.op_name = None
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

    def assign_op_name(self):
        if self.is_submitter:
            self.op_name = self.author
    
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
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        all_comments_dict = {}
        top_level_comments = []
        
        for comment in reader:
            reddit_comment = RedditComment(comment)
            all_comments_dict[reddit_comment.id] = reddit_comment
            if reddit_comment.is_tlc:
                top_level_comments.append(reddit_comment)
            else:
                if reddit_comment.parent_id is None:
                    print(f"Skipping comment with missing parent_id: {reddit_comment.original}")
                    continue
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
            # Assign the OP name if this comment is the submitter.
            if comment.is_submitter:
                comment.assign_op_name()
        return all_comments_dict

def update_parent_delta_flags(all_comments_dict):
    parent_not_found_count = 0
    delta_marked_count = 0
    for comment in all_comments_dict.values():
        if comment.op_delta:
            #print(f"Child comment {comment.id} has delta flag set.")
            if comment.parent_id.startswith("t1_"):
                parent_id = comment.parent_id[3:]
                parent_comment = all_comments_dict.get(parent_id)
                if parent_comment:
                    parent_comment.actually_has_delta = True
                    delta_marked_count += 1
                else:
                    parent_not_found_count += 1

    print(f"Total comments marked with actually_has_delta: {delta_marked_count}")
    print(f"Total parent comments not found: {parent_not_found_count}")

def update_csv_file(filepath):
    """
    Processes the CSV file to add new columns (is_tlc, tlc_id, is_tlc_author, children_count)
    and overwrites the original CSV with the updated data.
    """
    all_comments = process_chunk(filepath)
    update_parent_delta_flags(all_comments)
    
    # Create a new list of dictionaries from all_comments with the desired fields.
    updated_rows = []
    for comment in all_comments.values():
        # Start with the original row data
        row = comment.original.copy()
        
        # add new columns
        row['is_tlc'] = comment.is_tlc
        row['tlc_id'] = comment.tlc_id
        row['is_tlc_author'] = comment.is_tlc_author
        row['children_count'] = comment.children_count
        row['actually_has_delta'] = comment.actually_has_delta  
        row['op_name'] = comment.op_name
        updated_rows.append(row)
    
    if updated_rows:
        # Get the fieldnames from the keys of the first row (preserving original order if possible)
        fieldnames = list(updated_rows[0].keys())
    else:
        fieldnames = []
    
    # Overwrite the original CSV file with updated data.
    with open(filepath, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
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