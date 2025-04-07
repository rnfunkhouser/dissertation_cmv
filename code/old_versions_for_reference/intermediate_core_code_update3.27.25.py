import pandas as pd
import csv
#define the class object
class RedditComment:
    def __init__(self, row):
        # Existing code
        self.id = row['id']
        self.author = row['author']
        self.parent_id = row['parent_id']
        self.body = row.get('body', '').strip()
        self.CreatedAt = row.get('CreatedAt', '')
        self.is_submitter = str(row['is_submitter']).strip().lower() in ["true"]
        self.has_delta = str(row['has_delta_from_OP']).strip().lower() in ["true"]
        # Initialize other attributes
        self.children = []
        self.tlc_author = False
        self.delta_in_tree = False
        self.received_delta = False  # For TLC comments that received a delta
        self.awarded_delta = False   # For OP comments that awarded a delta
        self.parent_author = None
        self.contains_delta_award = False
        self.delta_path = []
    def find_parent_author(self, all_comments_dict):
        """Find and set the author of this comment's parent."""
        # If the parent author is already known, return it
        if self.parent_author is not None:
            return self.parent_author

        # If the parent is a submission (not a comment)
        if self.parent_id.startswith("t3_"):
            # We can't determine the author of the submission from our comment data
            self.parent_author = None
        # Otherwise, the parent is another comment
        else:
            # Extract the comment ID from the parent_id (removing the "t1_" prefix)
            parent_comment_id = self.parent_id[3:]
            
            # Look up the parent comment in our dictionary
            parent_comment = all_comments_dict.get(parent_comment_id)
            
            # If we found the parent, get its author
            if parent_comment:
                self.parent_author = parent_comment.author
        
        return self.parent_author
    def collect_TLC_comments_to_delta(self):
        """
        Collect all comments by the TLC author from this comment down to the one that received a delta.
        Returns a list of comment bodies if a path is found, otherwise an empty list.
        """
        # If this isn't a TLC author comment, we're on the wrong path
        if not self.tlc_author:
            return []
        
        # Start with the current comment
        tlc_comments = [self.body]
        
        # If this comment received a delta, we're done - return just this comment
        if self.received_delta:
            return tlc_comments
        
        # Otherwise, try to find a path through the children
        for child in self.children:
            # Recursively collect comments from this child
            child_path = child.collect_TLC_comments_to_delta()
            
            # If child path exists (not empty), we found a path to a delta
            if child_path:
                # Add the child's path to our result and return
                return tlc_comments + child_path
        
        # No path found through any children
        return []
    def find_OP_awarding_delta_to_TLC(self):
        """
        Recursively search the comment tree to find where OP awarded a delta to the TLC author.
        
        Returns True if this comment or any of its descendants contain a delta award interaction.
        This helps with building the path to the delta.
        """
        self.contains_delta_award = False
        
        # Check if this comment is by the TLC author and has received a delta
        if self.tlc_author and self.has_delta:
            # Look for an OP child that might have awarded the delta
            for child in self.children:
                if child.is_submitter:
                    # We found the interaction! Mark both comments accordingly
                    self.received_delta = True
                    child.awarded_delta = True
                    self.contains_delta_award = True
                    break  # Found the interaction, no need to check other children
        
        # Recursively search all children
        for child in self.children:
            # If any child contains a delta award, this subtree contains a delta award
            if child.find_OP_awarding_delta_to_TLC():
                self.contains_delta_award = True
        
        # Return the value so parent calls can update their status accordingly
        return self.contains_delta_award
    def find_delta_in_tree(self):
        """
        Determines if this comment or any of its descendants contains a delta award
        specifically between the TLC author and OP.
        """
        self.delta_in_tree = self.contains_delta_award
        
        # Also check immediate children
        if not self.delta_in_tree:
            for child in self.children:
                if child.contains_delta_award:
                    self.delta_in_tree = True
                    break
        
        return self.delta_in_tree

    def total_children(self):
        count_children = 0
        for child in self.children:
            count_children += 1
            count_children += child.total_children()
        return count_children
    
    def mark_tlc_author_comments(self):
        # Recursively mark if a comment is by the TLC author
        if self.parent_id.startswith("t3_"):
            self.tlc_author = True
        for child in self.children:
            if child.author == self.author and self.tlc_author:
                child.tlc_author = True
            child.mark_tlc_author_comments()
    

if __name__ == "__main__":
    filepath = '/Users/ryanfunkhouser/Documents/Research/backup_cmv_computational_small_stories/data/CMV_purged_columns.csv'
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # Initialize a dictionary to map id to RedditComment objects
        all_comments_dict = {}
        # Initialize a list for top-level comments
        top_level_comments = []
        # Process each row from the CSV
        for comment in reader:
            reddit_comment = RedditComment(comment)
            all_comments_dict[comment['id']] = reddit_comment
            # Check if the comment is a top-level comment
            if comment['parent_id'].startswith("t3_"):
                top_level_comments.append(reddit_comment)
            else:
                # This assumes that the parent_id field is prefixed with "t1_" when the parent is a comment
                parent_comment_id = comment['parent_id'][3:]
                parent_comment = all_comments_dict.get(parent_comment_id)
                if parent_comment:
                    parent_comment.children.append(reddit_comment)
        for comment in top_level_comments:
            comment.mark_tlc_author_comments()
            comment.find_OP_awarding_delta_to_TLC()
            comment.find_delta_in_tree()
     

        for comment in all_comments_dict.values():
            comment.find_parent_author(all_comments_dict)    

        print(f"Total top-level comments/trees: {len(top_level_comments)}")
        print(f"Total comments: {len(all_comments_dict)}")
        print(f"Total comments with children: {sum(comment.total_children() for comment in top_level_comments)}")
# Step 1: Initialize the qualified conversations dictionary
qualified_conversations = {}

# Step 2: Filter the top-level comments to find "qualified conversations"
# When building qualified_conversations
for comment in top_level_comments:
    if comment.delta_in_tree and comment.contains_delta_award:
        # Instead of collect_tlc_author_comments, use:
        tlc_comments_to_delta = comment.collect_TLC_comments_to_delta()
        
        # If we found a path to the delta
        if tlc_comments_to_delta:
            combined_text = "\n---\n".join(tlc_comments_to_delta)
            
            # Store in qualified_conversations
            qualified_conversations[comment.id] = {
                # Existing fields
                "id": comment.id,
                "author": comment.author,
                "is_submitter": comment.is_submitter,
                "has_delta": comment.has_delta,
                "only_OP_and_TLC": comment.only_op_and_tlc,
                "body": combined_text,  # Now contains only TLC comments up to delta
                "parent_id": comment.parent_id,
                "op_replied": comment.op_replied,
                "total_children": comment.total_children(),
                "CreatedAt": comment.CreatedAt,
                # You could add new fields
                "delta_path_length": len(tlc_comments_to_delta)
            }
# Step 5: Convert to a pandas DataFrame and save as CSV
df = pd.DataFrame.from_dict(qualified_conversations, orient="index")

# Save the DataFrame as a CSV
output_csv_path = "../data/qualified_conversations.csv"
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"✅ Output saved to {output_csv_path}")
print(len(qualified_conversations), "number of qualified conversations")