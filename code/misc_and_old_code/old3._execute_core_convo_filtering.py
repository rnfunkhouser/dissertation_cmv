import pandas as pd
import csv

#define the class object
class RedditComment:
    def __init__(self, row):
        # Directly pull data from the row dictionary using CSV column names
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
        self.op_replied = False
        self.only_op_and_tlc = False
        
    def total_children(self):
        count_children = 0
        for child in self.children:
            count_children += 1
            count_children += child.total_children()
        return count_children
    
    def update_op_replied(self):
        # Check if any child has is_submitter set to True
        if any(child.is_submitter for child in self.children):
            self.op_replied = True

    def mark_tlc_author_comments(self):
        # Recursively mark if a comment is by the TLC author
        if self.parent_id.startswith("t3_"):
            self.tlc_author = True
        for child in self.children:
            if child.author == self.author and self.tlc_author:
                child.tlc_author = True
            child.mark_tlc_author_comments()
    
    def delta_in_tree(self):
        # Start with current delta status
        has_delta = self.has_delta
        
        # Recursively check children
        for child in self.children:
            child_has_delta = child.delta_in_tree()
            if child_has_delta:
                has_delta = True
        
        # Update our own delta status
        self.has_delta = has_delta
        return has_delta
    def check_if_only_OP_and_TLC_commented(self):
        roles = set()
        # Determine the role for the current comment
        if self.is_submitter:
            roles.add("op")
        elif self.tlc_author:
            roles.add("tlc")
        else:
            roles.add("other")
        # Recursively process each child and add their roles
        for child in self.children:
            child_roles = child.check_if_only_OP_and_TLC_commented()
            roles.update(child_roles)
        # Update the attribute: it's True only if both OP and TLC are present
        self.only_op_and_tlc = (roles == {"op", "tlc"})
        return roles

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
        delta_count = 0
        eligible_trees = 0
        for comment in top_level_comments:
            comment.mark_tlc_author_comments()
            comment.update_op_replied()
            comment.delta_in_tree()
            comment.check_if_only_OP_and_TLC_commented()
            delta_count += comment.has_delta
            eligible_trees += comment.has_delta and comment.only_op_and_tlc
        print(f"Total trees with deltas: {delta_count}")
        print(f"Total top-level comments/trees: {len(top_level_comments)}")
        print(f"Total trees where only OP and TLC commented: {sum(comment.only_op_and_tlc for comment in top_level_comments)}")
        print(f"Total trees where OP replied: {sum(comment.op_replied for comment in top_level_comments)}")
        print(f"Total comments: {len(all_comments_dict)}")
        print(f"Total comments with children: {sum(comment.total_children() for comment in top_level_comments)}")
        print(f"Total eligible trees: {eligible_trees}")
# Step 1: Initialize the qualified conversations dictionary
qualified_conversations = {}
# Step 2: Filter the top-level comments to find "qualified conversations"
for comment in top_level_comments:
    if len(comment.children)==0:
        continue
    if comment.delta_in_tree() and comment.only_op_and_tlc:
        # Step 3: Combine all TLC author's comments within the tree
        tlc_author_name = comment.author
        combined_body = []

        def collect_tlc_author_comments(node):
            """ Recursively collect all comments by the TLC author in this tree. """
            if node.author == tlc_author_name:
                combined_body.append(node.body)
            for child in node.children:
                collect_tlc_author_comments(child)

        # Start the recursive collection
        collect_tlc_author_comments(comment)

        # Join the collected comments with the requested format
        combined_text = "\n---\n".join(combined_body)

        # Step 4: Store only the original TLC's metadata
        qualified_conversations[comment.id] = {
            "id": comment.id,
            "author": comment.author,
            "is_submitter": comment.is_submitter,
            "has_delta": comment.has_delta,
            "only_OP_and_TLC": comment.only_op_and_tlc,
            "body": combined_text,  # Replace the original body with the combined TLC text
            "parent_id": comment.parent_id,
            "op_replied": comment.op_replied,
            "total_children": comment.total_children(),
            "CreatedAt": comment.CreatedAt
        }

# Step 5: Convert to a pandas DataFrame and save as CSV
df = pd.DataFrame.from_dict(qualified_conversations, orient="index")

# Save the DataFrame as a CSV
output_csv_path = "../data/qualified_conversations.csv"
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"✅ Output saved to {output_csv_path}")
print(len(qualified_conversations), "number of qualified conversations")