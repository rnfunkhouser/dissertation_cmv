import pandas as pd
from collections import Counter


def detect_abnormal_values(csv_path, columns_to_check, drop_abnormal=False, output_path=None):
    # Load the CSV without enforcing dtype to see what pandas infers
    df = pd.read_csv(csv_path, low_memory=False)

    total_abnormal = 0
    
    if "op_name" in df.columns:
        df = df.drop(columns=["op_name"])

    for col in ["small_story", "hypothetical", "personal", "centrality_of_story"]:
        df[col] = pd.NA

    column_names = df.columns.tolist()
    for col_index in columns_to_check:
        if col_index >= len(column_names):
            print(f"Column index {col_index} is out of range.")
            continue
        col = column_names[col_index]

        # Get types of each value
        types = df[col].map(lambda x: type(x).__name__)
        common_type = types.value_counts().idxmax()

        # Find rows that do NOT match the common type
        abnormal_rows = df[types != common_type]

        print(f"\nColumn: {col} (index {col_index})")
        print(f"Most common type: {common_type}")
        print(f"Abnormal entries: {len(abnormal_rows)}")
        print(abnormal_rows[[col]].head())

        total_abnormal += len(abnormal_rows)

        if drop_abnormal and output_path:
            df = df[types == common_type]

    if drop_abnormal and output_path:
        print(f"\nTotal abnormal rows detected (and removed if enabled): {total_abnormal}")
        df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to {output_path}")


if __name__ == "__main__":
    csv_path = "/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/final_qualified_convos.csv"
    columns_to_check = [3, 10, 11, 13, 15]  
    detect_abnormal_values(csv_path, columns_to_check, drop_abnormal=True, output_path="../data/cleaned_final_convos.csv")