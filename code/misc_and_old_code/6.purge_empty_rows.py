import pandas as pd

def purge_invalid_rows(csv_path, output_path):
    # Load the CSV
    columns_to_check = ["has_delta_from_OP", "is_tlc", "is_tlc_author", "is_submitter"]
    df = pd.read_csv(csv_path, dtype={col: str for col in columns_to_check})

    # Function to check if a value is a proper boolean (True or False)
    def is_valid_boolean(val):
        if pd.isna(val):
            return False
        return str(val).strip().lower() in ['true', 'false']

    # Filter the DataFrame
    valid_rows = df[columns_to_check].apply(lambda col: col.map(is_valid_boolean)).all(axis=1)
    cleaned_df = df[valid_rows]

    if "op_name" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["op_name"])

    for col in ["small_story", "hypothetical", "personal", "centrality_of_story"]:
        cleaned_df[col] = pd.NA

    num_removed = len(df) - len(cleaned_df)
    print(f"Removed {num_removed} invalid rows.")

    # Save the cleaned CSV
    cleaned_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage
    input_csv = "/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/clean_qualified_conversations_4.csv"
    output_csv = "../data/final_qualified_convos.csv"
    purge_invalid_rows(input_csv, output_csv)
