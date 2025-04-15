import json
import pandas as pd
import os
from datetime import datetime, timezone

CHUNK_SIZE = 100_000  # adjust based on memory usage
OUTPUT_DIR = os.path.join('..', 'data', 'monthly_chunks')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_next_month_key(month_key):
    # month_key is in the format "YYYY_MM"
    year, month = map(int, month_key.split("_"))
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    return f"{next_year}_{next_month:02d}"


def dump_chunk_to_csv(month_key, chunk):
    if not chunk:
        return
    df = pd.DataFrame(chunk)
    desired_columns = ["id", "parent_id", "body", "created_utc", "author", "link_id", "edited", "score", "subreddit_id"]
    df = df.reindex(columns=desired_columns)
    out_file = os.path.join(OUTPUT_DIR, f"comments_{month_key}.csv")
    print(f"Saving {len(df)} records to {out_file}")
    if os.path.exists(out_file):
        df.to_csv(out_file, mode='a', header=False, index=False)
    else:
        df.to_csv(out_file, index=False)


def process_file(json_path):
    print("Starting processing of file:", json_path)
    current_month = None
    current_month_data = []
    current_month_post_ids = set()
    spillover_count = 0

    # DEBUG MODE: Cap processing to first 5 distinct months for debugging purposes
    DEBUG_MODE = True  # Set to False to process all months
    DEBUG_MONTH_LIMIT = 5
    processed_months = set()

    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                comment = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Ensure the comment has a link_id
            if not comment.get('link_id'):
                continue

            # Use only the created_utc field
            created_utc_val = comment.get('created_utc')
            if created_utc_val is None:
                continue

            if isinstance(created_utc_val, int):
                ts = created_utc_val
            elif isinstance(created_utc_val, str) and created_utc_val.isdigit():
                ts = int(created_utc_val)
            else:
                continue

            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            month_key = dt.strftime("%Y_%m")

            if current_month is None:
                current_month = month_key

            # DEBUG MODE: Cap processing to first DEBUG_MONTH_LIMIT distinct months
            if DEBUG_MODE and month_key not in processed_months and len(processed_months) == DEBUG_MONTH_LIMIT:
                print(f"DEBUG MODE: Reached processing cap of {DEBUG_MONTH_LIMIT} months. Stopping further processing.")
                break
            processed_months.add(month_key)

            # Process comment based on its month
            if month_key == current_month:
                current_month_data.append(comment)
                current_month_post_ids.add(comment['link_id'])
            else:
                expected_next_month = get_next_month_key(current_month)
                if month_key == expected_next_month:
                    # Next month's comment: add it if it belongs to a post from the current month
                    if comment['link_id'] in current_month_post_ids:
                        current_month_data.append(comment)
                        spillover_count += 1
                    else:
                        # Dump current month's data and start a new month
                        dump_chunk_to_csv(current_month, current_month_data)
                        current_month = month_key
                        current_month_data = [comment]
                        current_month_post_ids = {comment['link_id']}
                else:
                    # If the comment's month is beyond the immediate next month, dump current month's data
                    dump_chunk_to_csv(current_month, current_month_data)
                    current_month = month_key
                    current_month_data = [comment]
                    current_month_post_ids = {comment['link_id']}

            # If the current month's data reaches the CHUNK_SIZE, dump it and continue
            if len(current_month_data) >= CHUNK_SIZE:
                dump_chunk_to_csv(current_month, current_month_data)
                current_month_data = []  # Preserve current_month_post_ids for spillover tracking

    # Dump any remaining data after processing
    if current_month_data:
        dump_chunk_to_csv(current_month, current_month_data)
    print("Finished processing file. Spillover comments added:", spillover_count)


if __name__ == "__main__":
    json_file = os.path.join("..", "data", "changemyview_comments.json")
    process_file(json_file)