import zstandard as zstd
import json
import pandas as pd
import os
from datetime import datetime, timezone
from collections import defaultdict
import time

CHUNK_SIZE = 10_000  # adjust based on memory usage
OUTPUT_DIR = os.path.join('..', 'data', 'monthly_chunks')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(zst_path):
    print("Starting processing of file:", zst_path)
    processed_months = set()  # For limiting processing to the first three months
    with open(zst_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)

        buffer = b""
        chunk_count = 0
        decoder = json.JSONDecoder()

        monthly_data = defaultdict(list)

        start_date = datetime(2017, 1, 10, tzinfo=timezone.utc) #the earliest full date for which we have deltas from #deltalog
        end_date = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc) #the latest full date for which we have deltas from #deltalog
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        while True:
            chunk = stream_reader.read(2**20)
            if not chunk:
                break
            buffer += chunk
            chunk_count += 1
            print("Read chunk #", chunk_count, "of size:", len(chunk), "bytes")

            while True:
                try:
                    data_str = buffer.decode('utf-8')
                    obj, index = decoder.raw_decode(data_str)
                    remaining_str = data_str[index:].lstrip()
                    buffer = remaining_str.encode('utf-8')

                    # Get the timestamp value
                    created_at_val = obj.get('CreatedAt') or obj.get('created_utc')
                    if created_at_val is None:
                        continue

                    # Process based on type
                    if isinstance(created_at_val, int):
                        ts = created_at_val
                        if ts < start_ts or ts > end_ts:
                            continue
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    elif isinstance(created_at_val, str):
                        if created_at_val.isdigit():
                            ts = int(created_at_val)
                            if ts < start_ts or ts > end_ts:
                                continue
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        else:
                            try:
                                dt = datetime.strptime(created_at_val, "%Y-%m-%dT%H:%M:%S")
                                dt = dt.replace(tzinfo=timezone.utc)
                            except:
                                try:
                                    dt = datetime.fromtimestamp(int(created_at_val), tz=timezone.utc)
                                except:
                                    continue
                            ts = int(dt.timestamp())
                            if ts < start_ts or ts > end_ts:
                                continue
                    else:
                        continue

                    month_key = dt.strftime("%Y_%m")
                    # --- START OF LIMITING CODE: Limit processing to first three months ---
                    if month_key not in processed_months:
                        if len(processed_months) >= 3:
                            # Dump remaining data for the processed months and exit the function
                            for m in processed_months:
                                if monthly_data[m]:
                                    dump_month_to_csv(monthly_data, m, final=True)
                                    print("Dumping final data for month:", m, "with", len(monthly_data[m]), "records")
                            return
                        processed_months.add(month_key)
                        print("Started processing new month:", month_key)
                    # --- END OF LIMITING CODE ---

                    monthly_data[month_key].append(obj)

                    if len(monthly_data[month_key]) >= CHUNK_SIZE:
                        dump_month_to_csv(monthly_data, month_key)

                except json.JSONDecodeError:
                    break
            time.sleep(0.1)

        # Final dump of remaining data
        for month_key in monthly_data:
            if monthly_data[month_key]:
                dump_month_to_csv(monthly_data, month_key, final=True)
        print("Finished processing file.")

def dump_month_to_csv(monthly_data, month_key, final=False):
    print("Dumping data for month:", month_key, "- final dump:", final, "with", len(monthly_data[month_key]), "records")
    df = pd.DataFrame(monthly_data[month_key])
    out_file = os.path.join(OUTPUT_DIR, f"comments_{month_key}.csv")
    
    if os.path.exists(out_file) and not final:
        df.to_csv(out_file, mode='a', header=False, index=False)
    else:
        df.to_csv(out_file, index=False)

    # Clear memory
    monthly_data[month_key] = []

# Automatically process the ZST file when running this script as part of a make step
if __name__ == "__main__":
    zst_file = os.path.join("..", "data", "changemyview_comments.zst")
    process_file(zst_file)