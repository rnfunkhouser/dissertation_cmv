import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def process_ndjson(file_path):
    monthly_counts = defaultdict(int)

    with open(file_path, 'r') as f:
        for line in f:
            try:
                comment = json.loads(line)
                if 'created_utc' in comment:
                    timestamp = int(comment['created_utc'])
                    dt = datetime.utcfromtimestamp(timestamp)
                    month_str = dt.strftime('%Y-%m')
                    monthly_counts[month_str] += 1
            except json.JSONDecodeError:
                continue  # skip malformed lines

    return monthly_counts

def plot_monthly_counts(monthly_counts):
    # Convert to pandas Series for easy plotting and sorting
    series = pd.Series(monthly_counts)
    series = series.sort_index()

    plt.figure(figsize=(16, 6))
    plt.plot(series.index, series.values, marker='o')
    plt.xticks(rotation=45, fontsize=8)
    plt.title('Monthly Comment Counts')
    plt.xlabel('Month')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/ryanfunkhouser/Documents/Research/official_cmv_computational_small_stories/data/filtered_cmv_comments.json"  # Change this to your actual NDJSON file
    counts = process_ndjson(file_path)
    plot_monthly_counts(counts)