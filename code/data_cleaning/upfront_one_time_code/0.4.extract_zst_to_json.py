#!/usr/bin/env python3
import zstandard as zstd
import json
import io


def detect_json_structure(sample_text):
    """Detect the JSON structure from a sample text.

    Returns:
        'array' if the data appears to be a JSON array,
        'ndjson' if it is newline-delimited JSON objects,
        'concatenated' if it is a stream of concatenated JSON objects,
        None if undetermined.
    """
    stripped = sample_text.lstrip()
    if not stripped:
        return None
    if stripped[0] == '[':
        return 'array'
    elif stripped[0] == '{':
        # If there is at least one newline, assume newline-delimited (ndjson).
        if '\n' in sample_text:
            return 'ndjson'
        else:
            return 'concatenated'
    else:
        return None


def convert_zst_to_json(zst_file, output_file):
    dctx = zstd.ZstdDecompressor()
    chunk_size = 2**20  # 1MB chunks
    sample_bytes = b""
    max_sample_size = 1024 * 1024  # 1MB sample for structure detection

    # First, sample the decompressed data to detect the JSON structure
    with open(zst_file, 'rb') as fin:
        with dctx.stream_reader(fin) as reader:
            while len(sample_bytes) < max_sample_size:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                sample_bytes += chunk

    try:
        sample_text = sample_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        print("Error decoding sample:", e)
        return

    structure = detect_json_structure(sample_text)
    print("Detected JSON structure:", structure)

    # Re-open the file for full processing
    with open(zst_file, 'rb') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        with dctx.stream_reader(fin) as reader:
            object_count = 0
            if structure != 'ndjson':
                print(f"Error: Expected ndjson structure, but detected '{structure}'.")
                return
            anomaly_count = 0
            decoder = json.JSONDecoder()
            buffer = ""
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                try:
                    text = chunk.decode('utf-8')
                except UnicodeDecodeError:
                    text = chunk.decode('utf-8', errors='replace')
                buffer += text
                # Process complete lines
                lines = buffer.splitlines(keepends=False)
                for line in lines[:-1]:
                    line = line.strip()
                    if line:
                        try:
                            # Try to decode the line as a single JSON object
                            obj = json.loads(line)
                            fout.write(json.dumps(obj) + "\n")
                            object_count += 1
                            if object_count % 10000 == 0:
                                print(f"Exported {object_count} JSON objects so far.")
                        except json.JSONDecodeError:
                            # If it fails, attempt to extract multiple JSON objects from the same line
                            count = 0
                            idx = 0
                            while idx < len(line):
                                try:
                                    obj, offset = decoder.raw_decode(line[idx:])
                                    fout.write(json.dumps(obj) + "\n")
                                    object_count += 1
                                    count += 1
                                    idx += offset
                                    if object_count % 10000 == 0:
                                        print(f"Exported {object_count} JSON objects so far.")
                                except json.JSONDecodeError:
                                    break
                            print(f"Extracted {count} JSON objects from one line.")
                            if count > 1:
                                anomaly_count += 1
                buffer = lines[-1]
            # Process any remaining data in the buffer
            buffer = buffer.strip()
            if buffer:
                try:
                    obj = json.loads(buffer)
                    fout.write(json.dumps(obj) + "\n")
                    object_count += 1
                    if object_count % 10000 == 0:
                        print(f"Exported {object_count} JSON objects so far.")
                except json.JSONDecodeError:
                    pass
            print("Validation Summary:")
            print(f"Total JSON objects processed: {object_count}")
            print(f"Total lines with multiple JSON objects: {anomaly_count}")
            if object_count == 0:
                print("WARNING: No JSON objects were processed. Please check your data and extraction logic.")
    print("Finished processing.")


if __name__ == "__main__":
    zst_file = "../data/changemyview_comments.zst"
    output_file = "../data/changemyview_comments.json"
    convert_zst_to_json(zst_file, output_file)