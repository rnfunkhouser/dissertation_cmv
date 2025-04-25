# ---  Imports  ---------------------------------------------------------------
import pandas as pd
from tqdm import tqdm        # avoids the ipywidgets warning outside Jupyter
import warnings
from tqdm import TqdmWarning
from collections import defaultdict
import os
import math
# Silence the harmless "IProgress not found" warning that appears when
# transformers (via tqdm.auto) is used outside Jupyter.
warnings.filterwarnings(
    "ignore",
    message="IProgress not found.*",
    category=TqdmWarning,
)
# Silence the long-sequence tokenizer warning from transformers
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum sequence length.*",
    category=UserWarning,
)
# Shut up the Hugging Face transformers logger about sequence length
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# ---  File locations  --------------------------------------------------------
CSV_IN  = "../data/cleaned_final_convos.csv"         # ← your source file
CSV_OUT = "../data/pretrained_classified.csv"  # ← where to save results
TEXTCOL = "body"                                   # ← column that has the text
MODEL   = "mariaantoniak/storyseeker"              # pretrained story classifier

# --- Load data --------------------------------------------------------------

# --- Build classification pipeline -----------------------------------------
chunk_size = 16  # number of comment windows per forward pass
story_clf = pipeline(
    task="text-classification",
    model=AutoModelForSequenceClassification.from_pretrained(MODEL),
    tokenizer=AutoTokenizer.from_pretrained(MODEL),
    truncation=True,
    padding=True,
    max_length=512,
    batch_size=chunk_size,      # uses the chunk_size defined just below
    device_map="auto"           # picks GPU automatically if available
)

# --- Helper: classify a list of texts ------------------------------------
def classify_chunk(texts: list[str], show_progress: bool = False) -> list[str]:
    """
    Classify a list of comment strings and return a prediction label
    for each original comment, handling long comments with an
    overlapping sliding window.
    """
    MAX_LEN  = 512
    STRIDE   = 256
    SPECIALS = story_clf.tokenizer.num_special_tokens_to_add(pair=False)
    BODY_MAX = MAX_LEN - SPECIALS

    texts_expanded: list[str] = []
    orig_indices:   list[int] = []

    for idx, txt in enumerate(texts):
        input_ids = story_clf.tokenizer(
            txt,
            add_special_tokens=False
        )["input_ids"]

        if len(input_ids) <= MAX_LEN:
            texts_expanded.append(txt)
            orig_indices.append(idx)
        else:
            for start in range(0, len(input_ids), STRIDE):
                chunk_ids = input_ids[start:start + BODY_MAX]
                if not chunk_ids:
                    break
                chunk_txt = story_clf.tokenizer.decode(chunk_ids)
                texts_expanded.append(chunk_txt)
                orig_indices.append(idx)
                if start + MAX_LEN >= len(input_ids):
                    break

    label_scores = defaultdict(lambda: defaultdict(float))

    iterator = tqdm(
        range(0, len(texts_expanded), chunk_size),
        desc="windows",
        leave=False
    ) if show_progress else range(0, len(texts_expanded), chunk_size)

    for start in iterator:
        end   = start + chunk_size
        batch = texts_expanded[start:end]
        batch_preds = story_clf(batch)
        for offset, pred in enumerate(batch_preds):
            global_idx = start + offset
            orig_idx   = orig_indices[global_idx]
            label      = pred["label"]
            score      = pred["score"]
            label_scores[orig_idx][label] += score

    pred_labels = []
    for idx in range(len(texts)):
        scores = label_scores.get(idx, {})
        if scores:
            pred_labels.append(max(scores.items(), key=lambda kv: kv[1])[0])
        else:
            pred_labels.append("UNKNOWN")
    return pred_labels

# --- Stream through the CSV file ----------------------------------------
CHUNK_ROWS = 2500  # adjust based on available RAM/VRAM

# Remove existing output file (if any) so we start fresh
if os.path.exists(CSV_OUT):
    os.remove(CSV_OUT)

# --- Progress setup -----------------------------------------------------
try:
    with open(CSV_IN, "r", encoding="utf-8") as f:
        total_rows   = sum(1 for _ in f) - 1  # minus header
    total_chunks = math.ceil(total_rows / CHUNK_ROWS)
except Exception:
    total_chunks = None

reader = pd.read_csv(CSV_IN, chunksize=CHUNK_ROWS)

for chunk_idx, df_chunk in enumerate(
    tqdm(reader, total=total_chunks, desc="CSV chunks")
):
    texts = df_chunk[TEXTCOL].astype(str).tolist()
    pred_labels = classify_chunk(texts, show_progress=True)
    df_chunk["story_pred"] = pred_labels

    mode   = "w" if chunk_idx == 0 else "a"
    header =  chunk_idx == 0
    df_chunk.to_csv(CSV_OUT, mode=mode, header=header,
                    index=False, encoding="utf-8")
    print(f"✅  Appended {len(df_chunk):,} rows (chunk {chunk_idx+1})")