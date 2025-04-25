# --- Imports ---------------------------------------------------------------
import pandas as pd
from tqdm import tqdm
import warnings
from tqdm import TqdmWarning

# Silence the harmless “IProgress not found” warning outside Jupyter
warnings.filterwarnings(
    "ignore",
    message="IProgress not found.*",
    category=TqdmWarning,
)

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- File locations --------------------------------------------------------
CSV_IN  = "../data/cleaned_final_convos.csv"
CSV_OUT = "../data/pretrained_classified.csv"
TEXTCOL = "body"
MODEL   = "mariaantoniak/storyseeker"

# --- Read everything into memory ------------------------------------------
df    = pd.read_csv(CSV_IN)
texts = df[TEXTCOL].astype(str).tolist()

# --- Build the classifier pipeline ---------------------------------------
chunk_size = 32   # number of comments per forward pass
story_clf  = pipeline(
    task="text-classification",
    model=AutoModelForSequenceClassification.from_pretrained(MODEL),
    tokenizer=AutoTokenizer.from_pretrained(MODEL),
    truncation=True,
    padding=True,
    max_length=512,
    batch_size=chunk_size,
    device_map="auto"
)

# --- Classify with simple truncation --------------------------------------
pred_labels = []

for start in tqdm(
    range(0, len(texts), chunk_size),
    desc="Classifying",
    unit_scale=chunk_size,
    unit="comments"
):
    end        = start + chunk_size
    batch      = texts[start:end]
    batch_preds = story_clf(batch)              # each comment is truncated at 512
    # collect just the label from each output dict
    pred_labels.extend(p["label"] for p in batch_preds)

# --- Attach results & write out -------------------------------------------
df["story_pred"] = pred_labels
df.to_csv(CSV_OUT, index=False, encoding="utf-8")

print(f"✅  Saved {len(df):,} rows with predictions to: {CSV_OUT}")