"""
Smoke‑test run: fine‑tune a multi‑head RoBERTa model on a 100‑row
manually‑labelled CSV to verify the end‑to‑end pipeline on a laptop.
Original/full‑run settings are commented out below each override.
"""

import pathlib   # needed for debug print path
import os, random, math, json, tempfile, torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import transformers



from transformers import (
    RobertaModel, RobertaTokenizerFast, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ── 1. CONFIG ------------------------------------------------------------
# TRAIN_CSV      = Path("../data/2500_manually_labeled_training.csv")   # original full set
TRAIN_CSV      = Path("../data/100_row_smoke_test.csv")       # smoke‑test 100 rows
FULL_DATA_CSV  = Path("../data/cleaned_final_convos.csv")  # 209 k rows
# OUTPUT_DIR     = Path("../models/story_multitask")             # original
OUTPUT_DIR     = Path("../models/debug_run")                     # smoke‑test
SEED           = 42
BATCH_SIZE     = 4                 
LR             = 2e-5               # fine-tuning lr
EPOCHS         = 1
# EARLY_PATIENCE = 2
EARLY_PATIENCE = None    # disabled for smoke test

# loss-balance λ
LAMBDA_CENT = 3.0
LAMBDA_HYP  = 3.0
LAMBDA_PER  = 3.0

random.seed(SEED); torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Tokeniser & Encoder (pretrained on Reddit stories) ──────────────────
# MODEL_NAME = "mariaantoniak/storyseeker"   # original
MODEL_NAME = "roberta-base"                  # smoke‑test
tokenizer  = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
encoder    = RobertaModel.from_pretrained(MODEL_NAME)

# ── 3. Dataset class with optional sliding window (>512 tokens) ────────────
MAX_LEN = 508    # slightly below 512 to allow for CLS/SEP tokens that were triggering errors 
WINDOW_STRIDE = 256          # overlapping stride for long comments

LABEL_COLUMNS = ["small_story", "centrality", "hypothetical", "personal"]

def make_windows(encodings, max_len=512, stride=256):
    """Convert a tokenised sequence into overlapping 512-token windows."""
    input_ids, attention = encodings["input_ids"], encodings["attention_mask"]
    for start in range(0, len(input_ids), stride):
        end = start + max_len
        chunk_ids = input_ids[start:end]
        if len(chunk_ids) < max_len:
            continue   # skip shorter final window since loss should be minimal in most cases
        yield {
            "input_ids": torch.tensor(chunk_ids),
            "attention_mask": torch.tensor(attention[start:end]),
        }

class StoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, for_inference: bool = False):
        """
        If ``for_inference`` is True the dataset also returns the original
        dataframe index (``row_id``) so that window‑level predictions can be
        aggregated back to the comment after inference.
        """
        self.for_inference = for_inference
        self.rows = []
        for row_idx, row in df.iterrows():
            text   = str(row["body"])
            labels = {k: row[k] for k in LABEL_COLUMNS} if not for_inference else None
            enc    = tokenizer(text, truncation=False)
            # If <=512 tokens, single chunk; else sliding windows
            if len(enc["input_ids"]) <= MAX_LEN:
                self.rows.append((tokenizer(text,
                                            max_length=MAX_LEN,
                                            truncation=True,
                                            padding="max_length",
                                            return_tensors="pt"),
                                  labels,
                                  row_idx))
            else:
                for window in make_windows(enc, max_len=MAX_LEN, stride=WINDOW_STRIDE):
                    padded = tokenizer.pad(window,
                                           max_length=MAX_LEN,
                                           padding="max_length",
                                           return_tensors="pt")
                    self.rows.append((padded, labels, row_idx))

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        enc, labels, row_id = self.rows[idx]
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if labels:
            # ensure torch tensors & dtypes
            item["labels"] = torch.tensor([
                labels["small_story"],            # y0  {0/1}
                labels["centrality"],             # y1  {0,1,2}
                labels["hypothetical"],           # y2  {0/1}
                labels["personal"]                # y3  {0/1}
            ], dtype=torch.long)
        if self.for_inference:
            item["row_id"] = row_id          # keep as plain int (no CUDA)
        return item

# ── 4. Multi-head model definition ─────────────────────────────────────────
class MultiHeadStoryModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.story_head = nn.Linear(hidden, 1)
        self.cent_head  = nn.Linear(hidden, 3)
        self.hyp_head   = nn.Linear(hidden, 1)
        self.per_head   = nn.Linear(hidden, 1)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]       # CLS vector
        logits_story = self.story_head(cls).squeeze(-1)
        logits_cent  = self.cent_head(cls)
        logits_hyp   = self.hyp_head(cls).squeeze(-1)
        logits_per   = self.per_head(cls).squeeze(-1)

        if labels is None:
            return {
                "logits_story": logits_story,
                "logits_cent": logits_cent,
                "logits_hyp": logits_hyp,
                "logits_per": logits_per,
            }

        y_story, y_cent, y_hyp, y_per = (
            labels[:, 0].float(),
            labels[:, 1],
            labels[:, 2].float(),
            labels[:, 3].float()
        )

        bce = nn.BCEWithLogitsLoss()
        ce  = nn.CrossEntropyLoss()

        # story loss (always)
        loss_story = bce(logits_story, y_story)

        # mask for rows that contain a story (>=0.5 label)
        mask = (y_story > 0.5).float()

        # centrality (only where story==1)
        if mask.sum() > 0:
            loss_cent = ce(logits_cent[mask.bool()], y_cent[mask.bool()])
        else:
            loss_cent = torch.tensor(0.0, device=loss_story.device)

        # hypo / personal with masking
        if mask.sum() > 0:
            loss_hyp = bce(logits_hyp[mask.bool()], y_hyp[mask.bool()])
            loss_per = bce(logits_per[mask.bool()], y_per[mask.bool()])
        else:
            loss_hyp = torch.tensor(0.0, device=loss_story.device)
            loss_per = torch.tensor(0.0, device=loss_story.device)

        total_loss = loss_story \
                   + LAMBDA_CENT * loss_cent \
                   + LAMBDA_HYP  * loss_hyp \
                   + LAMBDA_PER  * loss_per
        return {"loss": total_loss,
                "logits_story": logits_story,
                "logits_cent": logits_cent,
                "logits_hyp": logits_hyp,
                "logits_per": logits_per,
               }

model = MultiHeadStoryModel(encoder)

# ── 5. Prepare data splits ─────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df_train = df.sample(frac=0.8, random_state=SEED)
df_val   = df.drop(df_train.index)

# ---- ensure label columns are numeric ints (no strings/NaNs) ----
for col in LABEL_COLUMNS:
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce").fillna(0).astype(int)
    df_val[col]   = pd.to_numeric(df_val[col],   errors="coerce").fillna(0).astype(int)

train_ds = StoryDataset(df_train.reset_index(drop=True))
val_ds   = StoryDataset(df_val.reset_index(drop=True))

# ── 6. Metric‑computing helper ────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    `eval_pred` is (predictions, label_ids) where `predictions` is whatever the
    model's forward pass returns, converted to numpy.  With our custom model
    the Trainer captures the *tuple* of logits, not the dict we returned
    earlier, so we branch on the type.
    """
    preds, labels = eval_pred

    # ---- unpack logits no matter the container --------------------------------
    if isinstance(preds, (list, tuple)):
        logits_story, logits_cent, logits_hyp, logits_per = preds
    elif isinstance(preds, dict):
        logits_story = preds["logits_story"]
        logits_cent  = preds["logits_cent"]
        logits_hyp   = preds["logits_hyp"]
        logits_per   = preds["logits_per"]
    else:
        raise ValueError(f"Unexpected predictions type: {type(preds)}")

    # ---- ground‑truth labels ---------------------------------------------------
    y_story = labels[:, 0]
    y_cent  = labels[:, 1]
    y_hyp   = labels[:, 2]
    y_per   = labels[:, 3]

    # ---- convert logits to class predictions ----------------------------------
    preds_story = (torch.sigmoid(torch.tensor(logits_story)) > 0.5).int().numpy()
    preds_hyp   = (torch.sigmoid(torch.tensor(logits_hyp))  > 0.5).int().numpy()
    preds_per   = (torch.sigmoid(torch.tensor(logits_per))  > 0.5).int().numpy()
    preds_cent  = torch.tensor(logits_cent).argmax(-1).numpy()

    # ---- compute F1s with masking (only where story == 1) ----------------------
    mask = y_story == 1

    def safe_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average="binary") if y_true.sum() > 0 else 0.0

    metrics = {
        "F1_story": safe_f1(y_story, preds_story),
        "F1_hyp":   safe_f1(y_hyp[mask], preds_hyp[mask]),
        "F1_per":   safe_f1(y_per[mask], preds_per[mask]),
        "F1_cent_macro": (
            f1_score(y_cent[mask], preds_cent[mask], average="macro")
            if mask.sum() > 0 else 0.0
        ),
    }
    return metrics

# ── 7. Trainer setup ───────────────────────────────────────────────────────
total_steps = math.ceil(len(train_ds) / BATCH_SIZE) * EPOCHS
warmup      = int(0.1 * total_steps)

training_args = TrainingArguments(
    output_dir           = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate        = LR,
    num_train_epochs     = EPOCHS,
    # evaluation_strategy  = "epoch",
    eval_strategy        = "epoch",       # transformers ≥4.49 uses "eval_strategy"
    metric_for_best_model= "F1_story",
    save_strategy        = "epoch",
    save_total_limit      = 1,     # keep only latest checkpoint (smoke‑test)
    # load_best_model_at_end = True,
    load_best_model_at_end = False,
    # fp16                 = torch.cuda.is_available(),  # use mixed precision if GPU
    fp16                 = False,
    logging_steps        = 10,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
#     callbacks       = [EarlyStoppingCallback(
#                            early_stopping_patience=EARLY_PATIENCE)]
    callbacks       = ([EarlyStoppingCallback(
                           early_stopping_patience=EARLY_PATIENCE)]
                       if EARLY_PATIENCE else [])
)

# ── 8. Train ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR / "final_best")
    tokenizer.save_pretrained(OUTPUT_DIR / "final_best")