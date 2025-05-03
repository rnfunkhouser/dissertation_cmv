"""
Load the best model produced by `train_multitask_roberta.py`
and label the entire 209 k-row CMV dataset.

❱ Edit CONFIG. Then run:
       python predict_multitask_roberta.py
"""

import torch, pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from train_multitask_roberta import (  # reuse dataset + model defs
    StoryDataset, MultiHeadStoryModel, MODEL_NAME
)
from transformers import RobertaModel

# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_DIR     = Path("../models/story_multitask/final_best")
FULL_DATA_CSV = Path("../data/cleaned_final_convos.csv")
SAVE_CSV      = Path("../data/cmv_with_story_features.csv")
BATCH_SIZE    = 32

# ── Load tokenizer & model ────────────────────────────────────────────────
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
encoder   = RobertaModel.from_pretrained(MODEL_DIR)      # fine‑tuned weights
model     = MultiHeadStoryModel(encoder).eval().cuda()

# ── Prepare dataset & loader ──────────────────────────────────────────────
df_full = pd.read_csv(FULL_DATA_CSV)
ds      = StoryDataset(df_full, for_inference=True)
loader  = torch.utils.data.DataLoader(ds,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=4)

# ── Inference loop + window aggregation ───────────────────────────────────
from collections import Counter, defaultdict

agg = defaultdict(lambda: {"story": [], "cent": [], "hyp": [], "per": []})

with torch.no_grad():
    for batch in tqdm(loader, desc="Predict"):
        row_ids   = batch.pop("row_id")                 # tensor of ids
        input_ids = batch["input_ids"].cuda()
        attention = batch["attention_mask"].cuda()

        out = model(input_ids=input_ids, attention_mask=attention)

        story_prob = torch.sigmoid(out["logits_story"]).cpu().numpy()
        hyp_prob   = torch.sigmoid(out["logits_hyp"]).cpu().numpy()
        per_prob   = torch.sigmoid(out["logits_per"]).cpu().numpy()
        cent_pred  = out["logits_cent"].softmax(-1).argmax(-1).cpu().numpy()

        for rid, s, c, h, p in zip(row_ids.cpu().tolist(),
                                   story_prob, cent_pred, hyp_prob, per_prob):
            agg[rid]["story"].append(float(s))
            agg[rid]["cent"].append(int(c))
            agg[rid]["hyp"].append(float(h))
            agg[rid]["per"].append(float(p))

# ── Aggregate to comment‑level predictions ────────────────────────────────
story_pred, cent_pred, hyp_pred, per_pred = [], [], [], []
for idx in range(len(df_full)):
    wins = agg[idx]
    if not wins["story"]:                       # safeguard (very short comments)
        story_pred.append(0.0)
        cent_pred.append(0)
        hyp_pred.append(0.0)
        per_pred.append(0.0)
        continue

    story_max = max(wins["story"])              # max prob across windows
    story_pred.append(story_max)

    cent_mode = Counter(wins["cent"]).most_common(1)[0][0]
    cent_pred.append(cent_mode)

    hyp_pred.append(max(wins["hyp"]))
    per_pred.append(max(wins["per"]))

df_full["story_flag"] = [int(p > 0.5) for p in story_pred]
df_full["centrality"] = cent_pred
df_full["hyp_flag"]   = [int(p > 0.5) if s > 0.5 else None
                         for p, s in zip(hyp_pred, story_pred)]
df_full["per_flag"]   = [int(p > 0.5) if s > 0.5 else None
                         for p, s in zip(per_pred, story_pred)]

df_full.to_csv(SAVE_CSV, index=False)
print(f"Saved enriched CSV → {SAVE_CSV}")