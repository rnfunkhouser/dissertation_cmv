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

# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_DIR     = Path("../models/story_multitask/final_best")
FULL_DATA_CSV = Path("../data/cleaned_final_convos.csv")
SAVE_CSV      = Path("../data/cmv_with_story_features.csv")
BATCH_SIZE    = 32

# ── Load tokenizer & model ────────────────────────────────────────────────
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
encoder   = MultiHeadStoryModel.encoder
encoder   = encoder.from_pretrained(MODEL_DIR)  # load fine-tuned weights
model     = MultiHeadStoryModel(encoder).eval().cuda()

# ── Prepare dataset & loader ──────────────────────────────────────────────
df_full = pd.read_csv(FULL_DATA_CSV)
ds      = StoryDataset(df_full, for_inference=True)
loader  = torch.utils.data.DataLoader(ds,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=4)

# ── Inference loop ────────────────────────────────────────────────────────
pred_rows: List[Dict] = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Predict"):
        input_ids = batch["input_ids"].cuda()
        attention = batch["attention_mask"].cuda()
        out = model(input_ids=input_ids, attention_mask=attention)

        story_prob = torch.sigmoid(out["logits_story"])
        hyp_prob   = torch.sigmoid(out["logits_hyp"])
        per_prob   = torch.sigmoid(out["logits_per"])
        cent_pred  = out["logits_cent"].softmax(-1).argmax(-1)

        # detach → cpu → numpy
        pred_rows.extend(zip(story_prob.cpu().numpy(),
                             cent_pred.cpu().numpy(),
                             hyp_prob.cpu().numpy(),
                             per_prob.cpu().numpy()))

# ── Post-processing for sliding-window aggregation (simple version) ───────
# If a comment was split into windows, take max probabilities / majority vote.
# NB: This simplified example assumes ds.rows aligns 1-to-1 with df rows.
# If you used multiple windows, you would need to aggregate by original row id.
story_pred, cent_pred, hyp_pred, per_pred = map(list, zip(*pred_rows))

df_full["story_flag"] = [int(p > 0.5) for p in story_pred]
df_full["centrality"] = cent_pred
df_full["hyp_flag"]   = [int(p > 0.5) if s > 0.5 else None
                         for p, s in zip(hyp_pred, story_pred)]
df_full["per_flag"]   = [int(p > 0.5) if s > 0.5 else None
                         for p, s in zip(per_pred, story_pred)]

df_full.to_csv(SAVE_CSV, index=False)
print(f"Saved enriched CSV → {SAVE_CSV}")