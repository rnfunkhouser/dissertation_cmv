# %%
"""
Jupyter‑style notebook: Hierarchical Multi‑Task RoBERTa
======================================================
Run cell‑by‑cell (VS Code, Jupyter, or Colab).  
Edit the CONFIG section first – then execute top → bottom.
"""

# %% [markdown]
# ## 0. Installation (if needed)
# Uncomment the next cell the first time you run on a fresh environment.

# %%
# !pip install -q transformers datasets scikit-learn torch --upgrade

# %% [markdown]
# ## 1. Configuration
# Central place to tweak paths and hyper‑parameters.

# %%
from __future__ import annotations
import torch, os, random
from pathlib import Path

DATA_PATH = Path("data/story_data.csv")  # <‑‑‑ point to your CSV
MODEL_NAME = "roberta-base"
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LEN = 256
SEED = 42
# Loss‑balancing weights (λ’s)
LAMBDA_CENT = 1.0  # weight for centrality loss
LAMBDA_HYP  = 1.0  # weight for hypothetical loss
LAMBDA_PER  = 1.0  # weight for personal loss
OUTPUT_DIR = Path("outputs/multi_task_roberta")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED); torch.manual_seed(SEED)

# %% [markdown]
# ## 2. Data Loading & Tokenisation

# %%
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import RobertaTokenizerFast

tok = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

raw_ds = load_dataset("csv", data_files=str(DATA_PATH))
if "train" not in raw_ds:
    raw_ds = raw_ds["train"].train_test_split(test_size=0.1, seed=SEED)

for split in raw_ds:
    for col, n in [("story_label", 2), ("cent_label", 3), ("hyp_label", 2), ("per_label", 2)]:
        raw_ds[split] = raw_ds[split].cast_column(col, ClassLabel(num_classes=n))

encoded_ds: DatasetDict = raw_ds.map(tokenize, batched=True)
encoded_ds

# %% [markdown]
# ## 3. Model Definition – Multi‑Task with Masked Loss

# %%
from typing import Dict, Any
from dataclasses import dataclass
from torch import nn
from transformers import RobertaModel, RobertaPreTrainedModel

class MultiTaskRoberta(RobertaPreTrainedModel):
    """RoBERTa with 4 heads + conditional (masked) multi‑task loss."""
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        h = config.hidden_size
        self.story_cls = nn.Linear(h, 2)
        self.cent_cls  = nn.Linear(h, 3)
        self.hyp_cls   = nn.Linear(h, 2)
        self.per_cls   = nn.Linear(h, 2)

        self.loss_story = nn.CrossEntropyLoss()
        self.loss_cent  = nn.CrossEntropyLoss()
        self.loss_bce   = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None,
                story_label=None, cent_label=None,
                hyp_label=None, per_label=None, **kwargs)->Dict[str,Any]:
        enc = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        pooled = enc.last_hidden_state[:, 0]
        ls, lc, lh, lp = None, None, None, None
        log_s = self.story_cls(pooled)
        log_c = self.cent_cls(pooled)
        log_h = self.hyp_cls(pooled)
        log_p = self.per_cls(pooled)

        if story_label is not None:
            ls = self.loss_story(log_s, story_label)
            mask = story_label.bool()
            if mask.any():
                lc = self.loss_cent(log_c[mask], cent_label[mask])
                lh = self.loss_bce(log_h[mask,1], hyp_label[mask].float())
                lp = self.loss_bce(log_p[mask,1], per_label[mask].float())
            else:
                lc = lh = lp = torch.tensor(0.0, device=ls.device)
            loss = ls + LAMBDA_CENT*lc + LAMBDA_HYP*lh + LAMBDA_PER*lp
        else:
            loss = None
        return {"loss": loss, "logits_story": log_s, "logits_cent": log_c,
                "logits_hyp": log_h, "logits_per": log_p}

# %% [markdown]
# ## 4. Metrics

# %%
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    log_s, log_c, log_h, log_p = eval_pred.predictions
    true_s, true_c, true_h, true_p = eval_pred.label_ids
    pred_s = log_s.argmax(-1)
    mask = true_s == 1
    f1_story = f1_score(true_s, pred_s)
    if mask.any():
        f1_cent = f1_score(true_c[mask], log_c[mask].argmax(-1), average="macro")
        f1_hyp  = f1_score(true_h[mask], (log_h[mask,1]>0).astype(int))
        f1_per  = f1_score(true_p[mask], (log_p[mask,1]>0).astype(int))
    else:
        f1_cent = f1_hyp = f1_per = 0.0
    return {"story_f1": f1_story, "cent_f1": f1_cent,
            "hyp_f1": f1_hyp, "per_f1": f1_per}

# %% [markdown]
# ## 5. Training

# %%
from transformers import TrainingArguments, Trainer

model = MultiTaskRoberta.from_pretrained(MODEL_NAME)
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="story_f1",
    seed=SEED,
    report_to="none",
)

task_keys = ["story_label", "cent_label", "hyp_label", "per_label"]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["test"],
    compute_metrics=compute_metrics,
    data_collator=lambda data: {**{k: torch.tensor([d[k] for d in data]) for k in task_keys},
                                **tok.pad({k: torch.tensor([d[k] for d in data]).tolist() if isinstance(d[k], torch.Tensor) else d[k] for k in data[0].keys() if k not in task_keys}, return_tensors="pt")}
)

trainer.train()
trainer.save_model(str(OUTPUT_DIR))

# %% [markdown]
# ## 6. Inference Example

# %%
text = """When I was in college, I skipped classes and still graduated—so you can too."""
inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
with torch.no_grad():
    outputs = model(**inputs)
    prob_story = outputs["logits_story"].softmax(-1)[0,1].item()
    cent_pred  = outputs["logits_cent"].argmax(-1).item()
    hyp_flag   = (outputs["logits_hyp"].sigmoid()[0,1] > 0.5).item()
    per_flag   = (outputs["logits_per"].sigmoid()[0,1] > 0.5).item()
print(f"P(story)={prob_story:.2f}  centrality={cent_pred}  hypothetical={hyp_flag}  personal={per_flag}")
