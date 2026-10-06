"""Token level language identification with a Transformer (L1).
python -m src.train_lid --model xlmr --run-name lid_xlmr
Needs data/processed/lid_{train,val,test}.json (see src/dataset.py lid)."""
import argparse
import inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from .config import ID2LID, LID2ID, LID_LABELS, MAX_LEN, METRICS_DIR, MODEL_NAMES, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, SEED
from .evaluate import plot_confusion_matrix, token_metrics
from .config import FIGURES_DIR
from .utils import load_json, save_json, set_seed


def _encode(batch, tok):
    enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    labels = []
    for i, tags in enumerate(batch["tags"]):
        prev, row = None, []
        for w in enc.word_ids(batch_index=i):
            row.append(-100 if w is None or w == prev else LID2ID[tags[w]])
            prev = w
        labels.append(row)
    enc["labels"] = labels
    return enc


def _ds(examples, tok):
    ds = Dataset.from_dict({"tokens": [e["tokens"] for e in examples], "tags": [e["labels"] for e in examples]})
    return ds.map(lambda b: _encode(b, tok), batched=True, remove_columns=["tokens", "tags"])


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    mask = labels != -100
    return {"token_accuracy": accuracy_score(labels[mask], preds[mask]),
            "macro_f1": f1_score(labels[mask], preds[mask], average="macro", zero_division=0)}


def train(args, seed):
    set_seed(seed)
    name = MODEL_NAMES.get(args.model, args.model)
    tok = AutoTokenizer.from_pretrained(name)
    tr, va, te = (load_json(f"{args.data_dir}/lid_{s}.json") for s in ("train", "val", "test"))
    model = AutoModelForTokenClassification.from_pretrained(name, num_labels=len(LID_LABELS), id2label=ID2LID, label2id=LID2ID)
    kw = dict(output_dir=str(MODELS_DIR / "_tmp" / f"{args.run_name}_s{seed}"), learning_rate=args.lr,
              per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64, num_train_epochs=args.epochs,
              weight_decay=0.01, warmup_ratio=0.1, save_strategy="epoch", load_best_model_at_end=True,
              metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1, seed=seed,
              fp16=torch.cuda.is_available(), report_to="none", logging_steps=20)
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kw[key] = "epoch"
    tk = {"processing_class": tok} if "processing_class" in inspect.signature(Trainer.__init__).parameters else {"tokenizer": tok}
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=_ds(tr, tok), eval_dataset=_ds(va, tok),
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], **tk)
    trainer.train()

    test_ds = _ds(te, tok)
    out = trainer.predict(test_ds)
    preds = out.predictions.argmax(-1)
    true_seqs, pred_seqs, rows = [], [], []
    for ex, p_row, l_row in zip(te, preds, out.label_ids):
        p_words = [ID2LID[int(p)] for p, l in zip(p_row, l_row) if l != -100]
        p_words += ["UR"] * (len(ex["tokens"]) - len(p_words))  # words lost to truncation (rare)
        true_seqs.append(ex["labels"]); pred_seqs.append(p_words)
        rows += [{"sentence_id": ex.get("id"), "token": t, "true": a, "pred": b} for t, a, b in zip(ex["tokens"], ex["labels"], p_words)]
    pd.DataFrame(rows).to_csv(PREDICTIONS_DIR / f"{args.run_name}_s{seed}.csv", index=False)
    m = token_metrics(true_seqs, pred_seqs)
    save_json(m, METRICS_DIR / f"{args.run_name}_s{seed}.json")
    plot_confusion_matrix([r["true"] for r in rows], [r["pred"] for r in rows], LID_LABELS,
                          FIGURES_DIR / f"{args.run_name}_s{seed}_confusion.png", "Token language ID")
    save_dir = MODELS_DIR / f"{args.run_name}_s{seed}"
    trainer.save_model(save_dir); tok.save_pretrained(save_dir)
    print(f"seed {seed}: token accuracy {m['accuracy']:.4f} macro F1 {m['macro_f1']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlmr")
    ap.add_argument("--run-name", default="lid_xlmr")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 1, SEED + 2])
    a = ap.parse_args()
    runs = [train(a, s) for s in a.seeds]
    f1s = [r["macro_f1"] for r in runs]
    save_json({"run": a.run_name, "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))},
              METRICS_DIR / f"{a.run_name}_summary.json")
    print("Copy the best seed folder to models/lid_best for the demo.")
