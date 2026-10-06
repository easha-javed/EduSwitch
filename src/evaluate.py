"""Metrics, confusion matrices and robustness slices.

Usage:  python -m src.evaluate --pred results/predictions/tfidf_lr.csv
The predictions csv needs columns: text, intent, pred (optional: lang_type, mix_level).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .config import FIGURES_DIR, INTENTS, LID_LABELS, METRICS_DIR
from .lid_rules import language_type, mix_level
from .utils import save_json


def intent_metrics(y_true, y_pred, labels=INTENTS) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "per_class": {l: rep[l] for l in labels if l in rep},
    }


def token_metrics(true_seqs, pred_seqs, labels=LID_LABELS) -> dict:
    yt = [t for s in true_seqs for t in s]
    yp = [t for s in pred_seqs for t in s]
    return intent_metrics(yt, yp, labels)


def plot_confusion_matrix(y_true, y_pred, labels, path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), max(4, len(labels) * 0.6)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slice_metrics(df, true_col="intent", pred_col="pred", slice_cols=("lang_type", "mix_level")) -> pd.DataFrame:
    """Robustness table: metrics per language type and per code mixing level."""
    rows = []
    for col in slice_cols:
        if col not in df.columns:
            continue
        for value, sub in df.groupby(col):
            rows.append({
                "slice": col, "value": value, "n": len(sub),
                "accuracy": accuracy_score(sub[true_col], sub[pred_col]),
                "macro_f1": f1_score(sub[true_col], sub[pred_col], average="macro", zero_division=0),
            })
    return pd.DataFrame(rows)


def evaluate_predictions(pred_csv, name=None):
    df = pd.read_csv(pred_csv)
    name = name or Path(pred_csv).stem
    if "lang_type" not in df.columns:
        df["lang_type"] = df["text"].map(language_type)
    if "mix_level" not in df.columns:
        df["mix_level"] = df["text"].map(mix_level)
    m = intent_metrics(df["intent"], df["pred"])
    save_json(m, METRICS_DIR / f"{name}_eval.json")
    slices = slice_metrics(df)
    slices.to_csv(METRICS_DIR / f"{name}_slices.csv", index=False)
    plot_confusion_matrix(df["intent"], df["pred"], INTENTS, FIGURES_DIR / f"{name}_confusion.png", name)
    df[df["intent"] != df["pred"]].to_csv(METRICS_DIR / f"{name}_errors.csv", index=False)  # input for error analysis
    print(f"{name}: macro_f1={m['macro_f1']:.3f} accuracy={m['accuracy']:.3f} n={m['n']}")
    print(slices.to_string(index=False))
    return m, slices


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    evaluate_predictions(a.pred, a.name)
