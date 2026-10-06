"""X2: few shot LLM intent classifier as a modern reference point.
python -m src.llm_fewshot --provider groq --k 3
Writes results/predictions/llm_fewshot.csv, then run: python -m src.evaluate --pred results/predictions/llm_fewshot.csv"""
import argparse
import time

import pandas as pd

from .config import INTENTS, PREDICTIONS_DIR, PROCESSED_DIR
from .llm_client import chat

SYSTEM = "You classify university student messages written in English, Roman Urdu, Urdu or a mix. Answer with one label only."


def build_prompt(train, text, k):
    shots = "\n".join(f'Message: {r.text}\nLabel: {r.intent}' for c in INTENTS for r in train[train.intent == c].head(k).itertuples())
    return f"Labels: {', '.join(INTENTS)}\n\n{shots}\n\nMessage: {text}\nLabel:"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    test = test.head(a.limit) if a.limit else test
    preds = []
    for text in test["text"]:
        out = chat(build_prompt(train, text, a.k), a.provider, SYSTEM, temperature=0.0).strip().lower()
        preds.append(next((l for l in INTENTS if l in out), "other"))
        time.sleep(0.5)
    test["pred"] = preds
    test.to_csv(PREDICTIONS_DIR / "llm_fewshot.csv", index=False)
