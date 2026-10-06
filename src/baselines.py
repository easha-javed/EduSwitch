"""Classical baselines.  B0 majority, B1 TF-IDF + Logistic Regression, L0 dictionary LID.

python -m src.baselines intent
python -m src.baselines lid
"""
import argparse

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import INTENTS, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DIR
from .evaluate import evaluate_predictions, intent_metrics, token_metrics
from .lid_rules import dictionary_lid, load_english_vocab
from .utils import load_json, save_json


def build_tfidf_lr(C: float = 10.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def run_intent(data_dir=PROCESSED_DIR):
    tr, va, te = (pd.read_csv(f"{data_dir}/{s}.csv") for s in ("train", "val", "test"))
    results = {}

    dummy = DummyClassifier(strategy="most_frequent").fit(tr["text"], tr["intent"])
    results["majority"] = intent_metrics(te["intent"], dummy.predict(te["text"]))["macro_f1"]

    model = build_tfidf_lr().fit(tr["text"], tr["intent"])
    print("val macro F1:", round(intent_metrics(va["intent"], model.predict(va["text"]))["macro_f1"], 3))
    proba = model.predict_proba(te["text"])
    out = te.copy()
    out["pred"] = model.classes_[proba.argmax(1)]
    out["confidence"] = proba.max(1)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_DIR / "tfidf_lr.csv", index=False)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "tfidf_intent.joblib")  # demo fallback until transformers are ready
    m, _ = evaluate_predictions(PREDICTIONS_DIR / "tfidf_lr.csv", "tfidf_lr")
    results["tfidf_lr"] = m["macro_f1"]
    save_json(results, METRICS_DIR / "baselines_summary.json")
    print("majority macro F1:", round(results["majority"], 3))


def run_lid(data_dir=PROCESSED_DIR):
    test = load_json(f"{data_dir}/lid_test.json")
    vocab = load_english_vocab()
    preds = [dictionary_lid(ex["tokens"], vocab) for ex in test]
    m = token_metrics([ex["labels"] for ex in test], preds)
    save_json(m, METRICS_DIR / "lid_dictionary.json")
    print(f"dictionary LID token accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} (n tokens={m['n']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["intent", "lid"])
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    a = ap.parse_args()
    run_intent(a.data_dir) if a.task == "intent" else run_lid(a.data_dir)
