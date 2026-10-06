"""Build clean, leakage safe train/val/test files.

Intent data:  python -m src.dataset intent --input data/raw/queries.csv [--real-only-test]
LID data:     python -m src.dataset lid --input data/raw/lid_annotated.json

Input intent csv columns: id, text, intent, source (human_real | human_seed | llm), optional lang_type.
Input LID json: [{"id": "1", "tokens": [...], "labels": [...]}]
"""
import argparse
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import INTENTS, LID_LABELS, PROCESSED_DIR, SEED, SOURCES
from .lid_rules import language_type, load_english_vocab, mix_level
from .preprocessing import normalize_for_dedup, normalize_text
from .utils import load_json, save_json


def near_duplicate_groups(texts, threshold: float = 0.9):
    """Return a group id per text. Texts with char n-gram cosine similarity >= threshold share a group."""
    norm = [normalize_for_dedup(t) for t in texts]
    n = len(norm)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(norm)
    for start in range(0, n, 1000):
        sims = (X[start:start + 1000] @ X.T).toarray()
        for i in range(sims.shape[0]):
            gi = start + i
            for j in (sims[i] >= threshold).nonzero()[0]:
                if j > gi:
                    parent[find(j)] = find(gi)
    return [find(i) for i in range(n)]


def group_stratified_split(df, label_col="intent", group_col="group", ratios=(0.7, 0.15, 0.15), seed=SEED):
    """Assign whole groups to splits so near duplicates never cross splits. Returns a Series of 0/1/2."""
    rng = random.Random(seed)
    majority = df.groupby(group_col)[label_col].agg(lambda s: s.value_counts().idxmax())
    sizes = df.groupby(group_col).size()
    assign = {}
    for label in sorted(majority.unique()):
        groups = [g for g in majority.index[majority == label]]
        rng.shuffle(groups)
        total = sum(sizes[g] for g in groups)
        targets = [r * total for r in ratios]
        counts = [0, 0, 0]
        for g in groups:
            i = max((k for k in range(3) if ratios[k] > 0), key=lambda k: targets[k] - counts[k])
            assign[g] = i
            counts[i] += sizes[g]
    return df[group_col].map(assign)


def check_no_leakage(*frames, group_col="group"):
    seen = {}
    for name, f in zip(("train", "val", "test"), frames):
        for g in set(f[group_col]):
            assert g not in seen or seen[g] == name, f"group {g} appears in {seen[g]} and {name}"
            seen[g] = name


def prepare_intent(input_csv, out_dir=PROCESSED_DIR, real_only_test=False, threshold=0.9, seed=SEED):
    df = pd.read_csv(input_csv)
    need = {"text", "intent"}
    assert need <= set(df.columns), f"csv needs columns {need}"
    if "id" not in df.columns:
        df["id"] = range(len(df))
    if "source" not in df.columns:
        df["source"] = "human_seed"
    bad = set(df["intent"]) - set(INTENTS)
    assert not bad, f"unknown intents {bad}. Allowed: {INTENTS}"
    bad_src = set(df["source"]) - set(SOURCES)
    assert not bad_src, f"unknown source values {bad_src}. Allowed: {SOURCES}"

    df["text"] = df["text"].map(normalize_text)
    # exact duplicates: keep the most trustworthy source
    df["_prio"] = df["source"].map({s: i for i, s in enumerate(SOURCES)})
    df["_key"] = df["text"].map(normalize_for_dedup)
    before = len(df)
    df = df.sort_values("_prio").drop_duplicates("_key").drop(columns=["_prio", "_key"]).reset_index(drop=True)
    print(f"dropped {before - len(df)} exact duplicates")

    df["group"] = near_duplicate_groups(df["text"].tolist(), threshold)
    vocab = load_english_vocab()
    if "lang_type" not in df.columns:
        df["lang_type"] = df["text"].map(lambda t: language_type(t, vocab))
    df["mix_level"] = df["text"].map(lambda t: mix_level(t, vocab))

    if real_only_test:
        test = df[df["source"] == "human_real"]
        rest = df[df["source"] != "human_real"]
        leaked = rest["group"].isin(set(test["group"]))
        print(f"removed {int(leaked.sum())} non real rows that were near duplicates of test rows")
        rest = rest[~leaked]
        s = group_stratified_split(rest, ratios=(0.85, 0.15, 0.0), seed=seed)
        train, val = rest[s == 0], rest[s == 1]
    else:
        s = group_stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=seed)
        train, val, test = df[s == 0], df[s == 1], df[s == 2]

    check_no_leakage(train, val, test)
    out_dir = type(PROCESSED_DIR)(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, f in (("train", train), ("val", val), ("test", test)):
        f.drop(columns=["group"]).to_csv(out_dir / f"{name}.csv", index=False)
        f[["id", "group"]].to_csv(out_dir / f"{name}_groups.csv", index=False)
    stats = pd.DataFrame({
        "train": train["intent"].value_counts(), "val": val["intent"].value_counts(), "test": test["intent"].value_counts(),
    }).fillna(0).astype(int)
    print(stats)
    print("\nlanguage types in test:\n", test["lang_type"].value_counts())
    return train, val, test


def prepare_lid(input_json, out_dir=PROCESSED_DIR, ratios=(0.7, 0.15, 0.15), seed=SEED):
    data = load_json(input_json)
    for ex in data:
        assert len(ex["tokens"]) == len(ex["labels"]), f"length mismatch in {ex.get('id')}"
        assert set(ex["labels"]) <= set(LID_LABELS), f"bad label in {ex.get('id')}"
    rng = random.Random(seed)
    rng.shuffle(data)
    n = len(data)
    a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    parts = {"lid_train": data[:a], "lid_val": data[a:b], "lid_test": data[b:]}
    for name, part in parts.items():
        save_json(part, type(PROCESSED_DIR)(out_dir) / f"{name}.json")
        print(name, len(part))
    return parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["intent", "lid"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--real-only-test", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.kind == "intent":
        prepare_intent(a.input, a.out, a.real_only_test, a.threshold, a.seed)
    else:
        prepare_lid(a.input, a.out, seed=a.seed)
