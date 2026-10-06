"""Speed up token annotation: pre label with the dictionary baseline, humans correct in Google Sheets.

1) export:  python -m src.prelabel_lid export --input data/raw/queries.csv --n 1000 --out data/raw/lid_to_correct.csv
   -> one token per row: sentence_id, token_idx, token, label.  Upload to Sheets, fix the label column.
2) import:  python -m src.prelabel_lid import --input corrected.csv --out data/raw/lid_annotated.json
"""
import argparse

import pandas as pd

from .lid_rules import dictionary_lid, load_english_vocab
from .preprocessing import tokenize_words
from .utils import save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "import"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mode == "export":
        df = pd.read_csv(a.input).sample(frac=1, random_state=a.seed).head(a.n)
        vocab, rows = load_english_vocab(), []
        for sid, text in zip(df["id"], df["text"]):
            toks = tokenize_words(text)
            for i, (t, l) in enumerate(zip(toks, dictionary_lid(toks, vocab))):
                rows.append({"sentence_id": sid, "token_idx": i, "token": t, "label": l})
        pd.DataFrame(rows).to_csv(a.out, index=False)
    else:
        df = pd.read_csv(a.input).sort_values(["sentence_id", "token_idx"])
        data = [{"id": str(sid), "tokens": g["token"].astype(str).tolist(), "labels": g["label"].tolist()}
                for sid, g in df.groupby("sentence_id", sort=False)]
        save_json(data, a.out)
        print(f"wrote {len(data)} sentences")
