"""Cohen's kappa between two annotators on the shared 200 items.
python -m src.agreement annotator_a.csv annotator_b.csv --col intent
Both csvs need an id column and the label column."""
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--col", default="intent")
    x = ap.parse_args()
    m = pd.read_csv(x.a).merge(pd.read_csv(x.b), on="id", suffixes=("_a", "_b"))
    k = cohen_kappa_score(m[f"{x.col}_a"], m[f"{x.col}_b"])
    print(f"items={len(m)} kappa={k:.3f} raw agreement={(m[f'{x.col}_a'] == m[f'{x.col}_b']).mean():.3f}")
    print(m[m[f"{x.col}_a"] != m[f"{x.col}_b"]].to_string(index=False))
