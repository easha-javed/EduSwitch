"""Draft synthetic student queries with an LLM. Drafts are NOT training data until humans review them.

python -m src.data_generation --provider groq --per-cell 15 --out data/raw/llm_drafts.csv

Output columns: id, text, intent, source=llm, style, reviewed (empty). Reviewers fill reviewed=ok and fix or delete bad rows.
Never use LLM drafts as the final test set.
"""
import argparse
import json
import re

import pandas as pd

from .config import INTENTS
from .llm_client import chat

STYLES = {
    "code_switched": "natural Urdu and English mixed in one sentence, written in Roman Urdu letters, like a university student texting a teacher",
    "roman_urdu": "Roman Urdu only (Urdu written with English letters), casual, with spelling variation like krni/karni",
    "english": "plain English as a Pakistani university student would write",
    "urdu_script": "Urdu written in Urdu script, with an occasional English word such as LMS or assignment",
}
INTENT_HINTS = {
    "assignment": "deadlines, submission, extensions, formats of assignments",
    "exam": "exam dates, eligibility, papers, venues, rechecking",
    "attendance": "attendance percentage, shortage, leaves, condonation",
    "course_withdrawal": "withdrawing or dropping a course, deadlines and effects",
    "registration": "course registration, add/drop, sections, prerequisites",
    "fee": "fee challan, deadlines, installments, fines, refunds",
    "tech_support": "LMS or portal login, password reset, upload errors, email access",
    "grades": "grade upload, grade correction, GPA, result queries",
    "schedule": "class timings, room changes, timetable, makeup classes",
    "other": "other academic questions that fit none of the other categories",
}


def build_prompt(intent, style, n):
    return (f"Write {n} different short messages a university student might send about: {intent} ({INTENT_HINTS[intent]}).\n"
            f"Style: {STYLES[style]}.\nVary length, tone, and spelling. Some should include numbers or course names. "
            f"Do not use real names. Return ONLY a JSON list of strings.")


def parse_list(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str) and s.strip()] if m else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq", choices=["groq", "gemini"])
    ap.add_argument("--per-cell", type=int, default=15, help="queries per intent and style")
    ap.add_argument("--out", default="data/raw/llm_drafts.csv")
    a = ap.parse_args()
    rows = []
    for intent in INTENTS:
        for style in STYLES:
            try:
                for t in parse_list(chat(build_prompt(intent, style, a.per_cell), a.provider)):
                    rows.append({"text": t, "intent": intent, "source": "llm", "style": style, "reviewed": ""})
            except Exception as e:
                print("skip", intent, style, e)
    df = pd.DataFrame(rows).drop_duplicates("text")
    df.insert(0, "id", [f"llm_{i}" for i in range(len(df))])
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} drafts to {a.out}. Review them before use.")
