import pandas as pd

from src.config import INTENTS, SAMPLE_DIR
from src.dataset import check_no_leakage, near_duplicate_groups, prepare_intent, prepare_lid
from src.evaluate import intent_metrics, slice_metrics
from src.lid_rules import dictionary_lid, language_type, mix_level
from src.preprocessing import normalize_text, rule_label, tokenize_words


def test_tokenize_keeps_numbers_and_punct():
    assert tokenize_words("attendance 70% hai?") == ["attendance", "70%", "hai", "?"]
    assert tokenize_words("can't  go") == ["can't", "go"]


def test_rule_label():
    assert rule_label("70%") == "NUM" and rule_label("?") == "OTHER" and rule_label("hai") is None


def test_normalize():
    assert normalize_text("  a \u200b  b ") == "a b"


def test_dictionary_lid_and_types():
    toks = tokenize_words("Mujhe assignment submit karni hai before Friday")
    assert dictionary_lid(toks) == ["UR", "EN", "EN", "UR", "UR", "EN", "EN"]
    assert language_type("Can I withdraw this course?") == "english"
    assert language_type("Mujhe assignment submit karni hai before Friday") == "code_switched"
    assert mix_level("Can I withdraw this course?") == "none"


def test_near_duplicates_grouped():
    g = near_duplicate_groups(["Assignment kab submit karni hai?", "assignment kab submit karni hai", "Fee challan kab aayega"])
    assert g[0] == g[1] != g[2]


def test_prepare_intent_no_leakage(tmp_path):
    train, val, test = prepare_intent(SAMPLE_DIR / "sample_queries.csv", tmp_path)
    check_no_leakage(train, val, test)
    assert set(train["intent"]) <= set(INTENTS) and len(train) > len(test)


def test_real_only_test(tmp_path):
    _, _, test = prepare_intent(SAMPLE_DIR / "sample_queries.csv", tmp_path, real_only_test=True)
    assert set(test["source"]) == {"human_real"}


def test_prepare_lid(tmp_path):
    parts = prepare_lid(SAMPLE_DIR / "sample_lid.json", tmp_path)
    assert sum(len(p) for p in parts.values()) == 12


def test_metrics_and_slices():
    m = intent_metrics(["exam", "fee", "fee"], ["exam", "fee", "exam"])
    assert 0 < m["macro_f1"] < 1
    df = pd.DataFrame({"intent": ["exam", "fee"], "pred": ["exam", "exam"], "lang_type": ["english", "english"]})
    assert slice_metrics(df).iloc[0]["n"] == 2
