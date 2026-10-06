"""Dictionary based language ID (baseline L0) and language type heuristics."""
from pathlib import Path

from .config import RAW_DIR
from .lexicon import EN_WORDS, UR_WORDS
from .preprocessing import URDU_SCRIPT_RE, is_acronym, rule_label, tokenize_words


def load_english_vocab(path: Path = RAW_DIR / "english_words.txt") -> set:
    vocab = set(EN_WORDS)
    if Path(path).exists():
        vocab |= {w.strip().lower() for w in Path(path).read_text(encoding="utf8").splitlines() if w.strip()}
    return vocab


def label_token(token: str, vocab: set) -> str:
    rule = rule_label(token)
    if rule:
        return rule
    low = token.lower()
    if URDU_SCRIPT_RE.search(token):
        return "UR"
    if low in UR_WORDS:
        return "UR"
    if low in vocab or is_acronym(token):
        return "EN"
    return "UR"  # unknown words in this domain are mostly Roman Urdu spelling variants


def dictionary_lid(tokens, vocab: set = None) -> list:
    vocab = vocab if vocab is not None else load_english_vocab()
    return [label_token(t, vocab) for t in tokens]


def language_counts(text: str, vocab: set = None):
    """Counts of words that are clearly English or clearly Urdu. Unknown words are ignored here,
    so an English sentence with rare words is not mistaken for code switching."""
    vocab = vocab if vocab is not None else load_english_vocab()
    en = ur = 0
    for t in tokenize_words(text):
        if rule_label(t):
            continue
        if URDU_SCRIPT_RE.search(t) or t.lower() in UR_WORDS:
            ur += 1
        elif t.lower() in vocab or is_acronym(t):
            en += 1
    return en, ur


def language_type(text: str, vocab: set = None) -> str:
    """Heuristic only. A human annotated lang_type column always wins over this."""
    en, ur = language_counts(text, vocab)
    if not tokenize_words(text):
        return "other"
    if URDU_SCRIPT_RE.search(text):
        return "code_switched" if en > 0 else "urdu_script"
    if ur == 0:
        return "english"
    return "roman_urdu" if en == 0 else "code_switched"


def mix_level(text: str, vocab: set = None) -> str:
    """none, low, medium, high based on the minority language share of words."""
    en, ur = language_counts(text, vocab)
    total = en + ur
    if total == 0 or min(en, ur) == 0:
        return "none"
    share = min(en, ur) / total
    return "low" if share <= 0.2 else "medium" if share <= 0.4 else "high"
