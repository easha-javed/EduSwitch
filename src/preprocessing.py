"""Text normalization and word tokenization shared by every script."""
import re
import unicodedata

URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_NUM = r"\d+(?:[.,:/]\d+)*%?"
TOKEN_RE = re.compile(
    rf"{_NUM}|[A-Za-z\u0600-\u06FF\u0750-\u077F]+(?:['\u2019][A-Za-z]+)*|[^\w\s]|_",
    re.UNICODE,
)
NUM_RE = re.compile(rf"^{_NUM}$")
PUNCT_RE = re.compile(r"^(?:[^\w\s]|_)$")


def normalize_text(text: str) -> str:
    """Unicode normalize, drop zero width characters, collapse whitespace. Keeps case and spelling."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> list:
    """Split into word level tokens. Example: 'attendance 70% hai?' -> ['attendance', '70%', 'hai', '?']"""
    return TOKEN_RE.findall(normalize_text(text))


def rule_label(token: str):
    """Labels that never need a model: numbers and punctuation. Returns None for real words."""
    if NUM_RE.match(token):
        return "NUM"
    if PUNCT_RE.match(token):
        return "OTHER"
    return None


def has_urdu_script(text: str) -> bool:
    return bool(URDU_SCRIPT_RE.search(text))


def is_acronym(token: str) -> bool:
    return len(token) >= 2 and token.isascii() and token.isalpha() and token.isupper()


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization used only to detect duplicates."""
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
