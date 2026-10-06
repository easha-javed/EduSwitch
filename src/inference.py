"""One entry point for the demo and for ad hoc testing.

    from src.inference import EduSwitchPredictor
    p = EduSwitchPredictor()
    p.predict("Sir mera LMS login nahi ho raha")

Uses fine tuned Transformers if models/intent_best and models/lid_best exist,
otherwise falls back to the TF-IDF model and the dictionary LID, so the demo always runs.
"""
from pathlib import Path

import joblib

from .config import MAX_LEN, MODELS_DIR
from .lid_rules import dictionary_lid, load_english_vocab
from .preprocessing import normalize_text, rule_label, tokenize_words


class EduSwitchPredictor:
    def __init__(self, intent_dir=MODELS_DIR / "intent_best", lid_dir=MODELS_DIR / "lid_best",
                 tfidf_path=MODELS_DIR / "tfidf_intent.joblib"):
        self.vocab = load_english_vocab()
        self.intent_backend = self.lid_backend = "none"
        self.intent_model = self.lid_model = self.tfidf = None
        if (Path(intent_dir) / "config.json").exists():
            self._load_intent(intent_dir)
        elif Path(tfidf_path).exists():
            self.tfidf, self.intent_backend = joblib.load(tfidf_path), "tfidf"
        if (Path(lid_dir) / "config.json").exists():
            self._load_lid(lid_dir)
        else:
            self.lid_backend = "dictionary"

    def _load_intent(self, d):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.intent_tok = AutoTokenizer.from_pretrained(d)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(d).eval()
        self.intent_backend = "transformer"

    def _load_lid(self, d):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.lid_tok = AutoTokenizer.from_pretrained(d)
        self.lid_model = AutoModelForTokenClassification.from_pretrained(d).eval()
        self.lid_backend = "transformer"

    def predict_intent(self, text):
        if self.intent_backend == "transformer":
            import torch
            enc = self.intent_tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            with torch.no_grad():
                probs = torch.softmax(self.intent_model(**enc).logits, -1)[0]
            i = int(probs.argmax())
            return self.intent_model.config.id2label[i], float(probs[i])
        if self.intent_backend == "tfidf":
            proba = self.tfidf.predict_proba([text])[0]
            i = int(proba.argmax())
            return self.tfidf.classes_[i], float(proba[i])
        return "unavailable", 0.0

    def predict_lid(self, tokens):
        if self.lid_backend != "transformer" or not tokens:
            return dictionary_lid(tokens, self.vocab)
        import torch
        enc = self.lid_tok(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            pred = self.lid_model(**enc).logits.argmax(-1)[0].tolist()
        labels, prev = {}, None
        for idx, w in enumerate(enc.word_ids(0)):
            if w is not None and w != prev:
                labels[w] = self.lid_model.config.id2label[pred[idx]]
            prev = w
        out = []
        for i, t in enumerate(tokens):
            out.append(rule_label(t) or labels.get(i) or dictionary_lid([t], self.vocab)[0])
        return out

    def predict(self, text):
        text = normalize_text(text)
        tokens = tokenize_words(text)
        langs = self.predict_lid(tokens)
        intent, conf = self.predict_intent(text)
        present = sorted({l for l in langs if l in ("EN", "UR")})
        return {"text": text, "tokens": [{"token": t, "lang": l} for t, l in zip(tokens, langs)],
                "languages": present, "code_switched": present == ["EN", "UR"],
                "intent": intent, "confidence": conf,
                "backend": {"intent": self.intent_backend, "lid": self.lid_backend}}
