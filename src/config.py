"""Single place for labels, paths and defaults. Change labels here only."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

INTENTS = [
    "assignment", "exam", "attendance", "course_withdrawal", "registration",
    "fee", "tech_support", "grades", "schedule", "other",
]
INTENT2ID = {l: i for i, l in enumerate(INTENTS)}
ID2INTENT = {i: l for l, i in INTENT2ID.items()}

# EN English, UR Urdu (Roman Urdu or Urdu script), NUM numbers, OTHER punctuation and the rest
LID_LABELS = ["EN", "UR", "NUM", "OTHER"]
LID2ID = {l: i for i, l in enumerate(LID_LABELS)}
ID2LID = {i: l for l, i in LID2ID.items()}

MODEL_NAMES = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}
MAX_LEN = 64

# Source column values in the intent csv. Only human_real rows may form the final test set.
SOURCES = ["human_real", "human_seed", "llm"]
