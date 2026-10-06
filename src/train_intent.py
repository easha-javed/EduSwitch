"""Fine tune or probe a multilingual Transformer for intent classification.

Examples (Colab or Kaggle GPU):
  python -m src.train_intent --model xlmr --mode frozen --run-name xlmr_frozen   # B3, the "base model"
  python -m src.train_intent --model xlmr --mode full   --run-name xlmr_finetuned # F2, main model
  python -m src.train_intent --model mbert --mode frozen --run-name mbert_frozen  # B2
  python -m src.train_intent --model mbert --mode full   --run-name mbert_finetuned # F1
  python -m src.train_intent --model xlmr --mode lora   --run-name xlmr_lora      # X1 (optional)

Modes: frozen = encoder frozen, only the linear head trains. full = everything trains. lora = PEFT adapters.
Runs 3 seeds by default and writes mean and std to results/metrics/<run>_summary.json.
"""
import argparse
import inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from .config import (ID2INTENT, INTENT2ID, INTENTS, MAX_LEN, METRICS_DIR, MODEL_NAMES, MODELS_DIR,
                     PREDICTIONS_DIR, PROCESSED_DIR, SEED)
from .evaluate import intent_metrics
from .utils import save_json, set_seed

DEFAULT_LR = {"full": 2e-5, "frozen": 1e-3, "lora": 2e-4}


def _softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def _make_ds(df, tok):
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": [INTENT2ID[i] for i in df["intent"]]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])


def train_one(args, seed):
    set_seed(seed)
    name = MODEL_NAMES.get(args.model, args.model)
    tok = AutoTokenizer.from_pretrained(name)
    tr, va, te = (pd.read_csv(f"{args.data_dir}/{s}.csv") for s in ("train", "val", "test"))
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(INTENTS), id2label=ID2INTENT, label2id=INTENT2ID)
    if args.mode == "frozen":
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif args.mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                                 lora_dropout=0.1, target_modules=["query", "value"]))
    lr = args.lr or DEFAULT_LR[args.mode]
    kw = dict(output_dir=str(MODELS_DIR / "_tmp" / f"{args.run_name}_s{seed}"), learning_rate=lr,
              per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
              num_train_epochs=args.epochs, weight_decay=0.01, warmup_ratio=0.1,
              save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
              greater_is_better=True, save_total_limit=1, seed=seed, data_seed=seed,
              fp16=torch.cuda.is_available(), report_to=args.report_to, logging_steps=20)
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kw[key] = "epoch"
    tk = {"processing_class": tok} if "processing_class" in inspect.signature(Trainer.__init__).parameters else {"tokenizer": tok}
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=_make_ds(tr, tok),
                      eval_dataset=_make_ds(va, tok), data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], **tk)
    trainer.train()
    val_m = trainer.evaluate()
    out = trainer.predict(_make_ds(te, tok))
    proba = _softmax(out.predictions)
    res = te.copy()
    res["pred"] = [ID2INTENT[i] for i in proba.argmax(1)]
    res["confidence"] = proba.max(1)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(PREDICTIONS_DIR / f"{args.run_name}_s{seed}.csv", index=False)
    m = intent_metrics(res["intent"], res["pred"])
    m.update({"seed": seed, "val_macro_f1": val_m["eval_macro_f1"], "lr": lr, "mode": args.mode, "model": name})
    save_json(m, METRICS_DIR / f"{args.run_name}_s{seed}.json")

    save_dir = MODELS_DIR / f"{args.run_name}_s{seed}"
    final = trainer.model.merge_and_unload() if args.mode == "lora" else trainer.model
    final.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print(f"seed {seed}: test macro F1 {m['macro_f1']:.4f}  accuracy {m['accuracy']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlmr", help="mbert | xlmr | any Hugging Face model id")
    ap.add_argument("--mode", default="full", choices=["full", "frozen", "lora"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 1, SEED + 2])
    ap.add_argument("--report-to", default="none", help="none | wandb")
    args = ap.parse_args()
    args.run_name = args.run_name or f"{args.model}_{args.mode}"
    runs = [train_one(args, s) for s in args.seeds]
    f1s, accs = [r["macro_f1"] for r in runs], [r["accuracy"] for r in runs]
    summary = {"run": args.run_name, "seeds": args.seeds, "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
               "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs))}
    save_json(summary, METRICS_DIR / f"{args.run_name}_summary.json")
    print(summary)
    print("Copy the best seed folder to models/intent_best for the demo.")
