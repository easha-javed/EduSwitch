# Contributing

This is a three person course project. Task lists live in docs/ROADMAP.md and the full plan in docs/PROJECT_PLAN.md.

## Working agreement

## Equal workload
Each person has the same shared duties and one lead track of similar size. Full table in docs/PROJECT_PLAN.md, section 11.

| | Hareem | Rida | Easha |
|---|---|---|---|
| Lead track | Data and classical | mBERT and language ID | XLM-R and evaluation |
| Heavy ML piece | X2 LLM few shot, B1 tuning | L1 token classifier, F1 mBERT | F2 XLM-R, X1 LoRA |
| Data or analysis piece | Real test set, dedup, splits, kappa | Token annotation process | Eval tables, robustness slices |
| Engineering piece | dataset.py, data_generation.py | train_lid.py, inference.py, app | train_intent.py, evaluate.py, setup |

Shared by all: 200 seed queries, correcting a third of labels, the shared 200 item agreement set, 10 error categorizations, a third of the report, a third of the presentation, 2 pull request reviews.

Rebalance check on day 7 and day 14. If anyone is 2 or more days ahead or behind, move a task.

## Git
* `main` always runs. Never push to it directly.
* Branch names: `hareem/data`, `rida/lid-app`, `easha/intent-models`.
* One pull request per task, reviewed by one teammate before merge.
* Do not commit `.env`, `models/`, or `data/raw/` (they are in `.gitignore`). Share big files on Google Drive.
* Keep seed 42 from `src/config.py`.
* Run `python -m pytest -q` before opening a pull request.

## Daily sync
10 minutes in the group chat: what I did, what I will do, what blocks me.
