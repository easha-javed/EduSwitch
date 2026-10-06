# EduSwitch: Project Plan

**Urdu and English code switched NLP for student query understanding**
Team: Easha, Rida, Hareem | Course: Natural Language Processing | Stack: Python, Hugging Face, Streamlit

---

## 1. Summary

Students at Pakistani universities rarely write in one language. A typical message looks like *"Sir assignment ki deadline extend hui hai?"* or *"Can I give the exam agar meri attendance 70% hai?"*. Models trained on clean English often handle this badly.

EduSwitch is an NLP system that takes one free form student message and returns:

1. **Token level language identification** (which word is English, which is Urdu)
2. **Intent classification** (what the student is asking about: exam, fee, attendance, and so on)

The project's value is the experiment, not the interface. We measure how much a multilingual Transformer improves over a classical baseline, how much fine tuning on code switched data helps, and where the models still fail.

**Research question:** How well do multilingual Transformers understand Urdu English code switched student queries, and does targeted fine tuning improve them?

## 2. Fit with course guidelines

| Guideline | How we meet it |
|---|---|
| Use pretrained models, no training from scratch | mBERT and XLM-R from Hugging Face |
| Base model vs fine tuned model on a held out test set | Frozen XLM-R with a linear head (base) vs fully fine tuned XLM-R, same test set |
| Small, clean, manageable dataset | About 3,000 intent queries, 1,000 with token labels |
| Methodology and results over interface | Baselines, robustness slices, error analysis are the core; Streamlit comes last |
| Not a simple sentiment or fake news project | Multi task, low resource, code switching problem |

**Risk to settle early:** the guidelines describe the fine tuning category with 1B to 3B models, while XLM-R base is about 270M parameters. Ask the instructor to confirm this fits the "fine tuning small models" category. If not, we add an optional LoRA run on a 1B class model (for example Qwen2.5 1.5B or Llama 3.2 1B) as an extra comparison.

## 3. Scope

**In scope:** Roman Urdu, English, mixed text, and a small share of Urdu script; 10 intent classes; 4 token labels; Streamlit demo.

**Out of scope:** speech, chatbots that generate answers, full Urdu script coverage, training any model from scratch.

## 4. Tasks and label sets

**Task A: Intent classification (sentence level)**

| Intent | Example |
|---|---|
| assignment | "Assignment kab submit karni hai?" |
| exam | "Final exam kis date ko hai?" |
| attendance | "Meri attendance short hai." |
| course_withdrawal | "Can I withdraw this course?" |
| registration | "Course registration kab start hogi?" |
| fee | "Fee challan kab generate hoga?" |
| tech_support | "Mera LMS login nahi ho raha." |
| grades | "Mera grade kab upload hoga?" |
| schedule | "Class ka time change hua hai?" |
| other | Anything academic that fits none of the above |

Rule for multi intent queries such as *"attendance 72% hai can I sit in the final?"*: label by the main question being asked (here exam eligibility goes to **exam**). Write this down in the annotation guide and keep it consistent.

**Task B: Token level language ID**

Labels: `EN` (English), `UR` (Urdu, Roman Urdu or Urdu script), `NUM` (numbers, dates, percentages), `OTHER` (punctuation, emojis, URLs, names that fit neither).

## 5. Data plan

This is the riskiest part of the project, so Hareem starts it on day one.

**Reality check:** a ready made dataset of Urdu English student queries with intent labels almost certainly does not exist. Plan to build a small one. Quickly look for existing Roman Urdu and code switched corpora (Hugging Face Hub, Kaggle, LinCE, Roman Urdu datasets) to borrow vocabulary and spelling variation, and check their licences. Do not depend on them.

**Building our dataset (target 3,000 queries, about 300 per intent):**

1. **Seed set by humans (about 600):** each of us writes queries in our natural style, plus a Google Form sent to classmates. This is the most valuable data.
2. **LLM assisted expansion (about 2,000):** use Gemini or Groq to generate varied queries per intent with prompts that ask for different registers, spellings (krni, karni, karni), and mixing levels. Every generated query is reviewed by a human; discard unnatural ones.
3. **Real test set (about 400):** written only by humans, never by the LLM, never used for training. This is the number we report.

**Target mix:** about 50% code switched, 25% Roman Urdu, 15% English, 10% Urdu script.

**Token labels:** pre label 1,000 queries with a lexicon plus LLM, then humans correct every sentence.

**Annotation quality:** write a one page annotation guide. All three annotate the same 200 items and we report Cohen's kappa. Resolve disagreements by discussion and update the guide.

**Splits:** 70/15/15 stratified by intent. Because generated queries can be near copies, split by *generation prompt group* and remove near duplicates (normalize text, then fuzzy match) before splitting. Freeze the test set on day 6 and never tune on it.

**Ethics:** no real student names, IDs or private messages. Anonymize anything collected.

## 6. Models and experiments

| ID | Model | Purpose |
|---|---|---|
| B0 | Majority class | Sanity floor |
| B1 | TF IDF (word and character n grams) + Logistic Regression | Classical baseline |
| B2 | mBERT, frozen encoder + linear head | Base multilingual model |
| B3 | XLM-R base, frozen encoder + linear head | Base model for the course "base vs fine tuned" requirement |
| F1 | mBERT fully fine tuned | Fine tuned comparison |
| F2 | XLM-R base fully fine tuned | **Main model** |
| X1 (optional) | XLM-R with LoRA (PEFT) | Efficiency comparison |
| X2 (optional) | Few shot LLM via Groq or Gemini API | Strong modern reference point |

**Fine tuning setup:** learning rate 2e-5 (try 1e-5, 3e-5), batch size 16 or 32, 3 to 5 epochs, max length 64, early stopping on validation macro F1, 3 random seeds, report mean and standard deviation. Runs on a free Colab or Kaggle T4.

**Language ID models:** L0 dictionary baseline (English word list lookup, else UR), L1 XLM-R token classification fine tuned on the 1,000 labeled sentences.

**Main experiment:** B1 vs B2 vs B3 vs F1 vs F2 on intent macro F1. **Robustness experiment:** evaluate the best model separately on English, Roman Urdu, Urdu script and code switched slices, and also by code mixing level (low, medium, high share of switches).

## 7. Evaluation

* **Intent:** accuracy, precision, recall, F1, **macro F1 (primary)**, confusion matrix
* **Language ID:** token accuracy, per label precision, recall, F1
* **Reliability:** mean and standard deviation over 3 seeds, bootstrap confidence interval on the test set
* **Error analysis (required section):** at least 30 manually categorized errors, grouped by cause: spelling variation, heavy switching, English technical terms (LMS, challan), abbreviations, ambiguous intent, label noise

Never invent numbers in the report. Every table cell comes from a logged run.

## 8. Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Models | Hugging Face Transformers, Datasets, PEFT, Evaluate; `bert-base-multilingual-cased`, `xlm-roberta-base` |
| Classical ML | scikit-learn |
| Training | PyTorch, Google Colab or Kaggle GPU |
| Data generation | Gemini API or Groq API for drafts only, human review required |
| Annotation | Google Sheets, or Label Studio or Doccano if time allows |
| Experiment tracking | Weights & Biases or a simple CSV log in `results/` |
| Visualization | Matplotlib, Seaborn |
| Demo | Streamlit (Gradio is also allowed) |
| Collaboration | Git and GitHub, shared Google Drive for large files, WhatsApp or Discord for daily sync |

## 9. System architecture

```text
User query
   |
Preprocessing (lowercase copy, normalize spaces, keep original tokens)
   |
Shared multilingual tokenizer (XLM-R)
   |
   +--> Token classification head --> EN / UR / NUM / OTHER per word
   |
   +--> Sentence classification head --> intent + confidence
   |
Post processing (code switched flag if both EN and UR present)
   |
Streamlit UI
```

The two tasks use separate fine tuned checkpoints to keep experiments simple. A shared encoder with two heads is a stretch goal.

## 10. Repository structure

```text
eduswitch/
  README.md                  setup, results summary, screenshots, limitations
  requirements.txt
  .gitignore                 models/, data/raw/, .env
  data/
    README.md                sources, licences, annotation guide link
    raw/                     untracked, shared via Drive
    processed/               train.csv, val.csv, test.csv, lid_train.json ...
    annotation_guide.md
  notebooks/
    01_data_exploration.ipynb
    02_baseline_tfidf.ipynb
    03_mbert_xlmr_intent.ipynb
    04_language_id.ipynb
    05_error_analysis.ipynb
  src/
    config.py                labels, paths, seeds
    preprocessing.py
    data_generation.py       LLM prompts for drafts
    dataset.py               loading, splitting, dedup
    baselines.py
    train_intent.py
    train_lid.py
    evaluate.py              metrics, confusion matrices, slices
    inference.py             predict(text) returns tokens + intent
    utils.py
  app/
    app.py                   Streamlit demo
  models/                    untracked checkpoints
  results/
    metrics/  figures/  predictions/
  report/
    report.md  slides/
  tests/
    test_preprocessing.py  test_inference.py
```

**Git rules:** `main` is always runnable; each person works on a branch (`easha/intent`, `harim/data`, `rida/lid-app`); merge by pull request reviewed by one teammate; fixed random seed 42 in `config.py`; no API keys in the repo (use `.env`).

## 11. Team roles (equal split)

Every member gets the same shared duties plus one lead track. The tracks are sized to be similar, each with one heavy ML piece, one data or analysis piece and one engineering piece.

**Shared duties, identical for all three**

* Write 200 seed queries in your own style
* Correct about 1,000 intent labels (a third of the data) and about 333 token label sentences
* Annotate the shared 200 item agreement set
* Categorize 10 model errors each in the error analysis
* Write one third of the report and present one third of the final demo
* Review at least two pull requests from teammates

**Lead tracks**

| | **Hareem: Data and classical track** | **Rida: mBERT and language ID track** | **Easha: XLM-R and evaluation track** |
|---|---|---|---|
| Heavy ML piece | X2 few shot LLM baseline, B1 TF IDF tuning | L1 XLM-R token classifier, F1 mBERT fine tuning | F2 XLM-R fine tuning with 3 seeds and tuning, X1 LoRA |
| Data or analysis piece | Collection form, LLM draft review queue, real test set (400), dedup, splits, kappa | Token annotation process (pre label, import, quality check), LID error analysis | Evaluation tables, robustness slices, confusion matrices |
| Engineering piece | `dataset.py`, `data_generation.py`, `llm_fewshot.py` | `train_lid.py`, `inference.py`, Streamlit app | `train_intent.py`, `evaluate.py`, repo and Colab setup |
| Report sections | Data and annotation, baselines | Language ID, demo, limitations | Methods, results, fine tuning discussion |
| Error analysis focus | Roman Urdu spelling variation, English slice | Urdu script slice, token level errors | Code switched slice, heavy mixing |

Rough effort is about 9 working days for each person. **Rebalance check on day 7 and day 14:** if anyone is more than 2 days ahead or behind, move a task (for example the LoRA run or a slice of error analysis) to even it out.

## 12. Timeline (3 weeks, adjust to the real deadline)

**Week 1: Data and baselines**

| Days | Hareem | Rida | Easha |
|---|---|---|---|
| 1 to 2 | Dataset search, licences, annotation guide, Google Form | Token label guide, English wordlist, extend `lexicon.py` | Push repo, Colab setup, smoke test on sample data |
| 3 to 4 | LLM draft generation, review queue | Write seeds, pre label script test | Write seeds, run B3 smoke test on sample data |
| 5 to 6 | Dedup, splits, **freeze test set**, kappa | Pre label 1,000 sentences, start corrections | B1 and B3 on real splits, first results table |
| 7 | **Milestone:** clean dataset, B0 and B1 results, rebalance check | | |

**Week 2: Models**

| Days | Hareem | Rida | Easha |
|---|---|---|---|
| 8 to 10 | X2 few shot LLM run, data fixes | B2 frozen mBERT, F1 mBERT fine tuning | F2 XLM-R fine tuning, tuning on validation |
| 11 to 13 | Slice labels, B1 tuning | L1 token classifier, `inference.py` | 3 seed runs, X1 LoRA, evaluation tables |
| 14 | **Milestone:** all models trained, results table complete, rebalance check | | |

**Week 3: Analysis, demo, writing**

| Days | Hareem | Rida | Easha |
|---|---|---|---|
| 15 to 16 | Error analysis (focus slices), category counts | Streamlit app, screenshots, LID errors | Robustness plots, confusion matrices |
| 17 to 18 | Report: data and baselines | Report: LID and demo | Report: methods and results |
| 19 to 20 | README polish, tests | Demo rehearsal, backup screenshots | Slides and presentation script |
| 21 | **Milestone:** code frozen, submission | | |

**MVP fallback:** if time runs short, the minimum valid project is dataset, B1, B3, F2 on intent, evaluation and error analysis. Language ID, LoRA, the LLM baseline and the polished demo are additions.

## 13. Risks and fixes

| Risk | Fix |
|---|---|
| No usable public dataset | Build our own small dataset (Section 5), start day one |
| Synthetic data too clean, inflated scores | Human written real test set, near duplicate removal, report scores on it separately |
| Low annotator agreement | Annotation guide, shared 200 items, kappa, revise labels |
| Colab GPU limits | Max length 64, small batches, mixed precision, save checkpoints to Drive |
| Fine tuning does not beat baseline | Still a valid finding; report honestly with error analysis |
| Class imbalance | Macro F1, class weights, stratified splits |
| Someone falls behind | Daily 10 minute sync, MVP fallback above |

## 14. Final deliverables checklist

* Cleaned dataset with documentation and annotation guide
* Baselines (B0, B1), base models (B2, B3), fine tuned models (F1, F2), language ID models (L0, L1)
* Results tables: model comparison, robustness slices, per class metrics
* Confusion matrices and a categorized error analysis
* Streamlit demo with screenshots
* GitHub repository with README, setup steps, methodology, results, limitations
* Presentation slides and a 5 minute demo script

## 15. Immediate next steps (this week)

1. Confirm with the instructor that XLM-R fine tuning fits the category (Section 2).
2. Create the GitHub repo and invite all three members.
3. Hareem starts dataset search and the annotation guide.
4. Each person writes 200 seed queries in their natural style.
5. Hold a 30 minute kickoff to agree on labels and the 10 intents before anyone annotates.
