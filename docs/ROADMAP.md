# Roadmap and task owners

Tasks are split so each member has a similar workload. Tick items off as pull requests merge. Details and the weekly timeline are in PROJECT_PLAN.md.

## Included in the starter code
Leakage safe splitting, TF IDF baseline, dictionary language ID, metrics, robustness slices, confusion matrices, error export, annotation helpers, inference class, Streamlit demo, tests, CI.

Written but not yet run on a GPU, so expect first run fixes: `train_intent.py`, `train_lid.py`, `llm_fewshot.py`, `data_generation.py`, the Colab notebook.

## Hareem: data and classical track
* [ ] Collect about 400 real student queries with a Google Form (test set, source human_real)
* [ ] Review LLM drafts and finish `data/annotation_guide.md` and `data/README.md`
* [ ] Build real splits and run kappa on the shared 200 items
* [ ] Run and tune B1, run X2
* [ ] Error analysis for Roman Urdu spelling variation and the English slice
* [ ] Report sections: data and annotation, baselines

## Rida: mBERT and language ID track
* [ ] Extend `lexicon.py` and run the token annotation process (about 1,000 sentences)
* [ ] Run B2 and F1, train L1
* [ ] Finish the Streamlit demo and screenshots
* [ ] Error analysis for the Urdu script slice and token errors
* [ ] Report sections: language ID, demo, limitations

## Easha: XLM-R and evaluation track
* [ ] Publish the repo and set up shared Drive storage
* [ ] Run B3 and F2 with three seeds and tuning, optional X1
* [ ] Results tables, robustness plots, per class metrics
* [ ] Error analysis for the code switched slice and heavy mixing
* [ ] Report sections: methods, results, fine tuning discussion

## Everyone
* [ ] 200 seed queries each
* [ ] Correct a third of the intent labels and token labels
* [ ] Annotate the shared 200 item agreement set
* [ ] Categorize 10 errors each
* [ ] Write a third of the report and present a third of the demo
* [ ] Review at least two pull requests
