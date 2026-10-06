# Annotation guide (draft, finish together before anyone annotates)

## Intent
Label the main thing the student wants to know or do.
* Multi intent query: choose the main question. "Attendance 72% hai can I sit in the final?" is **exam** (eligibility to sit the exam). Note such decisions here so everyone follows them.
* Use **other** only for academic or campus questions that fit no class. Not for unclear text.
* Unclear or off topic text: delete the row.

## Token language labels
* **EN**: English words, including technical terms and acronyms (LMS, GPA, assignment, submit).
* **UR**: Urdu words in Roman Urdu or Urdu script (hai, karni, mera, kab). Spelling variants count (krni, nhi).
* **NUM**: numbers, percentages, times (72%, 5, 10:30).
* **OTHER**: punctuation, emojis, URLs, names that are neither language.
* Decide and write down: Sir and Madam are EN. Loanwords written in English letters follow their usual language (challan is UR, assignment is EN).

## Process
1. Everyone labels the same 200 items, then run `python -m src.agreement`.
2. Discuss every disagreement, update this guide.
3. Split the remaining data into thirds and label independently.
