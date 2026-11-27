# Ask GBU: Project Roadmap

> Read-only reference. This file is write-protected on purpose. Keep the working plan here and update it through a deliberate commit.

This is a 4-week plan for a college FAQ chatbot that answers Gautam Buddha University questions using sentence embeddings and retrieval.

## Defaults

- **Core method:** Retrieval with sentence embeddings. The bot returns the best-matching verified answer, with its source link.
- **Optional LLM:** Off by default. Added only if the core works and time allows.
- **UI:** Streamlit.
- **Language:** English plus Hinglish queries, using a multilingual embedding model.
- **Repo:** `Ask-GBU-` for now. Rename decision pending.

---

## Phase 0: Setup (Days 1–2)

**Goal:** A working environment and a clean repo.

- Create the folder structure:
  ```
  Ask-GBU/
  ├── data/
  │   ├── raw/              # copied source text or PDFs (not committed if large)
  │   └── knowledge_base.json
  ├── src/
  │   ├── data_loader.py    # reads and validates the knowledge base
  │   ├── embedder.py       # sentence-transformer encoding
  │   ├── retriever.py      # FAISS search + threshold + fallback
  │   └── config.py         # model name, threshold, paths
  ├── app.py                # Streamlit UI
  ├── eval/
  │   ├── test_questions.csv
  │   └── evaluate.py
  ├── notebooks/            # exploration only
  ├── tests/
  ├── docs/
  ├── requirements.txt
  ├── README.md
  └── report/
  ```
- Create a virtual environment and `requirements.txt`: `sentence-transformers`, `faiss-cpu`, `numpy`, `pandas`, `streamlit`, `pytest`.
- Write a first README: what the bot does, a disclaimer that it is a student project and not an official GBU service, and setup instructions.
- Commit and push: `Initial project structure`.

---

## Phase 1: Knowledge base (Days 3–8)

**Goal:** 100 to 200 verified Q&A records with sources.

- **Topics (7):** Admissions, fees, courses and departments, hostel and campus, exams and results, scholarships, contacts and timings.
- **Collection:** Use only official GBU pages, prospectus PDFs, and notice pages. Save each source's URL and the date checked into `data/raw/`.
- **Record schema:**
  ```json
  {
    "id": "fee_btech_cse_2025",
    "topic": "fees",
    "question": "What is the B.Tech CSE fee?",
    "paraphrases": ["CSE fee kitni hai", "B.Tech computer science cost"],
    "answer": "...",
    "source_url": "https://...",
    "academic_year": "2025-26",
    "last_checked": "2026-11-27"
  }
  ```
- **Validation (`data_loader.py`):** Every record needs an answer, a source URL, and a year. Missing fields or duplicate IDs fail the check.
- **Rule:** If a fact cannot be verified from an official source, leave it out. Wrong fee or date information is worse than no answer.
- Commit after each topic: `Add admissions Q&A`, `Add fee Q&A`, and so on.

---

## Phase 2: Retrieval engine (Days 9–12)

**Goal:** Given a question, return the right answer or a fallback.

- **Embeddings:** Encode each question and each paraphrase with `paraphrase-multilingual-MiniLM-L12-v2`. Several vectors can point back to one answer.
- **Index:** Store vectors in FAISS. At this size, a NumPy array with cosine similarity also works.
- **Search:** Embed the query, retrieve the top 3 matches, and take the best.
- **Threshold:** If the best similarity is below a tuned value (set in Phase 5), return the fallback: *"I don't have a verified answer for this. Please contact the relevant office, or check gbu.ac.in."*
- **Output:** Answer, source link, academic year, and the matched question, so errors are easy to spot.
- **Tests (`tests/`):** Known questions return the right ID, nonsense input returns the fallback, and the data loader rejects bad records.
- Commit: `Add retrieval with threshold and fallback`.

---

## Phase 3: User interface (Days 13–15)

**Goal:** A chat interface someone else can use.

- Streamlit chat page with message history and a text input.
- Each reply shows the answer, the source link, and a note that information may change each year.
- "Topics" sidebar with clickable example questions.
- Disclaimer at the top: unofficial student project.
- Check the layout at phone width, since students will use phones.
- Commit: `Add Streamlit chat UI`.

---

## Phase 4: Deep learning extension (Days 16–18)

**Goal:** A second model that makes the deep learning part visible.

Pick **one**:

- **Option A (recommended): Intent classifier.** Fine-tune DistilBERT (or a small LSTM as a baseline) to classify queries into the 7 topics. Retrieval then searches only within the predicted topic. Compare results with and without it.
- **Option B: LLM rephrasing.** Pass the retrieved answer to Gemini or OpenAI and ask it to rephrase only using that text. Needs an API key, adds cost, and adds hallucination risk, so it needs its own evaluation.

Report the comparison table: baseline versus improved version.

---

## Phase 5: Evaluation (Days 19–22)

**Goal:** Numbers for the report.

- **Test set:** 50 questions with expected answer IDs, including:
  - Direct questions
  - Paraphrases ("hostel kahan milega")
  - Typos
  - Hinglish queries
  - 10 out-of-scope questions that should trigger the fallback
- **Metrics:**
  - Top-1 accuracy
  - Top-3 accuracy
  - MRR (mean reciprocal rank)
  - Fallback precision: out-of-scope questions correctly refused
  - Wrong-answer rate: incorrect answers given with confidence
- **Experiments:**
  - Compare 2 embedding models
  - Sweep the threshold (for example 0.4 to 0.7) and plot answer coverage against error rate
  - Baseline versus intent-classifier version
- **Error analysis:** List 10 failures and classify each (missing data, paraphrase gap, wrong threshold, ambiguous question).
- Commit: `Add evaluation set and results`.

---

## Phase 6: Packaging and demo (Days 23–25)

- Finish the README: architecture diagram, setup steps, how to add Q&A records, known limitations.
- Optional: deploy on Streamlit Community Cloud.
- Record a 2-minute demo: one normal query, one Hinglish query, one fallback case, and the evaluation numbers.
- Clean up unused notebooks and code. Check that `requirements.txt` installs on a fresh machine.

---

## Phase 7: Report and presentation (Days 26–28)

**Report outline:**
1. Introduction and problem
2. Related tools (brief)
3. Dataset and knowledge base construction
4. Method: embeddings, retrieval, threshold, fallback, intent classifier
5. Experiments and results
6. Error analysis
7. Limitations (data freshness, unofficial status, coverage)
8. Future work (live scraping, LLM answers, voice input)

**Presentation (about 8 slides):** Problem, approach, architecture, live demo, results, limitations, next steps.

---

## Timeline

| Week | Focus | Output |
|---|---|---|
| 1 | Setup + knowledge base | Repo structure, 100+ verified Q&A |
| 2 | Retrieval + UI | Working chatbot in Streamlit |
| 3 | Deep learning extension + evaluation | Intent classifier, results table |
| 4 | Packaging, report, demo | README, video, report, slides |

## Git workflow

- One commit per meaningful step, with clear messages.
- Solo project: work on `main`. Use a branch only for risky experiments such as the LLM option.
- Push after each phase.

## Risks

| Risk | Plan |
|---|---|
| GBU website changes or is slow | Keep local copies in `data/raw/` and note the check date |
| Not enough verified data | Start with admissions and fees; cut lower-priority topics |
| Low accuracy on Hinglish | Add paraphrases and test with the multilingual model |
| Time runs short | Phases 0–3 alone make a complete project; Phase 4 is the extra |

## Open decisions

1. Repo name: keep `Ask-GBU-` or rename to `Ask-GBU`?
2. Phase 4: intent classifier (recommended) or LLM rephrasing?
3. Data: official GBU information for admissions and fees, with sources attached.
4. Python level: sets how much code is written versus explained.
