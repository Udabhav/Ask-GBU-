# Ask GBU

A college FAQ chatbot for Gautam Buddha University (GBU), Greater Noida. It answers common questions about admissions, fees, hostels, scholarships, and contacts, using sentence embeddings for retrieval and an intent classifier for topic routing.

> **Disclaimer:** This is an unofficial student project. It is **not** an official Gautam Buddha University service. For official information, visit [gbu.ac.in](https://www.gbu.ac.in).

## Status

Project setup stage. See [docs/ROADMAP.md](docs/ROADMAP.md) for the plan and [data/verified_facts.md](data/verified_facts.md) for the facts checked so far.

## Project structure

```
Ask-GBU/
├── data/
│   ├── raw/                 # extracted source text (brochure)
│   ├── verified_facts.md    # facts checked against official sources
│   └── knowledge_base.json  # Q&A records (to be created)
├── src/                     # data loader, embeddings, retrieval (to be built)
├── eval/                    # test questions and evaluation script
├── tests/
├── notebooks/               # exploration only
├── docs/
│   └── ROADMAP.md
├── report/
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Evaluation

```bash
python -m eval.evaluate
```

This asks the bot 138 unseen questions from `eval/questions.csv` in English, Hinglish, and Hindi. 48 of them should get the fallback. The threshold is tuned on the dev split, and results are reported on the test split. Results are in [eval/results/results.md](eval/results/results.md), and the failure analysis is in [eval/error_analysis.md](eval/error_analysis.md).

Test split, current app settings (MiniLM + intent classifier, threshold 0.69, knowledge base v0.2):

| Right answer (answerable) | Refused (should refuse) | Wrong-answer rate | Overall accuracy |
|---|---|---|---|
| 64.4% | 91.7% | 10.1% | 73.9% |

Adding Hindi and Hinglish paraphrases raised overall accuracy from 63.8% to 73.9%, and Hindi right answers from 4 to 8 of 15. Hinglish (7 of 15) and typos are now the weakest parts.

## Data sources

Every answer must cite an official GBU source. Current sources:
- Admission Brochure 2026-27 (v1): `data/raw/brochure_2026-27-v1_extracted.txt`
- Hostels website: `hostels.gbu.ac.in`

## Limitations

- The knowledge base only covers what has been verified. Unknown questions get a fallback message.
- Fees and dates change every year. Answers show the academic year they refer to.
- Not an official service.
