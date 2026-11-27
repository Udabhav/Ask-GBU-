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

## Data sources

Every answer must cite an official GBU source. Current sources:
- Admission Brochure 2026-27 (v1): `data/raw/brochure_2026-27-v1_extracted.txt`
- Hostels website: `hostels.gbu.ac.in`

## Limitations

- The knowledge base only covers what has been verified. Unknown questions get a fallback message.
- Fees and dates change every year. Answers show the academic year they refer to.
- Not an official service.
