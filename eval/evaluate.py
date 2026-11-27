"""Phase 5: evaluate Ask GBU on questions it has never seen.

Run from the project root (about a minute; the first run downloads the second model):
    python -m eval.evaluate

The question set (eval/questions.csv) has two splits:
- dev:  used to choose the similarity threshold.
- test: used only for the final numbers. Nothing is tuned on it.

Each question has the record id it should get, or "none" when the bot should refuse
(off-topic questions, and GBU questions that the knowledge base does not cover).

Four setups are compared: 2 embedding models x (with / without the intent classifier).
Results go to eval/results/ (results.md, test_predictions.csv, and two charts).
"""

import csv
import time
from collections import Counter
from datetime import date

import numpy as np

from src.config import EMBEDDING_MODEL, ROOT, SIMILARITY_THRESHOLD, USE_INTENT_FILTER
from src.data_loader import load_knowledge_base
from src.embedder import encode, get_model
from src.intent import load_dataset, train_baseline
from src.retriever import build_retriever

QUESTIONS_PATH = ROOT / "eval" / "questions.csv"
RESULTS_DIR = ROOT / "eval" / "results"

# Short name -> model. The first one is the model the app uses.
MODELS = {
    "MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "paraphrase-multilingual-mpnet-base-v2",
}

LANGUAGES = ["en", "hinglish", "hi"]
ANSWERABLE_KINDS = ["direct", "paraphrase", "typo"]
REFUSE_KINDS = ["off_topic", "not_in_kb"]
SPLITS = ["dev", "test"]
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.30, 0.901, 0.01)]
# The untuned value from Phase 2, kept for the before/after comparison.
STARTING_THRESHOLD = 0.50


# ---------------------------------------------------------------- questions

def normalize(text: str) -> str:
    """Lowercase and tidy a question so near-identical copies are caught."""
    return " ".join(text.lower().strip(" ?.!").split())


def load_questions(path=QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{key: value.strip() for key, value in row.items()} for row in csv.DictReader(f)]


def texts_the_system_has_seen(records: list[dict]) -> list[str]:
    """Knowledge base questions and paraphrases, plus all intent dataset questions."""
    kb_texts = [text for r in records for text in [r["question"], *r["paraphrases"]]]
    train_texts, _, test_texts, _ = load_dataset()
    return kb_texts + train_texts + test_texts


def check_questions(questions: list[dict], records: list[dict], seen_texts: list[str]) -> list[str]:
    """Return a list of problems with the question set. Empty means it is fine."""
    errors = []
    record_ids = {r["id"] for r in records}
    seen = {normalize(t) for t in seen_texts}
    asked = set()
    for line, q in enumerate(questions, start=2):  # line 1 is the header
        where = f"line {line} ({q['question'][:40]})"
        key = normalize(q["question"])
        if key in asked:
            errors.append(f"{where}: duplicate question")
        asked.add(key)
        if key in seen:
            errors.append(f"{where}: already in the knowledge base or the intent data")
        if q["expected_id"] != "none" and q["expected_id"] not in record_ids:
            errors.append(f"{where}: unknown expected_id '{q['expected_id']}'")
        if q["language"] not in LANGUAGES:
            errors.append(f"{where}: unknown language '{q['language']}'")
        if q["split"] not in SPLITS:
            errors.append(f"{where}: unknown split '{q['split']}'")
        if q["kind"] not in ANSWERABLE_KINDS + REFUSE_KINDS:
            errors.append(f"{where}: unknown kind '{q['kind']}'")
        elif (q["kind"] in REFUSE_KINDS) != (q["expected_id"] == "none"):
            errors.append(f"{where}: kind '{q['kind']}' does not match expected_id")
    return errors


# ---------------------------------------------------------------- running the bot

def run_model(model_name: str, questions: list[dict]):
    """Ask every question once with and once without the intent filter.

    The threshold is switched off here (-1), so every threshold can be tried
    afterwards without asking the questions again.
    """
    retriever = build_retriever(use_intent=True, model_name=model_name)
    runs = {}
    for use_intent in (False, True):
        predictions = []
        start = time.perf_counter()
        for q in questions:
            r = retriever.search(q["question"], threshold=-1.0, use_intent=use_intent)
            predictions.append(
                {
                    "ranked_ids": [m["id"] for m in r["matches"]],
                    "score": r["score"],
                    "out_of_scope": r.get("reason") == "out_of_scope",
                    "intent": r["intent"] or "",
                    "intent_confidence": r["intent_confidence"],
                }
            )
        ms_per_question = (time.perf_counter() - start) * 1000 / len(questions)
        runs[use_intent] = (predictions, ms_per_question)
    return retriever, runs


def is_answered(pred: dict, threshold: float) -> bool:
    return not pred["out_of_scope"] and pred["score"] >= threshold


def outcome(question: dict, pred: dict, threshold: float) -> str:
    """correct / wrong / missed (refused but should answer) / refused (correctly)."""
    answered = is_answered(pred, threshold)
    if question["expected_id"] == "none":
        return "wrong" if answered else "refused"
    if not answered:
        return "missed"
    return "correct" if pred["ranked_ids"][0] == question["expected_id"] else "wrong"


# ---------------------------------------------------------------- metrics

def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def ranking_metrics(pairs) -> dict:
    """Top-1, top-3 and MRR on answerable questions. The threshold does not matter here."""
    ranks = []
    for q, p in pairs:
        if q["expected_id"] == "none":
            continue
        ids = p["ranked_ids"][:3]
        ranks.append(ids.index(q["expected_id"]) + 1 if q["expected_id"] in ids else None)
    return {
        "top1": safe_div(sum(r == 1 for r in ranks), len(ranks)),
        "top3": safe_div(sum(r is not None for r in ranks), len(ranks)),
        "mrr": safe_div(sum(1 / r for r in ranks if r), len(ranks)),
    }


def decision_metrics(pairs, threshold: float) -> dict:
    """What the user actually sees: an answer (right or wrong) or the fallback."""
    counts = Counter(outcome(q, p, threshold) for q, p in pairs)
    n = len(pairs)
    n_answerable = sum(q["expected_id"] != "none" for q, _ in pairs)
    answered = counts["correct"] + counts["wrong"]
    fallbacks = counts["missed"] + counts["refused"]
    return {
        "n": n,
        "counts": counts,
        # Answerable questions that got the right answer.
        "answer_accuracy": safe_div(counts["correct"], n_answerable),
        # Questions the bot should refuse that it did refuse.
        "refusal_rate": safe_div(counts["refused"], n - n_answerable),
        # Of all fallbacks shown, how many were deserved.
        "fallback_precision": safe_div(counts["refused"], fallbacks),
        # Wrong answers given with confidence, out of all questions.
        "wrong_answer_rate": safe_div(counts["wrong"], n),
        # Right answer or rightful refusal.
        "overall_accuracy": safe_div(counts["correct"] + counts["refused"], n),
        "coverage": safe_div(answered, n),
        "error_when_answering": safe_div(counts["wrong"], answered),
    }


def choose_threshold(pairs) -> float:
    """Best overall accuracy; on a tie, fewer wrong answers, then the lower threshold."""

    def key(t):
        m = decision_metrics(pairs, t)
        return (m["overall_accuracy"], -m["wrong_answer_rate"], -t)

    return max(THRESHOLDS, key=key)


def group_by(pairs, field: str) -> dict:
    groups = {}
    for q, p in pairs:
        groups.setdefault(q[field], []).append((q, p))
    return groups


def check_live_search(retriever, use_intent: bool, pairs, threshold: float) -> list[str]:
    """The sweep reuses one search per question. Confirm that a normal search with the
    chosen threshold gives the same result for every question."""
    mismatches = []
    for q, p in pairs:
        r = retriever.search(q["question"], threshold=threshold, use_intent=use_intent)
        swept_answered = is_answered(p, threshold)
        swept = (swept_answered, p["ranked_ids"][0] if swept_answered else None)
        if (r["answered"], r.get("matched_id")) != swept:
            mismatches.append(q["question"])
    return mismatches


def is_hindi(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def intent_accuracy_by_script(model_name: str) -> dict:
    """Intent classifier accuracy on the intent dataset's test split, by script."""
    train_texts, train_labels, test_texts, test_labels = load_dataset()
    clf = train_baseline(train_texts, train_labels, model_name)
    correct = clf.predict(encode(test_texts, model_name)) == np.array(test_labels)
    hindi = np.array([is_hindi(t) for t in test_texts])
    return {
        "All": (correct.mean(), len(correct)),
        "Latin script (English + Hinglish)": (correct[~hindi].mean(), int((~hindi).sum())),
        "Devanagari (Hindi)": (correct[hindi].mean(), int(hindi.sum())),
    }


# ---------------------------------------------------------------- charts

INK, INK_2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # draw to files, no window
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    return plt


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0, pad=6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Move label positions apart so they do not overlap (keeps their order)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def plot_threshold_sweep(pairs, chosen: float, old: float, setup_name: str, path) -> None:
    plt = _pyplot()
    metrics = {t: decision_metrics(pairs, t) for t in THRESHOLDS}
    lines = [
        ("Answerable: right answer", "answer_accuracy"),
        ("Should refuse: refused", "refusal_rate"),
        ("Wrong answers (all questions)", "wrong_answer_rate"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _style(ax)
    ends = []
    for color, (label, key) in zip(SERIES, lines):
        ys = [metrics[t][key] * 100 for t in THRESHOLDS]
        ax.plot(THRESHOLDS, ys, color=color, linewidth=1.6, label=label, solid_capstyle="round")
        ends.append(ys[-1])
    for (label, _), y_text, y_end in zip(lines, _spread(ends, 7), ends):
        ax.annotate(label, (THRESHOLDS[-1], y_end), xytext=(THRESHOLDS[-1] + 0.012, y_text),
                    textcoords="data", va="center", fontsize=8.5, color=INK, annotation_clip=False)
    for t, text, color in [(chosen, f"chosen {chosen:.2f}", INK), (old, f"starting {old:.2f}", INK_2)]:
        ax.axvline(t, color=color, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(t + 0.006, 103, text, fontsize=8.5, color=color, va="bottom")
    ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("% of questions")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3,
              loc="upper left", bbox_to_anchor=(-0.01, -0.14))
    fig.suptitle(f"Threshold sweep on the dev split ({setup_name})", x=0.06, ha="left",
                 fontsize=11, color=INK)
    fig.subplots_adjust(left=0.09, right=0.74, top=0.84, bottom=0.22)
    fig.savefig(path)
    plt.close(fig)


def plot_coverage_vs_error(setups: dict, path) -> None:
    """One line per embedding model (with intent). Each point is one threshold."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    _style(ax)
    with_intent = {name: s for name, s in setups.items() if s["use_intent"]}
    for color, (name, s) in zip(SERIES, with_intent.items()):
        xs, ys = [], []
        for t in THRESHOLDS:
            m = decision_metrics(s["dev"], t)
            if m["coverage"] > 0:
                xs.append(m["coverage"] * 100)
                ys.append(m["error_when_answering"] * 100)
        ax.plot(xs, ys, color=color, linewidth=1.6, label=name)
        m = decision_metrics(s["dev"], s["threshold"])
        x, y = m["coverage"] * 100, m["error_when_answering"] * 100
        ax.plot([x], [y], marker="o", markersize=6.5, color=color,
                markeredgecolor="white", markeredgewidth=1.4, linestyle="none")
        ax.annotate(f"{s['model']} at {s['threshold']:.2f}", (x, y), xytext=(8, -12),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Coverage: % of all dev questions answered")
    ax.set_ylabel("% of given answers that are wrong")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fig.suptitle("Coverage against error, dev split (dot = chosen threshold)", x=0.06,
                 ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- report

def pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:.1f}%"  # x != x is True only for NaN


def table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def got_text(pred: dict, threshold: float) -> str:
    if pred["out_of_scope"]:
        return "fallback (out of scope)"
    if pred["score"] < threshold:
        return f"fallback (low score; top was {pred['ranked_ids'][0]})"
    return pred["ranked_ids"][0]


def write_report(questions, setups, main_name, params, intent_scores, mismatches) -> str:
    main = setups[main_name]
    out = [
        "# Phase 5 evaluation results",
        "",
        f"Generated by `python -m eval.evaluate` on {date.today().isoformat()}. "
        "Do not edit by hand; run the script again instead.",
        "",
        "The threshold for each setup is chosen on the **dev** split. "
        "All result tables use the **test** split, which was never used for tuning.",
        "",
        "## Question set",
        "",
    ]
    rows = []
    for split in SPLITS:
        qs = [q for q in questions if q["split"] == split]
        langs = Counter(q["language"] for q in qs)
        rows.append([
            split, len(qs), sum(q["expected_id"] != "none" for q in qs),
            sum(q["kind"] == "off_topic" for q in qs), sum(q["kind"] == "not_in_kb" for q in qs),
            " / ".join(str(langs[lang]) for lang in LANGUAGES),
        ])
    out += table(["Split", "Questions", "Answerable", "Off-topic", "GBU but not in KB",
                  "en / hinglish / hi"], rows)

    out += ["## Ranking quality (test, answerable questions, threshold not applied)", ""]
    out += table(
        ["Setup", "Top-1", "Top-3", "MRR"],
        [[name, pct(m["top1"]), pct(m["top3"]), f"{m['mrr']:.3f}"]
         for name, s in setups.items() for m in [ranking_metrics(s["test"])]],
    )

    out += ["## Main results (test, at the threshold chosen on dev)", ""]
    rows = []
    for name, s in setups.items():
        m = decision_metrics(s["test"], s["threshold"])
        rows.append([name, f"{s['threshold']:.2f}", pct(m["answer_accuracy"]), pct(m["refusal_rate"]),
                     pct(m["fallback_precision"]), pct(m["wrong_answer_rate"]),
                     pct(m["overall_accuracy"]), f"{s['ms']:.0f} ms"])
    m = decision_metrics(main["test"], STARTING_THRESHOLD)
    rows.append([f"{main_name} (starting threshold)", f"{STARTING_THRESHOLD:.2f}", pct(m["answer_accuracy"]),
                 pct(m["refusal_rate"]), pct(m["fallback_precision"]), pct(m["wrong_answer_rate"]),
                 pct(m["overall_accuracy"]), f"{main['ms']:.0f} ms"])
    out += table(["Setup", "Threshold", "Right answer (answerable)", "Refused (should refuse)",
                  "Fallback precision", "Wrong-answer rate", "Overall accuracy", "Time per question"],
                 rows)
    out += [
        "- **Right answer:** answerable questions that got the correct record.",
        "- **Refused:** off-topic or not-in-KB questions that got the fallback.",
        "- **Fallback precision:** of all fallbacks shown, the share that were deserved.",
        "- **Wrong-answer rate:** wrong answers given with confidence, out of all questions.",
        "- **Overall accuracy:** right answer or rightful refusal, out of all questions.",
        "- Time per question is measured on this laptop's CPU and changes between runs.",
        "",
        "Model size: " + ", ".join(f"{k} {v / 1e6:.0f}M parameters" for k, v in params.items()) + ".",
        "",
    ]

    out += ["## By language (test, at the chosen threshold)", ""]
    rows = []
    for name, s in setups.items():
        groups = group_by(s["test"], "language")
        cells = []
        for lang in LANGUAGES:
            m = decision_metrics(groups[lang], s["threshold"])
            cells.append(f"{pct(m['answer_accuracy'])} / {pct(m['refusal_rate'])}")
        rows.append([name, *cells])
    out += table(["Setup (right answer / refused)", "English", "Hinglish", "Hindi"], rows)

    out += ["## By question kind (test, at the chosen threshold)", ""]
    rows = []
    for name, s in setups.items():
        groups = group_by(s["test"], "kind")
        cells = []
        for kind in ANSWERABLE_KINDS:
            cells.append(pct(decision_metrics(groups[kind], s["threshold"])["answer_accuracy"]))
        for kind in REFUSE_KINDS:
            cells.append(pct(decision_metrics(groups[kind], s["threshold"])["refusal_rate"]))
        rows.append([name, *cells])
    out += table(["Setup", "Direct (right)", "Paraphrase (right)", "Typo (right)",
                  "Off-topic (refused)", "Not in KB (refused)"], rows)

    out += ["## Intent classifier by script (intent dataset test split)", ""]
    rows = []
    for model, scores in intent_scores.items():
        rows.append([model, *[f"{pct(acc)} (n={n})" for acc, n in scores.values()]])
    out += table(["Embedding model", *next(iter(intent_scores.values())).keys()], rows)

    out += [f"## Threshold sweep (dev, {main_name})", "",
            "Table version of `threshold_sweep.png`.", ""]
    rows = []
    for t in sorted(set(THRESHOLDS[::5]) | {main["threshold"], STARTING_THRESHOLD}):
        m = decision_metrics(main["dev"], t)
        mark = " (chosen)" if t == main["threshold"] else " (starting)" if t == STARTING_THRESHOLD else ""
        rows.append([f"{t:.2f}{mark}", pct(m["answer_accuracy"]), pct(m["refusal_rate"]),
                     pct(m["wrong_answer_rate"]), pct(m["overall_accuracy"])])
    out += table(["Threshold", "Right answer", "Refused", "Wrong-answer rate", "Overall accuracy"], rows)

    out += [f"## Failures on the test split ({main_name}, threshold {main['threshold']:.2f})", "",
            "Each failure is classified by hand in `eval/error_analysis.md`.", ""]
    rows = []
    for q, p in main["test"]:
        result = outcome(q, p, main["threshold"])
        if result in ("wrong", "missed"):
            rows.append([q["question"], q["language"], q["kind"], q["expected_id"],
                         got_text(p, main["threshold"]), f"{p['score']:.3f}",
                         f"{p['intent']} {p['intent_confidence']:.2f}" if p["intent"] else "-"])
    out += table(["Question", "Lang", "Kind", "Expected", "Got", "Score", "Intent"], rows)

    out += ["## Checks", "",
            f"- Questions overlapping the knowledge base or intent data: 0 (checked before running).",
            f"- Live search with the chosen threshold matched the sweep for "
            f"{'every' if not mismatches else 'all but ' + str(len(mismatches))} test question.",
            f"- The app (`src/config.py`) uses threshold {SIMILARITY_THRESHOLD:.2f}; "
            f"the dev-tuned value for {main_name} is {main['threshold']:.2f}.",
            "", "Charts: `threshold_sweep.png`, `coverage_vs_error.png`.", ""]
    return "\n".join(out)


def write_predictions(pairs, threshold: float, path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "language", "kind", "expected_id", "outcome", "got",
                         "score", "top3", "intent", "intent_confidence"])
        for q, p in pairs:
            writer.writerow([q["question"], q["language"], q["kind"], q["expected_id"],
                             outcome(q, p, threshold), got_text(p, threshold), p["score"],
                             " ".join(p["ranked_ids"]), p["intent"], p["intent_confidence"]])


# ---------------------------------------------------------------- main

def main() -> None:
    records = load_knowledge_base()["records"]
    questions = load_questions()
    errors = check_questions(questions, records, texts_the_system_has_seen(records))
    if errors:
        print("Problems in eval/questions.csv:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print(f"{len(questions)} questions loaded and checked.")

    setups, retrievers, params, intent_scores = {}, {}, {}, {}
    for label, model_name in MODELS.items():
        print(f"Running {label} ({model_name})...")
        retriever, runs = run_model(model_name, questions)
        params[label] = sum(p.numel() for p in get_model(model_name).parameters())
        intent_scores[label] = intent_accuracy_by_script(model_name)
        for use_intent, (preds, ms) in runs.items():
            name = f"{label} + intent" if use_intent else f"{label}, no intent"
            pairs = list(zip(questions, preds))
            setups[name] = {
                "use_intent": use_intent,
                "model": label,
                "ms": ms,
                "dev": [(q, p) for q, p in pairs if q["split"] == "dev"],
                "test": [(q, p) for q, p in pairs if q["split"] == "test"],
            }
            retrievers[name] = retriever

    for s in setups.values():
        s["threshold"] = choose_threshold(s["dev"])

    mismatches = []
    for name, s in setups.items():
        mismatches += check_live_search(retrievers[name], s["use_intent"], s["test"], s["threshold"])

    main_label = next(k for k, v in MODELS.items() if v == EMBEDDING_MODEL)
    main_name = f"{main_label} + intent" if USE_INTENT_FILTER else f"{main_label}, no intent"
    main = setups[main_name]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = write_report(questions, setups, main_name, params, intent_scores, mismatches)
    (RESULTS_DIR / "results.md").write_text(report, encoding="utf-8")
    write_predictions(main["test"], main["threshold"], RESULTS_DIR / "test_predictions.csv")
    plot_threshold_sweep(main["dev"], main["threshold"], STARTING_THRESHOLD, main_name,
                         RESULTS_DIR / "threshold_sweep.png")
    plot_coverage_vs_error(setups, RESULTS_DIR / "coverage_vs_error.png")

    print("\nTest split, at the threshold chosen on dev:")
    for name, s in setups.items():
        m = decision_metrics(s["test"], s["threshold"])
        print(f"  {name:20s} threshold {s['threshold']:.2f}  right {pct(m['answer_accuracy']):>6}  "
              f"refused {pct(m['refusal_rate']):>6}  wrong {pct(m['wrong_answer_rate']):>6}  "
              f"overall {pct(m['overall_accuracy']):>6}")
    if mismatches:
        print(f"\nWARNING: live search disagreed with the sweep on {len(mismatches)} questions.")
    if SIMILARITY_THRESHOLD != main["threshold"]:
        print(f"\nNote: config.py uses SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}, "
              f"but the dev-tuned value for {main_name} is {main['threshold']:.2f}.")
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
