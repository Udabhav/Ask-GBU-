"""Checks for the Phase 5 question set and metric code (no models needed)."""

from eval.evaluate import (
    LANGUAGES,
    SPLITS,
    check_questions,
    choose_threshold,
    decision_metrics,
    load_questions,
    ranking_metrics,
    texts_the_system_has_seen,
)
from src.data_loader import load_knowledge_base


def test_question_set_is_valid_and_unseen():
    records = load_knowledge_base()["records"]
    questions = load_questions()
    assert check_questions(questions, records, texts_the_system_has_seen(records)) == []


def test_every_split_has_every_language_and_refusals():
    questions = load_questions()
    for split in SPLITS:
        qs = [q for q in questions if q["split"] == split]
        assert {q["language"] for q in qs} == set(LANGUAGES)
        assert any(q["expected_id"] == "none" for q in qs)


def test_check_questions_catches_problems():
    records = [{"id": "a", "question": "What is A?", "paraphrases": ["a kya hai"]}]
    questions = [
        {"question": "A kya hai?", "expected_id": "a", "language": "hinglish", "kind": "direct", "split": "dev"},
        {"question": "what is b", "expected_id": "b", "language": "en", "kind": "direct", "split": "dev"},
        {"question": "tell me a joke", "expected_id": "a", "language": "en", "kind": "off_topic", "split": "dev"},
    ]
    seen = [records[0]["question"], *records[0]["paraphrases"]]
    errors = check_questions(questions, records, seen)
    assert any("already in the knowledge base" in e for e in errors)
    assert any("unknown expected_id 'b'" in e for e in errors)
    assert any("does not match expected_id" in e for e in errors)


def _q(expected):
    return {"expected_id": expected}


def _p(ranked, score, out_of_scope=False):
    return {"ranked_ids": ranked, "score": score, "out_of_scope": out_of_scope}


PAIRS = [
    (_q("a"), _p(["a", "b"], 0.80)),  # right answer
    (_q("a"), _p(["b", "a"], 0.70)),  # wrong answer at 0.5, right record is rank 2
    (_q("b"), _p(["b"], 0.55)),       # right at 0.5, missed at 0.6
    (_q("none"), _p(["a"], 0.65)),    # should refuse: wrong at 0.5, refused at 0.7
    (_q("none"), _p(["a"], 0.90, out_of_scope=True)),  # refused by the intent classifier
]


def test_decision_metrics_counts_each_outcome():
    m = decision_metrics(PAIRS, 0.5)
    assert m["counts"] == {"correct": 2, "wrong": 2, "refused": 1}
    assert m["answer_accuracy"] == 2 / 3
    assert m["refusal_rate"] == 1 / 2
    assert m["wrong_answer_rate"] == 2 / 5
    assert m["overall_accuracy"] == 3 / 5


def test_ranking_metrics_ignore_threshold_and_refusals():
    m = ranking_metrics(PAIRS)
    assert m["top1"] == 2 / 3
    assert m["top3"] == 1.0
    assert m["mrr"] == (1 + 0.5 + 1) / 3


def test_choose_threshold_prefers_fewer_wrong_answers_on_a_tie():
    # Overall is 3/5 for 0.30-0.55, 0.66-0.70 and 0.71-0.80. Only 0.71-0.80 has
    # no wrong answers, and 0.71 is the lowest of those.
    assert choose_threshold(PAIRS) == 0.71
