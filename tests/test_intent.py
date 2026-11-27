import pytest

from src.intent import LABELS, IntentClassifier, load_dataset
from src.retriever import build_retriever


@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


@pytest.fixture(scope="module")
def retriever():
    return build_retriever(use_intent=True)


def test_dataset_has_all_labels_in_both_splits():
    train_texts, train_labels, test_texts, test_labels = load_dataset()
    assert set(train_labels) == set(range(len(LABELS)))
    assert set(test_labels) == set(range(len(LABELS)))
    assert not set(train_texts) & set(test_texts), "a test question is also in training"


@pytest.mark.parametrize(
    "question, expected_topic",
    [
        ("mess charges kitne hai", "hostel"),
        ("btech ki fees kitni hai", "fees"),
        ("best pizza recipe", "other"),
    ],
)
def test_classifier_predicts_training_examples(classifier, question, expected_topic):
    topic, confidence = classifier.predict(question)
    assert topic == expected_topic
    assert 0.0 <= confidence <= 1.0


def test_out_of_scope_question_gets_fallback(retriever):
    result = retriever.search("tell me a joke")
    assert result["answered"] is False
    assert result["intent"] == "other"


def test_confident_topic_limits_the_search(retriever):
    result = retriever.search("hostel ki fee kitni hai")
    assert result["topic_filter"] == "hostel"
    assert all("hostel" in m["topics"] for m in result["matches"])
    assert result["matched_id"] == "hostel_fees_2026"


def test_record_with_extra_topic_is_found_inside_that_topic(retriever):
    # contact_hostel is a "contacts" record that is also tagged "hostel".
    result = retriever.search("hostel office se baat kaise karein")
    assert result["matched_id"] == "contact_hostel"


def test_intent_routes_admission_status_question(retriever):
    # Without the intent filter this goes to the admissions contact record by mistake.
    result = retriever.search("admission abhi chal rahe hai kya")
    assert result["intent"] == "admissions"
    assert result["matched_id"] == "admissions_status_2026"


def test_search_without_intent_still_works(retriever):
    result = retriever.search("What is the B.Tech CSE fee?", use_intent=False)
    assert result["intent"] is None
    assert result["matched_id"] == "fee_btech_cse_2026"
