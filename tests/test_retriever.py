import pytest

from src.retriever import build_retriever


@pytest.fixture(scope="module")
def retriever():
    return build_retriever()


@pytest.mark.parametrize(
    "question, expected_id",
    [
        ("What is the B.Tech CSE fee?", "fee_btech_cse_2026"),
        ("hostel ki fee kitni hai", "hostel_fees_2026"),
        ("how many girls hostels are there", "hostel_count_2026"),
        ("hostel office phone number", "contact_hostel"),
        ("admission ka email kya hai", "contact_admissions"),
    ],
)
def test_known_question_finds_right_record(retriever, question, expected_id):
    result = retriever.search(question)
    assert result["answered"] is True
    assert result["matched_id"] == expected_id


def test_unrelated_question_gets_fallback(retriever):
    result = retriever.search("best pizza recipe with extra cheese")
    assert result["answered"] is False
    assert "verified answer" in result["answer"]


def test_answer_includes_source(retriever):
    result = retriever.search("What is the security fee?")
    assert result["answered"] is True
    assert result["source_url"].startswith("https://")
