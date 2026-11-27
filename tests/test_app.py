from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_with_greeting_and_no_errors():
    at = AppTest.from_file(APP_PATH, default_timeout=300).run()
    assert not at.exception
    assert at.title[0].value == "Ask GBU"
    assert len(at.chat_message) == 1  # the greeting


def test_typed_question_gets_answer_with_source():
    at = AppTest.from_file(APP_PATH, default_timeout=300).run()
    at.chat_input[0].set_value("What is the B.Tech CSE fee?").run()
    assert not at.exception
    answer_text = " ".join(m.value for m in at.markdown)
    assert "1,10,000" in answer_text


def test_sidebar_example_button_adds_question_to_chat():
    at = AppTest.from_file(APP_PATH, default_timeout=300).run()
    button = next(b for b in at.button if b.label == "How many hostels does GBU have?")
    button.click().run()
    assert not at.exception
    # greeting + user question + assistant answer
    assert len(at.chat_message) == 3
