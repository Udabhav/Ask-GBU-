"""Ask GBU: chat interface built with Streamlit.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st

from src.config import SIMILARITY_THRESHOLD
from src.retriever import build_retriever

EXAMPLE_QUESTIONS = [
    "What is the B.Tech CSE fee?",
    "How much is the hostel fee?",
    "How many hostels does GBU have?",
    "How is hostel allotted?",
    "Are admissions open?",
    "What is the admissions office contact?",
    "What scholarships does GBU offer?",
    "Where is GBU located?",
]

GREETING = (
    "Hello! I can answer questions about GBU admissions, fees, hostels, "
    "scholarships, and contacts. Pick an example on the left or type your question below."
)


@st.cache_resource(show_spinner="Loading the search model (first time only)...")
def load_retriever():
    """Build the retriever once and reuse it for every chat message."""
    return build_retriever()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def ask(question: str) -> None:
    """Get an answer and add the question and answer to the chat history."""
    result = load_retriever().search(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        result = message.get("result")
        if not result:
            return

        if result["answered"]:
            st.caption(
                f"Academic year: {result['academic_year']} · "
                f"[Official source]({result['source_url']})"
            )
        else:
            st.caption("No verified answer found for this question.")

        with st.expander("How was this matched?"):
            st.write(
                f"Best score: {result['score']} (answers need at least {SIMILARITY_THRESHOLD})"
            )
            for match in result["matches"]:
                st.write(f"- {match['score']:.2f} · {match['question']}")


def main() -> None:
    st.set_page_config(page_title="Ask GBU", page_icon="🎓", layout="centered")
    init_state()

    st.title("Ask GBU")
    st.warning(
        "This is an unofficial student project, not an official Gautam Buddha University "
        "service. Always confirm important details on gbu.ac.in."
    )

    # Sidebar: example questions and a reset button.
    with st.sidebar:
        st.header("Example questions")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": GREETING}]
            st.session_state.pending_question = None
            st.rerun()

    # A question comes either from the chat box or from a sidebar button.
    typed = st.chat_input("Ask about admissions, fees, hostels, or contacts")
    question = st.session_state.pending_question or typed
    st.session_state.pending_question = None

    if question:
        ask(question)

    for message in st.session_state.messages:
        show_message(message)


main()
