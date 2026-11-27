"""Settings for Ask GBU. Change values here, not in the other files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "knowledge_base.json"

# Multilingual model: works for English and Hinglish questions.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# How many matching records to show in the debug output.
TOP_K = 3

# If the best match scores below this (0 to 1), the bot says it does not know.
# Starting value only. It will be tuned in Phase 5 using the test set.
SIMILARITY_THRESHOLD = 0.5

FALLBACK_MESSAGE = (
    "I don't have a verified answer for this. Please contact the relevant office, "
    "or check https://www.gbu.ac.in."
)
