"""Settings for Ask GBU. Change values here, not in the other files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "knowledge_base.json"

# Multilingual model: works for English and Hinglish questions.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# How many matching records to show in the debug output.
TOP_K = 3

# If the best match scores below this (0 to 1), the bot says it does not know.
# Tuned in Phase 5 on the dev split of eval/questions.csv (the starting value was 0.50).
# Re-tune with `python -m eval.evaluate` whenever the knowledge base changes.
SIMILARITY_THRESHOLD = 0.69

# Intent classifier settings (starting values; tune in Phase 5, not on the test set).
USE_INTENT_FILTER = True
# Search only inside the predicted topic when the classifier is at least this sure.
INTENT_CONFIDENCE = 0.5
# Treat the question as out of scope when "other" is predicted with at least this confidence.
OTHER_CONFIDENCE = 0.6

FALLBACK_MESSAGE = (
    "I don't have a verified answer for this. Please contact the relevant office, "
    "or check https://www.gbu.ac.in."
)
