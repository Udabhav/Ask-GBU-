"""Turn text into vectors (lists of numbers) that capture meaning.

Two sentences with similar meaning get vectors that point in similar directions,
even if the words are different.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once. The first call downloads it (one time only)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def encode(texts: list[str]):
    """Return a normalized vector for each text, as a NumPy array of shape (n, dim).

    Normalized means each vector has length 1, so a dot product equals cosine similarity.
    """
    return get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
