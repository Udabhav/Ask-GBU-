"""Turn text into vectors (lists of numbers) that capture meaning.

Two sentences with similar meaning get vectors that point in similar directions,
even if the words are different.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=2)
def get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load a model once. The first call downloads it (one time only)."""
    return SentenceTransformer(model_name)


def encode(texts: list[str], model_name: str = EMBEDDING_MODEL):
    """Return a normalized vector for each text, as a NumPy array of shape (n, dim).

    Normalized means each vector has length 1, so a dot product equals cosine similarity.
    The evaluation script passes a different model_name to compare models;
    the app always uses EMBEDDING_MODEL from config.py.
    """
    return get_model(model_name).encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
