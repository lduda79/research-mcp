"""Wandelt Text in Vektoren um. Modell wird beim ersten Aufruf geladen."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
    """Gibt eine (n, 384)-Matrix normalisierter float32-Vektoren zurueck."""
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]