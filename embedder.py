"""
Embedder — converts text chunks into numerical vectors using Sentence Transformers.

WHY EMBEDDINGS:
Computers can't directly compare the "meaning" of two sentences. Embeddings
solve this by converting text into fixed-size numerical vectors (arrays of
numbers) where:
  - Similar meanings → vectors close together in space
  - Different meanings → vectors far apart

For example:
  "What is machine learning?" and "Explain ML"
  → Their vectors would be very close (cosine similarity ~ 0.9)

  "What is machine learning?" and "How to bake a cake?"
  → Their vectors would be far apart (cosine similarity ~ 0.2)

This module wraps sentence-transformers to provide a clean interface for
encoding text into embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig


class Embedder:
    """Encodes text into dense vector embeddings.

    Uses a sentence-transformers model that runs entirely on CPU.
    The model is loaded once and reused for all encoding calls.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding model.

        The first time this runs, it will download the model (~80 MB)
        from HuggingFace. Subsequent runs use the cached version.
        """
        self.config = config
        print(f"  Loading embedding model: {config.model_name}...")
        self.model = SentenceTransformer(config.model_name, device="cpu")
        print(f"  Embedding model loaded. Output dimension: {config.dimension}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Convert a list of text strings into embeddings.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
            Each row is the embedding vector for the corresponding text.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Separated from batch encoding for clarity.
        Returns a 1D vector of shape (dimension,).
        """
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
