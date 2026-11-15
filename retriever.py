"""
Retriever — implements Dense, BM25, and Hybrid retrieval strategies.

WHY MULTIPLE RETRIEVAL METHODS:

1. DENSE RETRIEVAL (embedding-based):
   - Matches based on semantic meaning.
   - "What causes global warming?" matches "greenhouse gas emissions" even
     though the words are different.
   - Weakness: Can miss exact keyword matches.

2. BM25 (keyword-based):
   - A classic information retrieval algorithm based on term frequency.
   - Great at matching specific terms, names, numbers, and acronyms.
   - "GPT-4" will exactly match documents mentioning "GPT-4".
   - Weakness: Misses semantic similarity ("car" won't match "automobile").

3. HYBRID (dense + BM25):
   - Combines both approaches using a weighted score.
   - Gets the best of both worlds: semantic understanding + keyword precision.
   - The alpha parameter controls the balance (0.7 = 70% dense, 30% BM25).
"""

import numpy as np
from rank_bm25 import BM25Okapi

from config import RetrieverConfig
from embedder import Embedder
from vector_store import VectorStore


class Retriever:
    """Retrieves the most relevant document chunks for a given query."""

    def __init__(
        self,
        config: RetrieverConfig,
        embedder: Embedder,
        vector_store: VectorStore,
    ):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self._bm25_index = None
        self._bm25_chunks = None

    def _build_bm25_index(self) -> None:
        """Build the BM25 index from all stored chunks.

        BM25 needs all documents in memory to compute term frequencies.
        We tokenize each chunk's text by splitting on whitespace (simple
        but effective for English text).
        """
        if self._bm25_index is not None:
            return  # Already built

        print("  Building BM25 index...")
        self._bm25_chunks = self.vector_store.get_all_chunks()

        if not self._bm25_chunks:
            raise ValueError("No chunks found in vector store. Ingest documents first.")

        # Tokenize: lowercase and split on whitespace
        tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in self._bm25_chunks
        ]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        print(f"  BM25 index built with {len(self._bm25_chunks)} documents")

    def retrieve_dense(self, query: str) -> list[dict]:
        """Retrieve using dense (embedding-based) vector search.

        1. Encode the query into an embedding vector.
        2. Search Qdrant for the closest chunk vectors.
        3. Return the top-K results with scores.
        """
        query_embedding = self.embedder.encode_query(query)
        results = self.vector_store.search(query_embedding, top_k=self.config.top_k)
        return results

    def retrieve_bm25(self, query: str) -> list[dict]:
        """Retrieve using BM25 keyword-based search.

        1. Tokenize the query the same way as the corpus.
        2. Score every chunk against the query using BM25.
        3. Return the top-K results.
        """
        self._build_bm25_index()

        query_tokens = query.lower().split()
        scores = self._bm25_index.get_scores(query_tokens)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:self.config.top_k]

        results = []
        for idx in top_indices:
            chunk = self._bm25_chunks[idx]
            results.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "page_number": chunk.get("page_number", -1),
                "chunk_index": chunk.get("chunk_index", -1),
                "score": float(scores[idx]),
            })
        return results

    def retrieve_hybrid(self, query: str) -> list[dict]:
        """Retrieve using a weighted combination of dense and BM25 scores.

        HOW HYBRID SCORING WORKS:
        1. Get dense results (scores are typically 0.0 to 1.0 cosine similarity).
        2. Get BM25 results (scores can vary widely, e.g., 0 to 20+).
        3. Normalize BM25 scores to [0, 1] range (min-max normalization).
        4. Combine: final_score = alpha * dense_score + (1 - alpha) * bm25_score
        5. Sort by final score and return top-K.

        The alpha parameter (from config) controls the balance:
          - alpha=0.7 → 70% semantic, 30% keyword (good default)
          - alpha=1.0 → pure dense
          - alpha=0.0 → pure BM25
        """
        # Get both sets of results with extra candidates for better fusion
        original_top_k = self.config.top_k
        self.config.top_k = original_top_k * 3  # Fetch more for better overlap
        dense_results = self.retrieve_dense(query)
        bm25_results = self.retrieve_bm25(query)
        self.config.top_k = original_top_k

        # Build a unified map keyed by (source_file, chunk_index)
        combined = {}

        for r in dense_results:
            key = (r["source_file"], r["chunk_index"])
            combined[key] = {
                **r,
                "dense_score": r["score"],
                "bm25_score": 0.0,
            }

        # Normalize BM25 scores to [0, 1]
        bm25_scores_raw = [r["score"] for r in bm25_results]
        max_bm25 = max(bm25_scores_raw) if bm25_scores_raw else 1.0
        min_bm25 = min(bm25_scores_raw) if bm25_scores_raw else 0.0
        score_range = max_bm25 - min_bm25

        for r in bm25_results:
            key = (r["source_file"], r["chunk_index"])
            norm_score = (r["score"] - min_bm25) / score_range if score_range > 0 else 0.0

            if key in combined:
                combined[key]["bm25_score"] = norm_score
            else:
                combined[key] = {
                    **r,
                    "dense_score": 0.0,
                    "bm25_score": norm_score,
                }

        # Compute hybrid score
        alpha = self.config.hybrid_alpha
        for key in combined:
            combined[key]["score"] = (
                alpha * combined[key]["dense_score"]
                + (1 - alpha) * combined[key]["bm25_score"]
            )

        # Sort by hybrid score and return top-K
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:original_top_k]

    def retrieve(self, query: str) -> list[dict]:
        """Main entry point — dispatches to the configured retrieval method.

        Args:
            query: The user's natural language question.

        Returns:
            List of dicts, each with: text, source_file, page_number,
            chunk_index, score.
        """
        methods = {
            "dense": self.retrieve_dense,
            "bm25": self.retrieve_bm25,
            "hybrid": self.retrieve_hybrid,
        }

        method_fn = methods.get(self.config.method)
        if method_fn is None:
            raise ValueError(
                f"Unknown retrieval method: '{self.config.method}'. "
                f"Available: {', '.join(methods.keys())}"
            )

        results = method_fn(query)
        return results
