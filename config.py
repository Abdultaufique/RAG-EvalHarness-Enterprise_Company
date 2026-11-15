"""
Configuration for the RAG pipeline.

WHY THIS FILE EXISTS:
Instead of scattering magic numbers and settings throughout the code,
we keep all configuration in one place. This makes it easy to change
parameters for experiments later (Milestone 7) without modifying
multiple files.
"""

from dataclasses import dataclass, field
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()


@dataclass
class ChunkingConfig:
    """Settings for how documents are split into chunks.

    chunk_size: Number of characters per chunk.
        - Too small → chunks lack context, retrieval returns fragments.
        - Too large → embeddings become diluted, retrieval less precise.
        - 500 chars is a reasonable default for dense retrieval.

    chunk_overlap: Number of overlapping characters between consecutive chunks.
        - Overlap prevents information at chunk boundaries from being lost.
        - 50 chars (~1 sentence) is a safe default.
    """
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class EmbeddingConfig:
    """Settings for the embedding model.

    model_name: HuggingFace model identifier for sentence-transformers.
        - 'all-MiniLM-L6-v2' produces 384-dimensional vectors.
        - Small (~80 MB), fast on CPU, widely used in RAG systems.

    dimension: Output vector size. Must match the model's output.
    """
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384


@dataclass
class QdrantConfig:
    """Settings for the Qdrant vector store.

    path: Where Qdrant stores data on disk.
        - Using local file storage (no server needed for prototyping).

    collection_name: Name of the vector collection in Qdrant.
    """
    path: str = str(PROJECT_ROOT / "data" / "qdrant_store")
    collection_name: str = "documents"


@dataclass
class RetrieverConfig:
    """Settings for retrieval.

    top_k: Number of chunks to retrieve.
        - More chunks → more context but higher latency and noise.
        - 5 is a standard starting point.

    method: Retrieval strategy — 'dense', 'bm25', or 'hybrid'.

    hybrid_alpha: Weight for dense vs BM25 in hybrid mode.
        - 0.0 = pure BM25, 1.0 = pure dense, 0.5 = equal blend.
    """
    top_k: int = 5
    method: str = "dense"  # 'dense', 'bm25', or 'hybrid'
    hybrid_alpha: float = 0.7  # Weight for dense scores in hybrid


@dataclass
class RerankConfig:
    """Settings for cross-encoder reranking.

    enabled: Whether to rerank retrieved results.
    model_name: HuggingFace cross-encoder model.
    top_k: How many results to keep after reranking.
    """
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 3


@dataclass
class LLMConfig:
    """Settings for the language model (via Ollama).

    model: Ollama model name (must be pulled first).
    base_url: Ollama API endpoint.
    temperature: Controls randomness.
        - 0.0 = deterministic, 1.0 = creative.
        - For RAG, low temperature (0.1) keeps answers grounded.
    max_tokens: Maximum tokens in the generated answer.
    """
    model: str = "llama3.1:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass
class PipelineConfig:
    """Top-level configuration that bundles all settings."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
