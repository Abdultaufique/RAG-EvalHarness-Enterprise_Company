"""
Chunker — splits documents into smaller pieces for embedding and retrieval.

WHY CHUNKING IS NEEDED:
Embedding models work best on shorter text passages (typically 1-3 paragraphs).
If you embed an entire 20-page document as one vector, the embedding becomes
a blurry average of everything — it won't match specific questions well.

By splitting documents into chunks, each chunk gets a focused embedding
that can be precisely matched to user questions.

CHUNKING STRATEGIES (we'll compare these in Milestone 7):
1. Fixed-size: Split every N characters with overlap. Simple and universal.
2. Structure-aware: Split on paragraph/section boundaries. Preserves meaning.
"""

from document_loader import DocumentChunk
from config import ChunkingConfig


def chunk_by_fixed_size(
    doc_chunks: list[DocumentChunk],
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    """Split document sections into fixed-size chunks with overlap.

    HOW IT WORKS:
    1. Take the text from each DocumentChunk (which represents a page or section).
    2. Slide a window of `chunk_size` characters across the text.
    3. Move forward by `chunk_size - chunk_overlap` characters each step.
    4. Each window becomes a new chunk, inheriting the source metadata.

    Example with chunk_size=100, overlap=20:
        Text: "AAAA...AAAA" (200 chars)
        Chunk 1: chars 0-99
        Chunk 2: chars 80-179  (overlaps by 20 with chunk 1)
        Chunk 3: chars 160-199 (overlaps by 20 with chunk 2)

    Args:
        doc_chunks: Raw document sections from the loader.
        config: Chunking settings (size and overlap).

    Returns:
        List of smaller DocumentChunks with chunk_index set.
    """
    result = []
    chunk_index = 0
    step = config.chunk_size - config.chunk_overlap

    if step <= 0:
        raise ValueError(
            f"chunk_overlap ({config.chunk_overlap}) must be less than "
            f"chunk_size ({config.chunk_size})"
        )

    for doc_chunk in doc_chunks:
        text = doc_chunk.text
        start = 0

        while start < len(text):
            end = start + config.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:  # Skip empty chunks
                result.append(DocumentChunk(
                    text=chunk_text,
                    source_file=doc_chunk.source_file,
                    page_number=doc_chunk.page_number,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            start += step

    return result


def chunk_documents(
    doc_chunks: list[DocumentChunk],
    config: ChunkingConfig,
    strategy: str = "fixed",
) -> list[DocumentChunk]:
    """Main entry point for chunking.

    Args:
        doc_chunks: Raw sections from the document loader.
        config: Chunking configuration.
        strategy: 'fixed' for fixed-size chunking (more strategies in M7).

    Returns:
        List of chunked DocumentChunks ready for embedding.
    """
    strategies = {
        "fixed": chunk_by_fixed_size,
    }

    chunker = strategies.get(strategy)
    if chunker is None:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Available: {', '.join(strategies.keys())}"
        )

    chunks = chunker(doc_chunks, config)
    print(f"  Chunked into {len(chunks)} chunks "
          f"(size={config.chunk_size}, overlap={config.chunk_overlap})")
    return chunks
