"""
RAG Pipeline — orchestrates the full Retrieval-Augmented Generation flow.

This is the central module that ties everything together:

  Documents → Loader → Chunker → Embedder → Vector Store → Retriever → Generator → Answer

WHY AN ORCHESTRATOR:
Instead of calling each module manually, the pipeline provides two
clean methods:
  1. ingest()   — process documents and store them
  2. query()    — answer a question using stored documents

This makes it easy to use from the API (Milestone 4), the CLI, and
experiment scripts (Milestone 7).
"""

from pathlib import Path

from config import PipelineConfig
from document_loader import load_document
from chunker import chunk_documents
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator


class RAGPipeline:
    """End-to-end RAG pipeline: ingest documents, then answer questions."""

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize all pipeline components.

        Args:
            config: Full pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()

        print("\n=== Initializing RAG Pipeline ===")

        # Initialize components
        self.embedder = Embedder(self.config.embedding)
        self.vector_store = VectorStore(self.config.qdrant, self.config.embedding)
        self.retriever = Retriever(
            self.config.retriever, self.embedder, self.vector_store
        )
        self.generator = Generator(self.config.llm)

        print("=== Pipeline Ready ===\n")

    def ingest(
        self,
        file_paths: list[str | Path],
        recreate_collection: bool = True,
    ) -> dict:
        """Process and store documents.

        This runs the full ingestion pipeline:
        1. Load each document (extract text + metadata).
        2. Chunk all documents into smaller pieces.
        3. Generate embeddings for each chunk.
        4. Store chunks + embeddings in Qdrant.

        Args:
            file_paths: List of paths to documents to process.
            recreate_collection: If True, wipe existing data first.
                                 Set to True during development.

        Returns:
            Dict with ingestion statistics.
        """
        print("--- Step 1: Loading Documents ---")
        all_doc_sections = []
        for path in file_paths:
            sections = load_document(path)
            all_doc_sections.extend(sections)

        print(f"\n--- Step 2: Chunking ({len(all_doc_sections)} sections) ---")
        chunks = chunk_documents(all_doc_sections, self.config.chunking)

        print(f"\n--- Step 3: Generating Embeddings ({len(chunks)} chunks) ---")
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        print(f"  Generated {len(embeddings)} embeddings "
              f"(shape: {embeddings.shape})")

        print(f"\n--- Step 4: Storing in Qdrant ---")
        self.vector_store.create_collection(recreate=recreate_collection)
        self.vector_store.add_chunks(chunks, embeddings)

        stats = {
            "documents_loaded": len(file_paths),
            "sections_extracted": len(all_doc_sections),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
            "embedding_dimension": embeddings.shape[1],
        }
        print(f"\n--- Ingestion Complete ---")
        print(f"  Stats: {stats}")
        return stats

    def query(self, question: str) -> dict:
        """Answer a question using the stored documents.

        This runs the full query pipeline:
        1. Retrieve relevant chunks using the configured method.
        2. Send the question + chunks to the LLM.
        3. Return the answer with citations.

        Args:
            question: Natural language question from the user.

        Returns:
            Dict with: answer, model, citations, latency_seconds, retrieval_method.
        """
        print(f"\n--- Query: \"{question}\" ---")

        # Step 1: Retrieve
        print(f"  Retrieval method: {self.config.retriever.method}")
        retrieved = self.retriever.retrieve(question)
        print(f"  Retrieved {len(retrieved)} chunks")
        for i, r in enumerate(retrieved):
            score_str = f"{r['score']:.4f}" if r['score'] is not None else "N/A"
            print(f"    [{i+1}] score={score_str} | "
                  f"{r['source_file']} chunk#{r['chunk_index']}")

        # Step 2: Generate answer
        print(f"  Generating answer with {self.config.llm.model}...")
        result = self.generator.generate(question, retrieved)
        result["retrieval_method"] = self.config.retriever.method
        print(f"  Answer generated in {result['latency_seconds']}s")

        return result

    def close(self) -> None:
        """Clean up resources."""
        self.vector_store.close()
