"""
Main -- CLI entry point for testing the RAG pipeline.

RUN THIS TO TEST:
    cd C:\\Users\\abdul\\.gemini\\antigravity-ide\\scratch\\rag-enterprise
    .venv\\Scripts\\python.exe main.py

This script:
1. Ingests the 3 sample documents.
2. Asks 5 test questions spanning different documents.
3. Prints the answers with citations.
4. Tests edge cases (cross-document, abstention).
"""

import sys
import time
from pathlib import Path

# Fix Windows console encoding — cp1252 can't handle Unicode symbols
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from config import PipelineConfig
from rag_pipeline import RAGPipeline


def main():
    # -- Configuration --
    config = PipelineConfig()  # Uses all defaults

    # -- Sample Documents --
    sample_dir = Path(__file__).parent / "sample_docs"
    documents = list(sample_dir.glob("*.txt"))

    if not documents:
        print("ERROR: No sample documents found in sample_docs/")
        sys.exit(1)

    print(f"Found {len(documents)} sample documents:")
    for doc in documents:
        print(f"  - {doc.name}")

    # -- Initialize Pipeline --
    pipeline = RAGPipeline(config)

    # -- Ingest Documents --
    print("\n" + "=" * 60)
    print("PHASE 1: DOCUMENT INGESTION")
    print("=" * 60)
    ingest_start = time.time()
    stats = pipeline.ingest(documents, recreate_collection=True)
    ingest_time = time.time() - ingest_start
    print(f"\nTotal ingestion time: {ingest_time:.1f}s")

    # -- Test Questions --
    print("\n" + "=" * 60)
    print("PHASE 2: QUESTION ANSWERING")
    print("=" * 60)

    test_questions = [
        # Q1: Should retrieve from AI document
        "What are the three types of machine learning?",

        # Q2: Should retrieve from Climate document
        "How much has global temperature increased since pre-industrial levels?",

        # Q3: Should retrieve from Space document
        "Who was the first human in space and when?",

        # Q4: Cross-document question (tests retrieval breadth)
        "What role does AI play in addressing climate change?",

        # Q5: Abstention test (no relevant info in documents)
        "What is the recipe for chocolate chip cookies?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'-' * 60}")
        print(f"QUESTION {i}/{len(test_questions)}")
        print(f"{'-' * 60}")

        result = pipeline.query(question)

        print(f"\n  ANSWER:\n")
        # Indent the answer for readability
        for line in result["answer"].split("\n"):
            print(f"    {line}")

        print(f"\n  MODEL: {result['model']}")
        print(f"  LATENCY: {result['latency_seconds']}s")
        print(f"  RETRIEVAL: {result['retrieval_method']}")
        print(f"  SOURCES USED: {len(result['citations'])} chunks")
        for j, cite in enumerate(result["citations"]):
            print(f"    [{j+1}] {cite['source_file']} "
                  f"(chunk #{cite['chunk_index']}, score: {cite['score']:.4f})")

    # -- Summary --
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Documents ingested: {stats['documents_loaded']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Questions answered: {len(test_questions)}")
    print(f"  Model: {config.llm.model}")
    print(f"  Retrieval method: {config.retriever.method}")
    print(f"  Chunk size: {config.chunking.chunk_size}")
    print(f"  Chunk overlap: {config.chunking.chunk_overlap}")
    print(f"  Top-K: {config.retriever.top_k}")

    # Cleanup
    pipeline.close()
    print("\nDone! Pipeline closed.")


if __name__ == "__main__":
    main()
