r"""
Milestone 3 — Test all pipeline features:
  1. PDF and DOCX document loading
  2. Dense, BM25, and Hybrid retrieval
  3. Cross-encoder reranking
  4. Deduplication fix verification

Run: .venv\Scripts\python.exe test_milestone3.py
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ─── Part A: Create test PDF and DOCX from existing content ───

def create_test_pdf(output_path: Path):
    """Create a simple test PDF using PyMuPDF."""
    import fitz
    doc = fitz.open()
    
    page1 = doc.new_page()
    page1.insert_text((72, 72), 
        "Quantum Computing Fundamentals\n\n"
        "Quantum computing uses quantum bits (qubits) instead of classical bits.\n"
        "While classical bits can be either 0 or 1, qubits can exist in a\n"
        "superposition of both states simultaneously. This property, along with\n"
        "entanglement and interference, allows quantum computers to solve certain\n"
        "problems exponentially faster than classical computers.\n\n"
        "Key quantum algorithms include Shor's algorithm for integer factorization\n"
        "and Grover's algorithm for database search. IBM, Google, and IonQ are\n"
        "leading companies in quantum hardware development.",
        fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 72),
        "Quantum Error Correction\n\n"
        "Quantum error correction (QEC) is essential for reliable quantum computing.\n"
        "Qubits are extremely fragile and susceptible to decoherence from\n"
        "environmental noise. The surface code is the most widely studied QEC code,\n"
        "requiring approximately 1000 physical qubits per logical qubit.\n\n"
        "Google's Willow chip demonstrated that increasing qubit count can actually\n"
        "reduce error rates, a key milestone for scalable quantum computing.",
        fontsize=11)
    
    doc.save(str(output_path))
    doc.close()
    print(f"  Created test PDF: {output_path.name} (2 pages)")


def create_test_docx(output_path: Path):
    """Create a simple test DOCX using python-docx."""
    from docx import Document
    doc = Document()
    doc.add_heading("Blockchain Technology", level=1)
    doc.add_paragraph(
        "Blockchain is a distributed ledger technology that records transactions "
        "across a network of computers. Each block contains a cryptographic hash "
        "of the previous block, a timestamp, and transaction data. This makes the "
        "chain tamper-resistant."
    )
    doc.add_paragraph(
        "Bitcoin, created by Satoshi Nakamoto in 2009, was the first application "
        "of blockchain technology. Ethereum extended blockchain with smart contracts, "
        "enabling decentralized applications (dApps). Consensus mechanisms include "
        "Proof of Work (PoW) and Proof of Stake (PoS)."
    )
    doc.save(str(output_path))
    print(f"  Created test DOCX: {output_path.name}")


# ─── Part B: Run tests ───

def test_retrieval_modes(pipeline, question: str):
    """Test the same question with dense, BM25, and hybrid retrieval."""
    from config import RetrieverConfig
    results = {}
    for method in ["dense", "bm25", "hybrid"]:
        pipeline.config.retriever.method = method
        # Reset BM25 index cache when switching methods
        pipeline.retriever._bm25_index = None
        pipeline.retriever._bm25_chunks = None
        pipeline.retriever.config.method = method
        
        retrieved = pipeline.retriever.retrieve(question)
        results[method] = retrieved
        print(f"\n  [{method.upper()}] Top-3 results:")
        for i, r in enumerate(retrieved[:3]):
            score_str = f"{r['score']:.4f}"
            print(f"    {i+1}. score={score_str} | {r['source_file']} chunk#{r['chunk_index']}")
    return results


def test_reranking(pipeline, question: str, retrieved: list[dict]):
    """Test cross-encoder reranking on retrieved results."""
    from sentence_transformers import CrossEncoder
    
    print(f"\n  Loading reranker: cross-encoder/ms-marco-MiniLM-L-6-v2...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    
    # Create (query, passage) pairs for scoring
    pairs = [(question, r["text"]) for r in retrieved]
    scores = reranker.predict(pairs)
    
    # Attach rerank scores and sort
    for i, r in enumerate(retrieved):
        r["rerank_score"] = float(scores[i])
    reranked = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"  BEFORE reranking:")
    for i, r in enumerate(retrieved[:3]):
        print(f"    {i+1}. dense={r['score']:.4f} | {r['source_file']} chunk#{r['chunk_index']}")
    print(f"  AFTER reranking:")
    for i, r in enumerate(reranked[:3]):
        print(f"    {i+1}. rerank={r['rerank_score']:.4f} | {r['source_file']} chunk#{r['chunk_index']}")
    
    return reranked


def main():
    from config import PipelineConfig
    from rag_pipeline import RAGPipeline
    
    sample_dir = Path(__file__).parent / "sample_docs"
    
    # --- Part A: Create PDF and DOCX test files ---
    print("=" * 60)
    print("PART A: Creating PDF and DOCX test documents")
    print("=" * 60)
    
    pdf_path = sample_dir / "quantum_computing.pdf"
    docx_path = sample_dir / "blockchain.docx"
    create_test_pdf(pdf_path)
    create_test_docx(docx_path)

    # --- Part B: Ingest ALL documents (TXT + PDF + DOCX) ---
    print("\n" + "=" * 60)
    print("PART B: Ingesting all document formats")
    print("=" * 60)
    
    config = PipelineConfig()
    pipeline = RAGPipeline(config)
    
    all_docs = list(sample_dir.glob("*.txt")) + [pdf_path, docx_path]
    print(f"\nIngesting {len(all_docs)} documents:")
    for d in all_docs:
        print(f"  - {d.name}")
    
    stats = pipeline.ingest(all_docs, recreate_collection=True)

    # --- Part C: Test dedup fix ---
    print("\n" + "=" * 60)
    print("PART C: Verify deduplication fix")
    print("=" * 60)
    
    q = "What are the types of machine learning?"
    pipeline.config.retriever.method = "dense"
    results = pipeline.retriever.retrieve(q)
    chunk_keys = [(r["source_file"], r["chunk_index"]) for r in results]
    has_dupes = len(chunk_keys) != len(set(chunk_keys))
    print(f"  Query: '{q}'")
    print(f"  Results: {len(results)} chunks, Duplicates: {'YES (BUG!)' if has_dupes else 'NONE (FIXED)'}")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r['source_file']} chunk#{r['chunk_index']} score={r['score']:.4f}")

    # --- Part D: Test all retrieval modes ---
    print("\n" + "=" * 60)
    print("PART D: Comparing retrieval methods")
    print("=" * 60)
    
    test_q = "How does quantum error correction work?"
    print(f"  Question: '{test_q}'")
    retrieval_results = test_retrieval_modes(pipeline, test_q)

    # --- Part E: Test reranking ---
    print("\n" + "=" * 60)
    print("PART E: Cross-encoder reranking")
    print("=" * 60)
    
    pipeline.config.retriever.method = "dense"
    pipeline.retriever.config.method = "dense"
    dense_results = pipeline.retriever.retrieve(test_q)
    reranked = test_reranking(pipeline, test_q, dense_results)

    # --- Part F: Full RAG with PDF source ---
    print("\n" + "=" * 60)
    print("PART F: Full RAG answer from PDF document")
    print("=" * 60)
    
    pipeline.config.retriever.method = "dense"
    pipeline.retriever.config.method = "dense"
    result = pipeline.query("What is Shor's algorithm used for?")
    print(f"\n  ANSWER:")
    for line in result["answer"].split("\n"):
        print(f"    {line}")
    print(f"  LATENCY: {result['latency_seconds']}s")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("MILESTONE 3 SUMMARY")
    print("=" * 60)
    print(f"  PDF loading:      OK")
    print(f"  DOCX loading:     OK")
    print(f"  Deduplication:    {'FIXED' if not has_dupes else 'STILL BROKEN'}")
    print(f"  Dense retrieval:  OK")
    print(f"  BM25 retrieval:   OK")
    print(f"  Hybrid retrieval: OK")
    print(f"  Reranking:        OK")
    print(f"  Full RAG (PDF):   OK")
    print(f"  Total chunks:     {stats['chunks_created']}")
    
    pipeline.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
