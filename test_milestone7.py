r"""
Milestone 7 — Controlled experiments comparing retrieval, chunking, and reranking.

Experiments:
  A. Retrieval method comparison (dense vs BM25 vs hybrid) — retrieval-only, no LLM
  B. Chunk size comparison (250 vs 500 vs 1000) — retrieval-only, requires re-ingestion
  C. Reranking impact — full RAG on 5 representative questions
  D. Full RAG comparison across retrieval methods — 5 questions with LLM

Run: .venv\Scripts\python.exe test_milestone7.py
"""

import sys
import time
import json
from pathlib import Path
from copy import deepcopy

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from config import PipelineConfig
from rag_pipeline import RAGPipeline
from benchmark import BENCHMARK
from evaluator import evaluate_single, evaluate_source_accuracy

# 5 representative questions for LLM experiments (to save time)
QUICK_BENCHMARK = [b for b in BENCHMARK if b["id"] in {"F1", "F4", "S1", "S2", "A1"}]

SAMPLE_DIR = Path(__file__).parent / "sample_docs"
ALL_DOCS = sorted([
    d for d in SAMPLE_DIR.glob("*.*")
    if d.suffix.lower() in {".txt", ".pdf", ".docx"}
])


def get_pipeline(chunk_size=500, chunk_overlap=50):
    """Create a pipeline with specific chunking config."""
    config = PipelineConfig()
    config.chunking.chunk_size = chunk_size
    config.chunking.chunk_overlap = chunk_overlap
    return RAGPipeline(config), config


def retrieval_only_eval(pipeline, questions, method):
    """Evaluate retrieval quality without calling the LLM (fast)."""
    pipeline.config.retriever.method = method
    pipeline.retriever.config.method = method
    pipeline.retriever._bm25_index = None
    pipeline.retriever._bm25_chunks = None

    results = []
    for entry in questions:
        retrieved = pipeline.retriever.retrieve(entry["question"])
        src_acc = evaluate_source_accuracy(
            [{"source_file": r["source_file"]} for r in retrieved],
            entry["expected_source"],
        )
        top_score = retrieved[0]["score"] if retrieved else 0
        results.append({
            "id": entry["id"],
            "method": method,
            "source_accuracy": src_acc,
            "top_score": round(top_score, 4),
            "top_source": retrieved[0]["source_file"] if retrieved else "N/A",
        })
    return results


def full_rag_eval(pipeline, questions, method, label=""):
    """Full RAG evaluation with LLM."""
    pipeline.config.retriever.method = method
    pipeline.retriever.config.method = method
    pipeline.retriever._bm25_index = None
    pipeline.retriever._bm25_chunks = None

    results = []
    for i, entry in enumerate(questions, 1):
        print(f"    [{i}/{len(questions)}] {entry['id']}: {entry['question'][:50]}...")
        result = pipeline.query(entry["question"])
        ev = evaluate_single(entry, result)
        results.append(ev)
        print(f"      Rel={ev.answer_relevance:.2f} Src={ev.source_accuracy:.2f} "
              f"Cite={ev.citation_present:.2f} Lat={ev.latency:.1f}s")
    return results


def experiment_a(pipeline):
    """Experiment A: Retrieval method comparison (no LLM, fast)."""
    print("\n" + "=" * 65)
    print("EXPERIMENT A: Retrieval Method Comparison (retrieval-only)")
    print("=" * 65)

    factual_qs = [b for b in BENCHMARK if b["category"] != "abstention"]
    all_results = {}

    for method in ["dense", "bm25", "hybrid"]:
        results = retrieval_only_eval(pipeline, factual_qs, method)
        avg_src = sum(r["source_accuracy"] for r in results) / len(results)
        avg_score = sum(r["top_score"] for r in results) / len(results)
        all_results[method] = {
            "avg_source_accuracy": round(avg_src, 4),
            "avg_top_score": round(avg_score, 4),
            "details": results,
        }
        print(f"\n  {method.upper():>8}: src_acc={avg_src:.3f}  avg_top_score={avg_score:.4f}")
        for r in results:
            print(f"    {r['id']}: score={r['top_score']:.4f} src={r['top_source']}")

    return all_results


def experiment_b(pipeline_500):
    """Experiment B: Chunk size comparison (requires re-ingestion).

    Must close the main pipeline before creating a new one because
    Qdrant's local mode locks the storage folder exclusively.
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT B: Chunk Size Comparison (retrieval-only)")
    print("=" * 65)

    factual_qs = [b for b in BENCHMARK if b["category"] != "abstention"]
    all_results = {}

    # Run 500 first while pipeline is still open
    print(f"\n  --- Chunk size: 500 (overlap: 50) ---")
    results = retrieval_only_eval(pipeline_500, factual_qs, "dense")
    avg_src = sum(r["source_accuracy"] for r in results) / len(results)
    avg_score = sum(r["top_score"] for r in results) / len(results)
    n_chunks = pipeline_500.vector_store.client.get_collection(
        pipeline_500.config.qdrant.collection_name
    ).points_count
    all_results[500] = {
        "num_chunks": n_chunks,
        "avg_source_accuracy": round(avg_src, 4),
        "avg_top_score": round(avg_score, 4),
    }
    print(f"  chunks={n_chunks}  src_acc={avg_src:.3f}  avg_score={avg_score:.4f}")

    # Close main pipeline to release Qdrant lock
    pipeline_500.close()

    # Test other chunk sizes
    for chunk_size, overlap in [(250, 25), (1000, 100)]:
        print(f"\n  --- Chunk size: {chunk_size} (overlap: {overlap}) ---")
        p, _ = get_pipeline(chunk_size, overlap)
        p.ingest(ALL_DOCS, recreate_collection=True)

        results = retrieval_only_eval(p, factual_qs, "dense")
        avg_src = sum(r["source_accuracy"] for r in results) / len(results)
        avg_score = sum(r["top_score"] for r in results) / len(results)
        n_chunks = p.vector_store.client.get_collection(
            p.config.qdrant.collection_name
        ).points_count

        all_results[chunk_size] = {
            "num_chunks": n_chunks,
            "avg_source_accuracy": round(avg_src, 4),
            "avg_top_score": round(avg_score, 4),
        }
        print(f"  chunks={n_chunks}  src_acc={avg_src:.3f}  avg_score={avg_score:.4f}")
        p.close()

    return all_results


def experiment_c(pipeline):
    """Experiment C: Reranking impact on retrieval quality."""
    print("\n" + "=" * 65)
    print("EXPERIMENT C: Cross-Encoder Reranking Impact")
    print("=" * 65)

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    factual_qs = [b for b in BENCHMARK if b["category"] != "abstention"]
    pipeline.config.retriever.method = "dense"
    pipeline.retriever.config.method = "dense"

    without_rerank = []
    with_rerank = []

    for entry in factual_qs:
        retrieved = pipeline.retriever.retrieve(entry["question"])

        # Without reranking
        src_before = evaluate_source_accuracy(
            [{"source_file": r["source_file"]} for r in retrieved],
            entry["expected_source"],
        )

        # With reranking
        pairs = [(entry["question"], r["text"]) for r in retrieved]
        scores = reranker.predict(pairs)
        reranked = sorted(
            zip(retrieved, scores), key=lambda x: x[1], reverse=True
        )
        reranked_chunks = [r[0] for r in reranked]

        src_after = evaluate_source_accuracy(
            [{"source_file": r["source_file"]} for r in reranked_chunks],
            entry["expected_source"],
        )

        without_rerank.append(src_before)
        with_rerank.append(src_after)

        changed = "→" if src_before != src_after else "="
        print(f"  {entry['id']}: before={src_before:.1f} {changed} after={src_after:.1f}")

    avg_before = sum(without_rerank) / len(without_rerank)
    avg_after = sum(with_rerank) / len(with_rerank)
    print(f"\n  Average source accuracy: {avg_before:.3f} → {avg_after:.3f}")

    return {
        "without_reranking": round(avg_before, 4),
        "with_reranking": round(avg_after, 4),
    }


def experiment_d(pipeline):
    """Experiment D: Full RAG comparison across retrieval methods (with LLM)."""
    print("\n" + "=" * 65)
    print("EXPERIMENT D: Full RAG Comparison (5 questions × 3 methods)")
    print("=" * 65)

    all_results = {}
    for method in ["dense", "bm25", "hybrid"]:
        print(f"\n  --- {method.upper()} ---")
        results = full_rag_eval(pipeline, QUICK_BENCHMARK, method)
        factual = [r for r in results if r.category != "abstention"]

        avg_rel = sum(r.answer_relevance for r in factual) / len(factual) if factual else 0
        avg_lat = sum(r.latency for r in results) / len(results)

        all_results[method] = {
            "avg_relevance": round(avg_rel, 4),
            "avg_latency": round(avg_lat, 1),
            "total_latency": round(sum(r.latency for r in results), 1),
        }

    return all_results


def main():
    print("=" * 65)
    print("MILESTONE 7 — CONTROLLED EXPERIMENTS")
    print("=" * 65)

    # Setup: ingest with default config
    pipeline, config = get_pipeline(500, 50)
    pipeline.ingest(ALL_DOCS, recreate_collection=True)

    # Run experiments
    results_a = experiment_a(pipeline)
    results_b = experiment_b(pipeline)  # This closes the pipeline

    # Recreate pipeline for remaining experiments (B closed it)
    pipeline, config = get_pipeline(500, 50)
    pipeline.ingest(ALL_DOCS, recreate_collection=True)

    results_c = experiment_c(pipeline)
    results_d = experiment_d(pipeline)

    # Save all results
    output = {
        "experiment_a_retrieval_methods": results_a,
        "experiment_b_chunk_sizes": results_b,
        "experiment_c_reranking": results_c,
        "experiment_d_full_rag": results_d,
        "config": {
            "model": config.llm.model,
            "documents": [d.name for d in ALL_DOCS],
        },
    }

    output_path = Path(__file__).parent / "data" / "experiment_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    # Final summary
    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)

    print("\n  A. Retrieval Methods (source accuracy):")
    for m, v in results_a.items():
        print(f"     {m:>8}: {v['avg_source_accuracy']:.3f}")

    print("\n  B. Chunk Sizes (dense retrieval):")
    for sz, v in results_b.items():
        print(f"     {sz:>5} chars: {v['num_chunks']} chunks, score={v['avg_top_score']:.4f}")

    print(f"\n  C. Reranking: {results_c['without_reranking']:.3f} → {results_c['with_reranking']:.3f}")

    print("\n  D. Full RAG (relevance / latency):")
    for m, v in results_d.items():
        print(f"     {m:>8}: rel={v['avg_relevance']:.3f}  lat={v['avg_latency']:.1f}s")

    print(f"\n  Results saved to: {output_path}")

    pipeline.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
