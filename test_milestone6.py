r"""
Milestone 6 — Run the benchmark evaluation.

Run: .venv\Scripts\python.exe test_milestone6.py
"""

import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from config import PipelineConfig
from rag_pipeline import RAGPipeline
from benchmark import BENCHMARK
from evaluator import evaluate_single, compute_summary


def main():
    print("=" * 65)
    print("MILESTONE 6 — BENCHMARK EVALUATION")
    print("=" * 65)

    # Ingest all docs
    config = PipelineConfig()
    pipeline = RAGPipeline(config)

    sample_dir = Path(__file__).parent / "sample_docs"
    all_docs = sorted(sample_dir.glob("*.*"))
    all_docs = [d for d in all_docs if d.suffix.lower() in {".txt", ".pdf", ".docx"}]

    print(f"\nIngesting {len(all_docs)} documents...")
    pipeline.ingest(all_docs, recreate_collection=True)

    # Run benchmark
    print(f"\n{'=' * 65}")
    print(f"Running {len(BENCHMARK)} benchmark questions (dense retrieval)")
    print(f"{'=' * 65}")

    results = []
    for i, entry in enumerate(BENCHMARK, 1):
        q = entry["question"]
        print(f"\n[{i}/{len(BENCHMARK)}] {entry['id']} ({entry['category']}) — {q}")

        start = time.time()
        pipeline_result = pipeline.query(q)
        elapsed = time.time() - start

        # Ensure latency is recorded
        if "latency_seconds" not in pipeline_result:
            pipeline_result["latency_seconds"] = round(elapsed, 2)

        ev = evaluate_single(entry, pipeline_result)
        results.append(ev)

        # Print per-question results
        print(f"  Relevance: {ev.answer_relevance:.2f}  "
              f"Source: {ev.source_accuracy:.2f}  "
              f"Citation: {ev.citation_present:.2f}  "
              f"Abstention: {ev.abstention_correct:.2f}  "
              f"Latency: {ev.latency:.1f}s")
        print(f"  Answer: {ev.answer[:120]}...")

    # Summary
    summary = compute_summary(results)

    print(f"\n{'=' * 65}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 65}")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k}: {v}")

    # Per-question table
    print(f"\n{'─' * 65}")
    print(f"{'ID':<5} {'Cat':<12} {'Rel':>5} {'Src':>5} {'Cite':>5} {'Abst':>5} {'Time':>6}")
    print(f"{'─' * 65}")
    for r in results:
        print(f"{r.question_id:<5} {r.category:<12} "
              f"{r.answer_relevance:>5.2f} {r.source_accuracy:>5.2f} "
              f"{r.citation_present:>5.2f} {r.abstention_correct:>5.2f} "
              f"{r.latency:>5.1f}s")

    # Save results to JSON
    output_path = Path(__file__).parent / "data" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "summary": summary,
        "config": {
            "model": config.llm.model,
            "retrieval_method": config.retriever.method,
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "top_k": config.retriever.top_k,
        },
        "results": [
            {
                "id": r.question_id,
                "question": r.question,
                "category": r.category,
                "answer": r.answer,
                "answer_relevance": r.answer_relevance,
                "source_accuracy": r.source_accuracy,
                "citation_present": r.citation_present,
                "abstention_correct": r.abstention_correct,
                "latency": r.latency,
                "top_score": r.top_score,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    pipeline.close()
    print("Done!")


if __name__ == "__main__":
    main()
