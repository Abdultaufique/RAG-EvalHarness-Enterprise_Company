"""
Evaluator — Metrics for scoring RAG pipeline quality.

Metrics computed WITHOUT an external LLM judge (deterministic + fast):

1. Answer Relevance   — Do expected keywords appear in the answer?
2. Source Accuracy     — Was the correct document retrieved?
3. Citation Present    — Did the answer cite specific sources?
4. Abstention Correct  — Did it refuse when it should (and not when it shouldn't)?
5. Recall@K            — Was the relevant source in the top-K results?
6. MRR                 — Mean Reciprocal Rank of the correct source.
7. nDCG@K              — Normalized Discounted Cumulative Gain.
8. Faithfulness        — RAGAS-style: is the answer grounded in retrieved chunks?
9. Latency             — Response time in seconds.
"""

import math

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Evaluation result for a single benchmark question."""
    question_id: str
    question: str
    category: str
    difficulty: str

    # The raw pipeline output
    answer: str = ""
    latency: float = 0.0
    retrieval_method: str = ""
    top_score: float = 0.0
    retrieved_sources: list = field(default_factory=list)

    # Metric scores (0.0 to 1.0)
    answer_relevance: float = 0.0
    source_accuracy: float = 0.0
    citation_present: float = 0.0
    abstention_correct: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    faithfulness: float = 0.0


def evaluate_answer_relevance(answer: str, expected_keywords: list[str]) -> float:
    """What fraction of expected keywords appear in the answer?

    Case-insensitive substring match. A score of 1.0 means all
    expected keywords were found.
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def evaluate_source_accuracy(
    retrieved_sources: list[dict],
    expected_source: str | None,
) -> float:
    """Was the expected source document among the retrieved chunks?

    Returns 1.0 if the expected source is the top-1 result,
    0.5 if it's in top-5, 0.0 otherwise.
    For abstention questions (expected_source=None), returns 1.0
    (source accuracy is N/A).
    """
    if expected_source is None:
        return 1.0  # Not applicable for abstention

    if not retrieved_sources:
        return 0.0

    source_files = [r.get("source_file", "") for r in retrieved_sources]

    if source_files and source_files[0] == expected_source:
        return 1.0
    if expected_source in source_files:
        return 0.5
    return 0.0


def evaluate_citation_present(answer: str, category: str) -> float:
    """Does the answer contain source citations?

    Checks for patterns like [Source: ...] or (Source: ...) or
    references to filenames. For abstention, no citations = correct.
    """
    if category == "abstention":
        # Abstention answers SHOULD NOT have citations
        has_citation = "[source:" in answer.lower() or ".txt" in answer.lower()
        return 1.0 if not has_citation else 0.5

    # Non-abstention answers SHOULD have citations
    indicators = ["[source:", "chunk #", ".txt", ".pdf", ".docx"]
    found = any(ind in answer.lower() for ind in indicators)
    return 1.0 if found else 0.0


def evaluate_abstention(answer: str, category: str) -> float:
    """Did the model correctly abstain (or not abstain)?

    For abstention questions: should refuse to answer → 1.0
    For non-abstention: should provide an answer → 1.0
    """
    refusal_signals = [
        "cannot answer", "can't answer", "cannot provide",
        "don't have", "do not have", "no information",
        "not enough information", "insufficient",
        "not available", "not found in",
        "unable to answer", "beyond the scope",
    ]
    is_refusal = any(sig in answer.lower() for sig in refusal_signals)

    if category == "abstention":
        return 1.0 if is_refusal else 0.0
    else:
        return 1.0 if not is_refusal else 0.0


def evaluate_recall_at_k(
    retrieved_sources: list[dict],
    expected_source: str | None,
    k: int = 5,
) -> float:
    """Recall@K — Was the relevant source in the top-K results?

    Standard information retrieval metric.
    Returns 1.0 if expected source appears in top-K, 0.0 otherwise.
    """
    if expected_source is None:
        return 1.0  # N/A for abstention
    source_files = [r.get("source_file", "") for r in retrieved_sources[:k]]
    return 1.0 if expected_source in source_files else 0.0


def evaluate_mrr(
    retrieved_sources: list[dict],
    expected_source: str | None,
) -> float:
    """Mean Reciprocal Rank — How high is the correct source ranked?

    MRR = 1/rank where rank is the position of the first relevant result.
    Higher is better: 1.0 = first result, 0.5 = second, 0.33 = third.
    """
    if expected_source is None:
        return 1.0  # N/A for abstention
    for i, r in enumerate(retrieved_sources):
        if r.get("source_file", "") == expected_source:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ndcg_at_k(
    retrieved_sources: list[dict],
    expected_source: str | None,
    k: int = 5,
) -> float:
    """nDCG@K — Normalized Discounted Cumulative Gain.

    Measures ranking quality: rewards placing the right document higher.
    Uses binary relevance: 1 if source matches, 0 otherwise.
    """
    if expected_source is None:
        return 1.0  # N/A for abstention

    # DCG: sum of rel_i / log2(i+2) for i in 0..k-1
    dcg = 0.0
    for i, r in enumerate(retrieved_sources[:k]):
        rel = 1.0 if r.get("source_file", "") == expected_source else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG: best possible ranking (relevant doc at position 0)
    idcg = 1.0 / math.log2(2)  # = 1.0

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: list[dict],
    category: str,
) -> float:
    """RAGAS-style faithfulness — Is the answer grounded in retrieved chunks?

    Checks what fraction of answer sentences can be traced back to
    the retrieved context. Uses substring overlap as a proxy
    (no LLM judge needed).

    Score: fraction of answer sentences that overlap with retrieved text.
    """
    if category == "abstention":
        return 1.0  # Abstention answers are inherently faithful

    if not answer or not retrieved_chunks:
        return 0.0

    # Build combined context from retrieved chunks
    context = " ".join(c.get("text", "") for c in retrieved_chunks).lower()

    # Split answer into sentences
    sentences = [s.strip() for s in answer.replace("\n", ". ").split(".") if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    # Check each sentence for overlap with context
    grounded = 0
    for sent in sentences:
        words = sent.lower().split()
        # A sentence is grounded if >40% of its content words appear in context
        content_words = [w for w in words if len(w) > 3]  # Skip short words
        if not content_words:
            grounded += 1
            continue
        overlap = sum(1 for w in content_words if w in context)
        if overlap / len(content_words) >= 0.4:
            grounded += 1

    return round(grounded / len(sentences), 4)


def evaluate_single(benchmark_entry: dict, pipeline_result: dict) -> EvalResult:
    """Run all metrics on a single question's result."""
    answer = pipeline_result.get("answer", "")
    citations = pipeline_result.get("citations", [])

    result = EvalResult(
        question_id=benchmark_entry["id"],
        question=benchmark_entry["question"],
        category=benchmark_entry["category"],
        difficulty=benchmark_entry["difficulty"],
        answer=answer,
        latency=pipeline_result.get("latency_seconds", 0),
        retrieval_method=pipeline_result.get("retrieval_method", ""),
        top_score=citations[0]["score"] if citations else 0.0,
        retrieved_sources=citations,
    )

    result.answer_relevance = evaluate_answer_relevance(
        answer, benchmark_entry["expected_answer_keywords"]
    )
    result.source_accuracy = evaluate_source_accuracy(
        citations, benchmark_entry["expected_source"]
    )
    result.citation_present = evaluate_citation_present(
        answer, benchmark_entry["category"]
    )
    result.abstention_correct = evaluate_abstention(
        answer, benchmark_entry["category"]
    )
    result.recall_at_k = evaluate_recall_at_k(
        citations, benchmark_entry["expected_source"]
    )
    result.mrr = evaluate_mrr(
        citations, benchmark_entry["expected_source"]
    )
    result.ndcg_at_k = evaluate_ndcg_at_k(
        citations, benchmark_entry["expected_source"]
    )
    result.faithfulness = evaluate_faithfulness(
        answer, citations, benchmark_entry["category"]
    )

    return result


def compute_summary(results: list[EvalResult]) -> dict:
    """Compute aggregate scores across all results."""
    n = len(results)
    if n == 0:
        return {}

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0

    factual = [r for r in results if r.category != "abstention"]
    abstention = [r for r in results if r.category == "abstention"]

    return {
        "total_questions": n,
        "avg_answer_relevance": avg([r.answer_relevance for r in factual]),
        "avg_source_accuracy": avg([r.source_accuracy for r in factual]),
        "avg_citation_present": avg([r.citation_present for r in factual]),
        "abstention_accuracy": avg([r.abstention_correct for r in abstention]) if abstention else None,
        "overall_abstention_correct": avg([r.abstention_correct for r in results]),
        "avg_recall_at_k": avg([r.recall_at_k for r in factual]),
        "avg_mrr": avg([r.mrr for r in factual]),
        "avg_ndcg_at_k": avg([r.ndcg_at_k for r in factual]),
        "avg_faithfulness": avg([r.faithfulness for r in factual]),
        "avg_latency_seconds": avg([r.latency for r in results]),
        "avg_top_score": avg([r.top_score for r in factual]),
        "total_latency": round(sum(r.latency for r in results), 1),
    }
