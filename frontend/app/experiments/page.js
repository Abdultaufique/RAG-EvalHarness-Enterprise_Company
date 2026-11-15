"use client";

import { useState, useEffect } from "react";

export default function ExperimentsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("evaluation");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/experiments")
      .then((r) => {
        if (!r.ok) throw new Error("No experiment data found");
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="exp-page"><p className="exp-loading">Loading experiments...</p></div>;
  if (error) return <div className="exp-page"><p className="exp-error">⚠ {error}. Run the benchmark scripts first.</p></div>;

  const evaluation = data?.evaluation;
  const experiments = data?.experiments;

  return (
    <div className="exp-page">
      <div className="exp-header">
        <a href="/" className="exp-back">← Back to Chat</a>
        <h1>Research Experiments</h1>
        <p>Systematic evaluation of retrieval methods, chunking strategies, and reranking.</p>
      </div>

      <div className="exp-tabs">
        {evaluation && <button className={tab === "evaluation" ? "active" : ""} onClick={() => setTab("evaluation")}>Benchmark</button>}
        {experiments && <button className={tab === "retrieval" ? "active" : ""} onClick={() => setTab("retrieval")}>Retrieval Methods</button>}
        {experiments && <button className={tab === "chunking" ? "active" : ""} onClick={() => setTab("chunking")}>Chunk Sizes</button>}
        {experiments && <button className={tab === "reranking" ? "active" : ""} onClick={() => setTab("reranking")}>Reranking</button>}
        {experiments && <button className={tab === "fullrag" ? "active" : ""} onClick={() => setTab("fullrag")}>Full RAG</button>}
      </div>

      {/* ── Benchmark Results ── */}
      {tab === "evaluation" && evaluation && (
        <div className="exp-section">
          <h2>Benchmark Evaluation ({evaluation.summary.total_questions} questions)</h2>
          <div className="exp-metrics-grid">
            {Object.entries(evaluation.summary).filter(([k]) => k.startsWith("avg_") || k.includes("accuracy")).map(([k, v]) => (
              <div key={k} className="exp-metric-card">
                <div className="exp-metric-value">{typeof v === "number" ? (v * 100).toFixed(1) + "%" : v ?? "N/A"}</div>
                <div className="exp-metric-label">{k.replace("avg_", "").replace(/_/g, " ")}</div>
              </div>
            ))}
          </div>

          <h3>Per-Question Results</h3>
          <div className="exp-table-wrap">
            <table className="exp-table">
              <thead>
                <tr>
                  <th>ID</th><th>Category</th><th>Question</th><th>Relevance</th><th>Source</th><th>Citation</th><th>Faithful</th><th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.results.map((r) => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td><span className={`exp-badge exp-badge-${r.category}`}>{r.category}</span></td>
                    <td className="exp-q">{r.question}</td>
                    <td className={r.answer_relevance >= 0.8 ? "exp-good" : "exp-warn"}>{(r.answer_relevance * 100).toFixed(0)}%</td>
                    <td className={r.source_accuracy >= 1 ? "exp-good" : "exp-warn"}>{(r.source_accuracy * 100).toFixed(0)}%</td>
                    <td className={r.citation_present >= 1 ? "exp-good" : "exp-warn"}>{(r.citation_present * 100).toFixed(0)}%</td>
                    <td className="exp-good">{((r.faithfulness ?? r.abstention_correct) * 100).toFixed(0)}%</td>
                    <td>{r.latency?.toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Retrieval Methods ── */}
      {tab === "retrieval" && experiments?.experiment_a_retrieval_methods && (
        <div className="exp-section">
          <h2>Experiment A: Retrieval Method Comparison</h2>
          <div className="exp-compare-grid">
            {Object.entries(experiments.experiment_a_retrieval_methods).map(([method, v]) => (
              <div key={method} className="exp-compare-card">
                <h3>{method.toUpperCase()}</h3>
                <div className="exp-metric-row">
                  <span>Source Accuracy</span>
                  <span className={v.avg_source_accuracy >= 1 ? "exp-good" : "exp-warn"}>{(v.avg_source_accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="exp-metric-row">
                  <span>Avg Top Score</span>
                  <span>{v.avg_top_score?.toFixed(4)}</span>
                </div>
                <div className="exp-bar" style={{ "--pct": (v.avg_source_accuracy * 100) + "%" }} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Chunk Sizes ── */}
      {tab === "chunking" && experiments?.experiment_b_chunk_sizes && (
        <div className="exp-section">
          <h2>Experiment B: Chunk Size Comparison</h2>
          <div className="exp-compare-grid">
            {Object.entries(experiments.experiment_b_chunk_sizes).map(([size, v]) => (
              <div key={size} className="exp-compare-card">
                <h3>{size} chars</h3>
                <div className="exp-metric-row"><span>Chunks</span><span>{v.num_chunks}</span></div>
                <div className="exp-metric-row"><span>Source Accuracy</span><span>{(v.avg_source_accuracy * 100).toFixed(1)}%</span></div>
                <div className="exp-metric-row"><span>Avg Score</span><span>{v.avg_top_score?.toFixed(4)}</span></div>
                <div className="exp-bar" style={{ "--pct": (v.avg_top_score * 100) + "%" }} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Reranking ── */}
      {tab === "reranking" && experiments?.experiment_c_reranking && (
        <div className="exp-section">
          <h2>Experiment C: Reranking Impact</h2>
          <div className="exp-compare-grid">
            <div className="exp-compare-card">
              <h3>Without Reranking</h3>
              <div className="exp-metric-value-lg">{(experiments.experiment_c_reranking.without_reranking * 100).toFixed(1)}%</div>
              <div className="exp-metric-label">Source Accuracy</div>
            </div>
            <div className="exp-compare-card exp-highlight">
              <h3>With Reranking</h3>
              <div className="exp-metric-value-lg">{(experiments.experiment_c_reranking.with_reranking * 100).toFixed(1)}%</div>
              <div className="exp-metric-label">Source Accuracy</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Full RAG ── */}
      {tab === "fullrag" && experiments?.experiment_d_full_rag && (
        <div className="exp-section">
          <h2>Experiment D: Full RAG Quality</h2>
          <div className="exp-compare-grid">
            {Object.entries(experiments.experiment_d_full_rag).map(([method, v]) => (
              <div key={method} className="exp-compare-card">
                <h3>{method.toUpperCase()}</h3>
                <div className="exp-metric-row"><span>Relevance</span><span className={v.avg_relevance >= 1 ? "exp-good" : "exp-warn"}>{(v.avg_relevance * 100).toFixed(1)}%</span></div>
                <div className="exp-metric-row"><span>Avg Latency</span><span>{v.avg_latency}s</span></div>
                <div className="exp-metric-row"><span>Total Time</span><span>{v.total_latency}s</span></div>
                <div className="exp-bar" style={{ "--pct": (v.avg_relevance * 100) + "%" }} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
