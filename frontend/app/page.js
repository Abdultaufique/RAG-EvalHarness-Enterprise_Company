"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { api, uploadDocument } from "@/lib/api";

export default function Home() {
  // ── State ──
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [retrievalMethod, setRetrievalMethod] = useState("dense");
  const [topK, setTopK] = useState(5);
  const [health, setHealth] = useState(null);
  const [toast, setToast] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState({});
  const [dragging, setDragging] = useState(false);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // ── Toast helper ──
  const showToast = useCallback((msg, type = "success") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  }, []);

  // ── Load initial data ──
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "degraded" }));
    api.getConversations().then(setConversations).catch(() => {});
    api.getDocuments().then((d) => setDocuments(d.documents || [])).catch(() => {});
  }, []);

  // ── Auto-scroll ──
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Auto-resize textarea ──
  const handleInputChange = (e) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  // ── Send message ──
  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);

    try {
      const res = await api.chat(q, {
        conversationId,
        retrievalMethod,
        topK,
      });

      setConversationId(res.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          model: res.model,
          latency: res.latency_seconds,
          method: res.retrieval_method,
          citations: res.citations,
        },
      ]);

      // Refresh conversations list
      api.getConversations().then(setConversations).catch(() => {});
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message}`, error: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── New chat ──
  const startNewChat = () => {
    setConversationId(null);
    setMessages([]);
  };

  // ── Load conversation ──
  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setConversationId(id);
      setMessages(
        conv.messages.map((m) => ({
          role: m.role,
          content: m.content,
          model: m.model,
          latency: m.latency_seconds,
          method: m.retrieval_method,
          citations: m.citations,
        }))
      );
    } catch (err) {
      showToast("Failed to load conversation", "error");
    }
  };

  // ── File upload ──
  const handleFileUpload = async (files) => {
    for (const file of files) {
      try {
        await uploadDocument(file);
        showToast(`Uploaded ${file.name}`);
      } catch (err) {
        showToast(`Failed: ${err.message}`, "error");
      }
    }
    api.getDocuments().then((d) => setDocuments(d.documents || [])).catch(() => {});
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFileUpload(Array.from(e.dataTransfer.files));
  };

  // ── Ingest ──
  const handleIngest = async () => {
    setIngesting(true);
    try {
      const res = await api.ingest();
      showToast(`Ingested ${res.chunks_created} chunks from ${res.documents_loaded} docs`);
      api.getDocuments().then((d) => setDocuments(d.documents || [])).catch(() => {});
    } catch (err) {
      showToast(`Ingest failed: ${err.message}`, "error");
    } finally {
      setIngesting(false);
    }
  };

  // ── Delete document ──
  const handleDeleteDoc = async (id, name) => {
    try {
      await api.deleteDocument(id);
      showToast(`Deleted ${name}`);
      api.getDocuments().then((d) => setDocuments(d.documents || [])).catch(() => {});
    } catch (err) {
      showToast(`Failed: ${err.message}`, "error");
    }
  };

  // ── Toggle citations ──
  const toggleCitations = (idx) => {
    setExpandedCitations((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  // ── File type icon ──
  const fileIcon = (name) => {
    if (name?.endsWith(".pdf")) return "📄";
    if (name?.endsWith(".docx")) return "📝";
    return "📃";
  };

  return (
    <div className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="icon">⚡</div>
            RAG Enterprise
          </div>
        </div>

        <div className="sidebar-section">
          <button className="new-chat-btn" onClick={startNewChat}>
            + New Chat
          </button>

          <div className="sidebar-section-title">Conversations</div>
          {conversations.length === 0 && (
            <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
              No conversations yet
            </p>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${conv.id === conversationId ? "active" : ""}`}
              onClick={() => loadConversation(conv.id)}
            >
              💬 {conv.title || `Chat ${conv.id}`}
            </div>
          ))}
        </div>

        {/* ── Documents Section ── */}
        <div className="sidebar-footer">
          <div className="sidebar-section-title">Documents</div>

          {documents.map((doc) => (
            <div key={doc.id} className="doc-item">
              <span className="doc-icon">{fileIcon(doc.filename)}</span>
              <span className="doc-name">{doc.filename}</span>
              <span className={`doc-status ${doc.status}`} title={doc.status} />
              <button
                className="delete-btn"
                onClick={() => handleDeleteDoc(doc.id, doc.filename)}
                title="Delete"
              >
                ✕
              </button>
            </div>
          ))}

          <div
            className={`doc-upload-area ${dragging ? "dragging" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            📎 Drop files or click to upload
            <br />
            <span style={{ fontSize: 10 }}>PDF, DOCX, TXT</span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            hidden
            multiple
            accept=".pdf,.docx,.txt"
            onChange={(e) => handleFileUpload(Array.from(e.target.files))}
          />

          <button
            className="ingest-btn"
            onClick={handleIngest}
            disabled={ingesting || documents.length === 0}
          >
            {ingesting ? "Processing..." : "🔄 Ingest Documents"}
          </button>

          <a href="/experiments" className="nav-link">📊 View Experiments</a>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main-content">
        {/* Header */}
        <div className="main-header">
          <h2>
            {conversationId
              ? conversations.find((c) => c.id === conversationId)?.title || "Chat"
              : "New Chat"}
          </h2>
          <div className="header-controls">
            <select value={retrievalMethod} onChange={(e) => setRetrievalMethod(e.target.value)}>
              <option value="dense">Dense</option>
              <option value="bm25">BM25</option>
              <option value="hybrid">Hybrid</option>
            </select>
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              <option value={3}>Top 3</option>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
            </select>
            {health && (
              <span className={`status-badge ${health.status}`}>
                ● {health.status}
              </span>
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          {messages.length === 0 && !loading && (
            <div className="chat-empty">
              <div className="big-icon">🔍</div>
              <h3>Ask your documents anything</h3>
              <p>
                Upload documents, ingest them, then ask questions. Answers are
                grounded in your documents with source citations.
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message message-${msg.role}`}>
              <div className="message-bubble">
                {msg.content.split("\n").map((line, i) => (
                  <p key={i}>{line || "\u00A0"}</p>
                ))}
              </div>

              {msg.role === "assistant" && !msg.error && (
                <>
                  <div className="message-meta">
                    {msg.model && <span>🤖 {msg.model}</span>}
                    {msg.latency && <span>⏱ {msg.latency}s</span>}
                    {msg.method && <span>🔎 {msg.method}</span>}
                    {msg.citations?.length > 0 && (
                      <span>📎 {msg.citations.length} sources</span>
                    )}
                  </div>

                  {msg.citations?.length > 0 && (
                    <>
                      <button
                        className="citations-toggle"
                        onClick={() => toggleCitations(idx)}
                      >
                        {expandedCitations[idx] ? "▾ Hide" : "▸ Show"} Sources (
                        {msg.citations.length})
                      </button>

                      {expandedCitations[idx] && (
                        <div className="citations-list">
                          {msg.citations.map((c, ci) => (
                            <div key={ci} className="citation-card">
                              <div className="citation-header">
                                <span className="citation-source">
                                  {fileIcon(c.source_file)} {c.source_file}
                                  {c.page_number > 0 && ` · p.${c.page_number}`}
                                  {c.chunk_index >= 0 && ` · chunk #${c.chunk_index}`}
                                </span>
                                <span className="citation-score">
                                  {(c.score * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="citation-text">{c.text}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </div>
          ))}

          {loading && (
            <div className="message message-assistant">
              <div className="message-bubble">
                <div className="loading-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              rows={1}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!input.trim() || loading}
              title="Send"
            >
              ➤
            </button>
          </div>
        </div>
      </main>

      {/* Toast */}
      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  );
}
