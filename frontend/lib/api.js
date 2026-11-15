const API_BASE = "http://127.0.0.1:8000";

export async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API error");
  }
  return res.json();
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export const api = {
  health: () => apiFetch("/api/health"),
  getDocuments: () => apiFetch("/api/documents"),
  deleteDocument: (id) => apiFetch(`/api/documents/${id}`, { method: "DELETE" }),
  ingest: (recreate = true) =>
    apiFetch("/api/documents/ingest", {
      method: "POST",
      body: JSON.stringify({ recreate_collection: recreate }),
    }),
  chat: (question, opts = {}) =>
    apiFetch("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        conversation_id: opts.conversationId || null,
        retrieval_method: opts.retrievalMethod || "dense",
        top_k: opts.topK || 5,
      }),
    }),
  getConversations: () => apiFetch("/api/conversations"),
  getConversation: (id) => apiFetch(`/api/conversations/${id}`),
  getStats: () => apiFetch("/api/stats"),
};
