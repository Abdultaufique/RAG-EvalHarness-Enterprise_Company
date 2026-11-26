"""
Streamlit Frontend — RAG-Based Enterprise Knowledge Management System

This is a full-featured Streamlit UI that connects directly to the FastAPI
backend running at localhost:8000. It provides:
  - Document upload and ingestion
  - Multi-turn chat with citation display
  - Retrieval mode toggle (Dense / BM25 / Hybrid)
  - Conversation history sidebar
  - System health & stats dashboard

HOW TO RUN:
    cd c:\Users\abdul\OneDrive\Documents\Desktop\rag-enterprise
    .venv\Scripts\streamlit run streamlit_app.py

The FastAPI backend must be running first:
    .venv\Scripts\uvicorn api:app --reload
"""

import requests
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Enterprise Knowledge System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark professional theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3148;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 60px;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
        font-size: 14px;
        line-height: 1.6;
    }
    .assistant-bubble {
        background: #1e2130;
        border: 1px solid #2d3148;
        color: #e2e8f0;
        padding: 14px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 60px 8px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Citation pills */
    .citation-pill {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #2563eb;
        color: #93c5fd;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        margin: 3px 2px;
        cursor: default;
    }

    /* Stats cards */
    .stat-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .stat-num {
        font-size: 32px;
        font-weight: 700;
        color: #60a5fa;
    }
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Health badge */
    .badge-healthy { color: #22c55e; font-weight: bold; }
    .badge-degraded { color: #f59e0b; font-weight: bold; }
    .badge-offline  { color: #ef4444; font-weight: bold; }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2d3148;
        border-radius: 10px;
        padding: 10px;
    }

    /* Button override */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    }

    /* Thinking spinner */
    .thinking {
        color: #64748b;
        font-style: italic;
        font-size: 13px;
    }

    /* Input */
    .stTextInput input, .stTextArea textarea {
        background-color: #1e2130 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3148 !important;
    }

    /* Remove default padding */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10):
    """Make a GET request to the API."""
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(path: str, json_data: dict = None, timeout: int = 300):
    """Make a POST request to the API."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"error": "Cannot connect to FastAPI backend. Please start it first."}
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": detail}
    except Exception as e:
        return {"error": str(e)}


def upload_document(file_bytes: bytes, filename: str):
    """Upload a file to the API."""
    try:
        r = requests.post(
            f"{API_BASE}/api/documents/upload",
            files={"file": (filename, file_bytes)},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"error": "Cannot connect to FastAPI backend."}
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": detail}
    except Exception as e:
        return {"error": str(e)}


def check_api_health():
    """Check if the backend API is running."""
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, citations, latency}]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "hybrid"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "api_ok" not in st.session_state:
    st.session_state.api_ok = False
if "health_data" not in st.session_state:
    st.session_state.health_data = {}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:40px;">🧠</div>
        <div style="font-size:17px; font-weight:700; color:#e2e8f0;">RAG Enterprise</div>
        <div style="font-size:11px; color:#64748b;">Knowledge Management System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Status ──
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    api_ok, health_data = check_api_health()
    st.session_state.api_ok = api_ok
    st.session_state.health_data = health_data

    if api_ok:
        ollama = health_data.get("ollama_connected", False)
        vector = health_data.get("vector_store_ready", False)
        db_ok  = health_data.get("database_ready", False)
        overall = health_data.get("status", "unknown")

        badge_cls = "badge-healthy" if overall == "healthy" else "badge-degraded"
        st.markdown(f"""
        <div style="background:#1e2130; border:1px solid #2d3148; border-radius:8px; padding:10px; font-size:13px;">
            <span class="{badge_cls}">● {overall.upper()}</span><br>
            <span style="color:#94a3b8;">Ollama LLM:</span> {'<span style="color:#22c55e;">✓ Connected</span>' if ollama else '<span style="color:#ef4444;">✗ Offline</span>'}<br>
            <span style="color:#94a3b8;">Vector Store:</span> {'<span style="color:#22c55e;">✓ Ready</span>' if vector else '<span style="color:#f59e0b;">○ Empty</span>'}<br>
            <span style="color:#94a3b8;">Database:</span> {'<span style="color:#22c55e;">✓ OK</span>' if db_ok else '<span style="color:#ef4444;">✗ Error</span>'}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#2d1515; border:1px solid #ef4444; border-radius:8px; padding:10px; font-size:13px; color:#fca5a5;">
            ✗ <strong>API Offline</strong><br>
            Start backend first:<br>
            <code style="font-size:11px;">.venv\\Scripts\\uvicorn api:app --reload</code>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Retrieval Settings ──
    st.markdown('<div class="section-header">Retrieval Settings</div>', unsafe_allow_html=True)
    method = st.radio(
        "Method",
        ["hybrid", "dense", "bm25"],
        index=0,
        horizontal=False,
        key="retrieval_radio",
        help="Hybrid combines dense vectors + BM25 keyword search for best results."
    )
    st.session_state.retrieval_method = method

    top_k = st.slider(
        "Top-K chunks",
        min_value=1, max_value=10, value=5,
        help="Number of document chunks retrieved per query."
    )
    st.session_state.top_k = top_k

    st.divider()

    # ── Conversation Controls ──
    st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    # Load past conversations
    if api_ok:
        convs = api_get("/api/conversations")
        if convs and len(convs) > 0:
            conv_titles = {c["id"]: c.get("title", f"Conversation #{c['id']}") for c in convs}
            selected = st.selectbox(
                "Load conversation",
                options=["— Select —"] + list(conv_titles.keys()),
                format_func=lambda x: "— Select —" if x == "— Select —" else conv_titles.get(x, str(x)),
            )
            if selected != "— Select —":
                if st.button("📂 Load", use_container_width=True):
                    conv_detail = api_get(f"/api/conversations/{selected}")
                    if conv_detail:
                        st.session_state.conversation_id = selected
                        st.session_state.messages = []
                        for msg in conv_detail.get("messages", []):
                            st.session_state.messages.append({
                                "role": msg["role"],
                                "content": msg["content"],
                                "citations": msg.get("citations", []),
                                "latency": msg.get("latency_seconds"),
                                "model": msg.get("model"),
                            })
                        st.rerun()

    st.divider()

    # ── Stats ──
    if api_ok:
        st.markdown('<div class="section-header">Statistics</div>', unsafe_allow_html=True)
        stats = api_get("/api/stats")
        if stats:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-num">{stats.get("total_documents", 0)}</div>
                    <div class="stat-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-num">{stats.get("vector_store_chunks") or 0}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-num">{stats.get("total_conversations", 0)}</div>
                    <div class="stat-label">Chats</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-num">{stats.get("total_messages", 0)}</div>
                    <div class="stat-label">Messages</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Content Area — Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_about = st.tabs(["💬 Chat", "📄 Documents", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════
# TAB 1: Chat
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        "<h2 style='color:#e2e8f0; margin-bottom:4px;'>Ask Your Documents</h2>"
        "<p style='color:#64748b; font-size:13px; margin-top:0;'>Upload documents first, then ask questions about them.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.api_ok:
        st.warning(
            "⚠️ **FastAPI backend is not running.**\n\n"
            "Start it with:\n"
            "```\n.venv\\Scripts\\uvicorn api:app --reload\n```"
        )

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding: 60px 0; color:#475569;">
                <div style="font-size:48px;">💬</div>
                <div style="font-size:16px; margin-top:12px;">No messages yet</div>
                <div style="font-size:13px; margin-top:6px;">Upload documents and start asking questions</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    answer_html = msg["content"].replace("\n", "<br>")
                    st.markdown(
                        f'<div class="assistant-bubble">🤖 {answer_html}</div>',
                        unsafe_allow_html=True
                    )
                    # Citations
                    citations = msg.get("citations", [])
                    if citations:
                        with st.expander(f"📚 {len(citations)} source(s) cited", expanded=False):
                            for i, cite in enumerate(citations, 1):
                                score = cite.get("score", 0)
                                src = cite.get("source_file", "unknown")
                                chunk_idx = cite.get("chunk_index", -1)
                                text = cite.get("text", "")
                                st.markdown(f"""
                                <div style="border-left:3px solid #2563eb; padding:8px 12px; margin:8px 0; background:#1a1d27; border-radius:0 6px 6px 0;">
                                    <div style="font-size:12px; color:#60a5fa; font-weight:600;">
                                        [{i}] {src} — Chunk #{chunk_idx}
                                        <span style="float:right; color:#94a3b8;">Score: {score:.4f}</span>
                                    </div>
                                    <div style="font-size:12px; color:#94a3b8; margin-top:6px; line-height:1.5;">
                                        {text[:300]}{'...' if len(text) > 300 else ''}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Metadata row
                    meta_parts = []
                    if msg.get("latency"):
                        meta_parts.append(f"⚡ {msg['latency']}s")
                    if msg.get("model"):
                        meta_parts.append(f"🤖 {msg['model']}")
                    if meta_parts:
                        st.markdown(
                            f"<div style='font-size:11px; color:#475569; margin: -4px 0 12px 4px;'>"
                            f"{' · '.join(meta_parts)}</div>",
                            unsafe_allow_html=True,
                        )

    st.divider()

    # Input row
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Your question",
            placeholder="e.g. What are the main findings from the quarterly report?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        send = st.button("Send →", use_container_width=True, disabled=not st.session_state.api_ok)

    if send and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip(),
            "citations": [],
        })

        # Call API
        with st.spinner("🔍 Searching documents and generating answer..."):
            payload = {
                "question": user_input.strip(),
                "retrieval_method": st.session_state.retrieval_method,
                "top_k": st.session_state.top_k,
            }
            if st.session_state.conversation_id:
                payload["conversation_id"] = st.session_state.conversation_id

            result = api_post("/api/chat", payload)

        if "error" in result:
            st.error(f"❌ {result['error']}")
            st.session_state.messages.pop()  # Remove the user message on error
        else:
            st.session_state.conversation_id = result.get("conversation_id")
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", "No answer returned."),
                "citations": result.get("citations", []),
                "latency": result.get("latency_seconds"),
                "model": result.get("model"),
            })
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2: Documents
# ══════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown(
        "<h2 style='color:#e2e8f0; margin-bottom:4px;'>Document Management</h2>"
        "<p style='color:#64748b; font-size:13px;'>Upload PDF, DOCX, or TXT files and ingest them into the knowledge base.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.api_ok:
        st.warning("⚠️ Backend offline — start FastAPI first.")
    else:
        col_upload, col_list = st.columns([1, 1])

        with col_upload:
            st.markdown("### 📤 Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )

            if uploaded_files:
                if st.button("⬆️ Upload All Files", use_container_width=True):
                    success_count = 0
                    for uf in uploaded_files:
                        with st.spinner(f"Uploading {uf.name}..."):
                            result = upload_document(uf.read(), uf.name)
                        if "error" in result:
                            st.error(f"✗ {uf.name}: {result['error']}")
                        else:
                            st.success(f"✓ {uf.name} uploaded (ID: {result.get('document_id')})")
                            success_count += 1

                    if success_count > 0:
                        st.info(f"Now click **Ingest into Vector Store** to process {success_count} file(s).")

            st.divider()
            st.markdown("### ⚙️ Ingest Documents")
            st.markdown(
                "<p style='color:#94a3b8; font-size:13px;'>Ingest rebuilds the vector store from all uploaded documents. "
                "This embeds all chunks and stores them for retrieval.</p>",
                unsafe_allow_html=True
            )
            recreate = st.checkbox("Recreate collection (clean slate)", value=True)
            if st.button("🚀 Ingest into Vector Store", use_container_width=True):
                with st.spinner("Ingesting documents... this may take 30-60 seconds for large files."):
                    result = api_post("/api/documents/ingest", {"recreate_collection": recreate})
                if "error" in result:
                    st.error(f"❌ Ingestion failed: {result['error']}")
                else:
                    st.success(
                        f"✅ Ingested {result.get('documents_loaded', '?')} documents → "
                        f"{result.get('chunks_created', '?')} chunks → "
                        f"{result.get('embeddings_generated', '?')} embeddings"
                    )
                    st.balloons()

        with col_list:
            st.markdown("### 📋 Uploaded Documents")
            if st.button("🔄 Refresh List", use_container_width=True):
                st.rerun()

            docs_data = api_get("/api/documents")
            docs = docs_data.get("documents", []) if docs_data else []

            if not docs:
                st.markdown("""
                <div style="text-align:center; padding:40px 0; color:#475569;">
                    <div style="font-size:36px;">📂</div>
                    <div style="font-size:14px; margin-top:8px;">No documents uploaded yet</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for doc in docs:
                    status = doc.get("status", "unknown")
                    status_color = {
                        "processed": "#22c55e",
                        "pending": "#f59e0b",
                        "failed": "#ef4444",
                    }.get(status, "#94a3b8")
                    size_kb = doc.get("file_size_bytes", 0) / 1024
                    chunks = doc.get("num_chunks", 0) or 0

                    with st.expander(f"📄 {doc.get('filename', 'unknown')}", expanded=False):
                        st.markdown(f"""
                        <div style="font-size:13px; color:#94a3b8;">
                            <b>ID:</b> {doc.get('id')} &nbsp;|&nbsp;
                            <b>Type:</b> {doc.get('file_type', '?').upper()} &nbsp;|&nbsp;
                            <b>Size:</b> {size_kb:.1f} KB<br>
                            <b>Status:</b> <span style="color:{status_color};">● {status}</span> &nbsp;|&nbsp;
                            <b>Chunks:</b> {chunks}<br>
                            <b>Uploaded:</b> {doc.get('created_at', 'N/A')[:19] if doc.get('created_at') else 'N/A'}
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"🗑️ Delete", key=f"del_{doc.get('id')}"):
                            try:
                                r = requests.delete(f"{API_BASE}/api/documents/{doc.get('id')}", timeout=10)
                                if r.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.error(f"Delete failed: {r.text}")
                            except Exception as e:
                                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 3: About
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <h2 style='color:#e2e8f0;'>About This System</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏗️ Architecture
        **RAG-Based Enterprise Knowledge Management System**
        — a production-grade Retrieval-Augmented Generation (RAG) system
        built on top of open-weight models for complete data privacy.

        | Component | Technology |
        |-----------|-----------|
        | **LLM** | Llama 3.1 8B via Ollama |
        | **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
        | **Vector DB** | Qdrant (local HNSW) |
        | **Backend** | FastAPI (REST) |
        | **Frontend** | Streamlit |
        | **Session DB** | SQLite (aiosqlite) |
        | **Chunking** | 500 chars, 50 overlap |
        """)

    with col2:
        st.markdown("""
        ### 🔍 Retrieval Modes

        | Mode | Description |
        |------|------------|
        | **Dense** | Semantic vector search using sentence embeddings. Best for conceptual questions. |
        | **BM25** | Traditional keyword TF-IDF search. Best for exact term matching. |
        | **Hybrid** | Weighted combination (70% dense + 30% BM25). Best overall performance. |

        ### 🚀 How to Use
        1. **Upload** your documents (PDF, DOCX, TXT) in the Documents tab
        2. **Ingest** them to build the vector store
        3. **Ask** questions in the Chat tab
        4. View **citations** to see where answers came from
        """)

    st.divider()

    st.markdown("""
    ### 👨‍💻 Project Details
    - **Student**: Abdul Taufique (Roll No: 61)
    - **Year**: Fourth Year, B.E. 2026-27
    - **Department**: Artificial Intelligence & Data Science
    - **Institution**: DR. D. Y. PATIL SCHOOL OF SCIENCE & TECHNOLOGY, TATHAWADE, PUNE
    - **Guide**: Mrs. Anagha Kulkarni
    - **Course Code**: PEC-AI-701

    > **Note**: This system runs entirely locally — your documents never leave your machine.
    > Ollama serves the Llama 3.1 model locally without any cloud API calls.
    """)
