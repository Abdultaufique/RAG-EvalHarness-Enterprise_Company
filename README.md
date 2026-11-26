# RAG-Based Enterprise Knowledge Management System

A production-grade **Retrieval-Augmented Generation (RAG)** system for enterprise document intelligence, built entirely on open-weight models for complete data privacy.

## 🚀 Features

- **Document Ingestion**: Upload PDF, DOCX, and TXT files
- **Hybrid Retrieval**: Dense vector search + BM25 keyword search
- **LLM Generation**: Llama 3.1 8B via Ollama (fully local, no API costs)
- **Source Citations**: Every answer includes chunk-level citations
- **Streamlit UI**: Beautiful dark-themed frontend
- **FastAPI Backend**: 9 REST endpoints with Swagger docs
- **Session Persistence**: SQLite conversation history

## 🏗️ Architecture

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.1 8B via Ollama |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector DB | Qdrant (local HNSW index) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Session DB | SQLite (aiosqlite) |
| Chunking | 500 chars, 50 overlap |

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Abdultaufique/RAG-EvalHarness-Enterprise_Company.git
cd RAG-EvalHarness-Enterprise_Company

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull the model
# Download from: https://ollama.ai
ollama pull llama3.1
```

## 🖥️ Running Locally

```bash
# Terminal 1 — Start Ollama (if not running)
ollama serve

# Terminal 2 — Start FastAPI backend
.venv\Scripts\uvicorn api:app --reload

# Terminal 3 — Start Streamlit frontend
.venv\Scripts\streamlit run streamlit_app.py
```

Then open: **http://localhost:8501**

Or use the one-click launcher: `start_streamlit.bat`

## 📖 Usage

1. Open the **Documents** tab → Upload your PDF/DOCX/TXT files
2. Click **Ingest into Vector Store** to process them
3. Switch to the **Chat** tab → Ask questions about your documents
4. View **cited sources** for every answer

## 🔍 Retrieval Modes

| Mode | Best For |
|------|----------|
| **Hybrid** (recommended) | General use — combines dense + BM25 |
| **Dense** | Conceptual / semantic questions |
| **BM25** | Exact keyword / term matching |

## 👨‍💻 Project Details

- **Student**: Abdul Taufique (Roll No: 61)
- **Year**: Fourth Year, B.E. 2026-27
- **Department**: Artificial Intelligence & Data Science
- **Institution**: DR. D. Y. PATIL SCHOOL OF SCIENCE & TECHNOLOGY, TATHAWADE, PUNE
- **Guide**: Mrs. Anagha Kulkarni
- **Course Code**: PEC-AI-701
