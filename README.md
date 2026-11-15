# RAG Enterprise — Document Intelligence System

A complete, research-oriented Retrieval-Augmented Generation (RAG) system that enables users to upload enterprise documents, ask natural language questions, and receive evidence-grounded answers with source citations.

## Features

- **Multi-format document ingestion** — PDF, DOCX, TXT with metadata preservation
- **Configurable chunking** — Fixed-size with overlap (250/500/1000 chars)
- **Dense, BM25, and Hybrid retrieval** — Semantic + keyword search with weighted fusion
- **Cross-encoder reranking** — Optional re-scoring with `ms-marco-MiniLM-L-6-v2`
- **LLM-powered answers** — Via Ollama (llama3.1 8B), with source citations and abstention
- **Conversation history** — SQLite-backed chat persistence
- **Modern web UI** — Next.js dark theme with glassmorphism, citation viewer
- **Evaluation framework** — 13-question benchmark with deterministic metrics
- **Experiment runner** — Systematic comparison of retrieval methods, chunk sizes, reranking

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Store | Qdrant (local file-based) |
| BM25 | rank-bm25 |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Ollama → llama3.1:latest (8B) |
| API | FastAPI + Uvicorn |
| Database | SQLite (via aiosqlite) |
| Frontend | Next.js 16 + React |
| Document Parsing | PyMuPDF, python-docx |

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai/) with `llama3.1` pulled

### Setup

```bash
# Clone and enter project
cd rag-enterprise

# Create virtual environment
python -m venv .venv

# Install Python dependencies
.venv\Scripts\pip.exe install -r requirements.txt

# Pull the LLM model
ollama pull llama3.1

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Run

```bash
# Terminal 1 — Ollama (if not already running)
ollama serve

# Terminal 2 — Backend API
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

# Terminal 3 — Frontend
cd frontend && npm run dev
```

- **Frontend**: http://localhost:3000
- **Swagger API docs**: http://localhost:8000/docs

## Project Structure

```
rag-enterprise/
├── api.py                 # FastAPI application (9 endpoints)
├── config.py              # All pipeline configuration
├── rag_pipeline.py        # Orchestrator (ingest + query)
├── document_loader.py     # PDF/DOCX/TXT text extraction
├── chunker.py             # Fixed-size chunking with overlap
├── embedder.py            # Sentence Transformers wrapper
├── vector_store.py        # Qdrant vector storage + search
├── retriever.py           # Dense, BM25, Hybrid retrieval
├── generator.py           # LLM answer generation via Ollama
├── database.py            # SQLite schema + CRUD
├── schemas.py             # Pydantic request/response models
├── benchmark.py           # 13-question ground truth dataset
├── evaluator.py           # Deterministic evaluation metrics
├── requirements.txt       # Python dependencies
├── main.py                # CLI test entry point
├── test_milestone3.py     # Multi-format + retrieval tests
├── test_milestone4.py     # API endpoint tests
├── test_milestone6.py     # Benchmark evaluation runner
├── test_milestone7.py     # Controlled experiment runner
├── sample_docs/           # Test documents (AI, climate, space, quantum, blockchain)
├── data/                  # Qdrant storage, SQLite DB, eval results
└── frontend/              # Next.js web application
    ├── app/
    │   ├── layout.js
    │   ├── page.js        # Chat UI with sidebar, citations, controls
    │   └── globals.css    # Dark theme design system
    └── lib/
        └── api.js         # Backend API client
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload PDF, DOCX, or TXT |
| GET | `/api/documents` | List all documents |
| DELETE | `/api/documents/{id}` | Delete a document |
| POST | `/api/documents/ingest` | Process uploads into vector store |
| POST | `/api/chat` | Ask a question (RAG) |
| GET | `/api/conversations` | List conversations |
| GET | `/api/conversations/{id}` | Get conversation with messages |
| GET | `/api/stats` | System statistics |
| GET | `/api/health` | Health check |

## Research Results

### Retrieval Method Comparison
| Method | Source Accuracy | Answer Relevance |
|---|---|---|
| **Dense** | **1.000** | **1.000** |
| Hybrid | 1.000 | 0.917 |
| BM25 | 0.900 | 0.833 |

### Chunk Size Impact
| Size | Chunks | Retrieval Score |
|---|---|---|
| 250 chars | 133 | **0.696** |
| 500 chars | 46 | 0.627 |
| 1000 chars | 157 | 0.696 |

### Key Findings
1. Dense retrieval is the best default for this corpus
2. Hybrid adds confidence on harder queries
3. Smaller chunks improve retrieval precision
4. Cross-encoder reranking is valuable at scale
5. llama3.1 8B achieves 100% accuracy on factual questions with proper RAG

## Security

- File type validation (PDF, DOCX, TXT only)
- File size limit (50 MB)
- Filename sanitization (path traversal prevention)
- CORS restricted to localhost origins
- Input length validation on chat queries
- SQL injection prevention via parameterized queries

## License

MIT
