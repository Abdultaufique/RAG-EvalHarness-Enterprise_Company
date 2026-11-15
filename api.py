"""
API — FastAPI application for the RAG Enterprise system.

WHY FASTAPI:
- Async-native: handles concurrent requests without blocking during LLM inference.
- Automatic OpenAPI/Swagger docs at /docs (great for testing without a frontend).
- Pydantic integration for request/response validation.
- Easy to add middleware (CORS, auth) for production use.

ARCHITECTURE:
- The RAG pipeline is initialized ONCE at startup as a global singleton.
- Each API request uses the same pipeline instance (thread-safe for reads).
- Database connections are opened per-request and closed automatically.
- Uploaded files are saved to data/uploads/ for re-ingestion.

ENDPOINTS:
  POST   /api/documents/upload     Upload a document (PDF, DOCX, TXT)
  GET    /api/documents             List all documents
  DELETE /api/documents/{id}        Delete a document
  POST   /api/documents/ingest     Re-ingest all uploaded documents into vector store

  POST   /api/chat                  Ask a question (RAG pipeline)
  GET    /api/conversations         List all conversations
  GET    /api/conversations/{id}    Get full conversation with messages

  GET    /api/stats                 System statistics
  GET    /api/health                Health check
"""

import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from config import PipelineConfig, PROJECT_ROOT
from rag_pipeline import RAGPipeline
from database import (
    get_db, init_db,
    insert_document, update_document_status,
    get_all_documents, get_document_by_id, delete_document,
    create_conversation, get_all_conversations, get_conversation_by_id,
    insert_message, get_conversation_messages,
)
from schemas import (
    ChatRequest, ChatResponse, Citation,
    DocumentResponse, DocumentListResponse,
    UploadResponse, IngestRequest, IngestResponse,
    ConversationSummary, ConversationDetail, MessageResponse,
    StatsResponse, HealthResponse,
)

# ── Constants ──
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ── Global pipeline instance (initialized at startup) ──
pipeline: RAGPipeline | None = None


# ══════════════════════════════════════════════════════════════
# Application Lifecycle
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI app.

    WHY LIFESPAN:
    - The RAG pipeline (embedding model, Qdrant client) is expensive to
      initialize (~3-5 seconds). We do it ONCE at startup, not per request.
    - On shutdown, we cleanly close the Qdrant client to release file locks.
    """
    global pipeline

    print("\n=== Starting RAG Enterprise API ===")

    # Create directories
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()

    # Initialize RAG pipeline
    config = PipelineConfig()
    pipeline = RAGPipeline(config)

    # Auto-ingest any previously uploaded documents
    uploaded_files = list(UPLOAD_DIR.iterdir())
    if uploaded_files:
        valid_files = [
            f for f in uploaded_files
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if valid_files:
            print(f"\n  Auto-ingesting {len(valid_files)} previously uploaded documents...")
            try:
                pipeline.ingest(valid_files, recreate_collection=True)
            except Exception as e:
                print(f"  Warning: Auto-ingest failed: {e}")

    print("\n=== API Ready ===")
    print("  Swagger docs: http://localhost:8000/docs")
    print("  Upload dir:   ", UPLOAD_DIR)

    yield  # App is running

    # Shutdown
    print("\n=== Shutting down ===")
    if pipeline:
        pipeline.close()
    print("  Pipeline closed.")


# ══════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="RAG Enterprise API",
    description="Enterprise Document Intelligence System — "
                "upload documents, ask questions, get evidence-grounded answers.",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS middleware — allows the Next.js frontend (port 3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Logging Middleware ──
import os
import time as _time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("rag_api")

API_KEY = os.environ.get("RAG_API_KEY", "")  # Set to enable auth


@app.middleware("http")
async def log_and_auth_middleware(request, call_next):
    """Log every request and optionally enforce API key auth."""
    start = _time.time()

    # API key check (skip if not configured — dev mode)
    if API_KEY:
        # Skip auth for health check and CORS preflight
        if request.url.path != "/api/health" and request.method != "OPTIONS":
            key = request.headers.get("x-api-key", "")
            if key != API_KEY:
                from starlette.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key. Set x-api-key header."},
                )

    response = await call_next(request)
    elapsed = _time.time() - start

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.2f}s)"
    )
    return response


# ══════════════════════════════════════════════════════════════
# Document Endpoints
# ══════════════════════════════════════════════════════════════

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file (PDF, DOCX, or TXT).

    The file is saved to disk and a database record is created.
    It is NOT automatically ingested — call /api/documents/ingest
    after uploading to process all documents into the vector store.

    WHY SEPARATE UPLOAD AND INGEST:
    - Allows batch uploading multiple files before processing.
    - Ingestion rebuilds the entire vector store, which is expensive.
    - This matches the research workflow: upload a dataset, then ingest once.
    """
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Sanitize filename — prevent path traversal attacks
    safe_name = Path(file.filename).name  # Strips any directory components
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Read and validate file size (50 MB limit)
    content = await file.read()
    max_size = 50 * 1024 * 1024  # 50 MB
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max: {max_size} bytes.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Save file to uploads directory
    save_path = UPLOAD_DIR / safe_name
    save_path.write_bytes(content)

    # Create database record
    db = await get_db()
    try:
        doc_id = await insert_document(
            db,
            filename=file.filename,
            file_type=suffix.lstrip("."),
            file_size_bytes=len(content),
        )
    finally:
        await db.close()

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status="uploaded",
        message=f"File '{file.filename}' uploaded successfully. "
                f"Call POST /api/documents/ingest to process it.",
    )


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents with their processing status."""
    db = await get_db()
    try:
        docs = await get_all_documents(db)
    finally:
        await db.close()

    return DocumentListResponse(
        documents=[DocumentResponse(**doc) for doc in docs],
        total=len(docs),
    )


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int):
    """Delete a document record and its uploaded file.

    NOTE: This does NOT remove the document's chunks from the vector store.
    Call /api/documents/ingest after deleting to rebuild the vector store
    without the deleted document.
    """
    db = await get_db()
    try:
        doc = await get_document_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        # Delete file from disk
        file_path = UPLOAD_DIR / doc["filename"]
        if file_path.exists():
            file_path.unlink()

        # Delete database record
        await delete_document(db, doc_id)
    finally:
        await db.close()

    return {"message": f"Document '{doc['filename']}' deleted.", "id": doc_id}


@app.post("/api/documents/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest = IngestRequest()):
    """Ingest all uploaded documents into the vector store.

    This processes every file in the uploads directory:
    1. Extract text from each document.
    2. Split into chunks.
    3. Generate embeddings.
    4. Store in Qdrant.

    If recreate_collection is true (default), the vector store is wiped
    first — this ensures a clean state for experiments.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    # Find all valid files in uploads
    uploaded_files = [
        f for f in UPLOAD_DIR.iterdir()
        if f.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not uploaded_files:
        raise HTTPException(
            status_code=400,
            detail="No documents to ingest. Upload files first.",
        )

    # Run ingestion
    try:
        stats = pipeline.ingest(
            uploaded_files,
            recreate_collection=request.recreate_collection,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )

    # Update database records for all successfully processed documents
    db = await get_db()
    try:
        all_docs = await get_all_documents(db)
        for doc in all_docs:
            if doc["status"] == "pending":
                await update_document_status(
                    db, doc["id"],
                    status="processed",
                    num_sections=stats.get("sections_extracted", 0),
                    num_chunks=stats.get("chunks_created", 0),
                )
    finally:
        await db.close()

    return IngestResponse(
        documents_loaded=stats["documents_loaded"],
        sections_extracted=stats["sections_extracted"],
        chunks_created=stats["chunks_created"],
        embeddings_generated=stats["embeddings_generated"],
        embedding_dimension=stats["embedding_dimension"],
        message=f"Successfully ingested {stats['documents_loaded']} documents "
                f"into {stats['chunks_created']} chunks.",
    )


# ══════════════════════════════════════════════════════════════
# Chat Endpoints
# ══════════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question and get an evidence-grounded answer.

    HOW IT WORKS:
    1. Retrieves the most relevant document chunks using the specified method.
    2. Sends the question + chunks to the LLM for answer generation.
    3. Returns the answer with source citations.
    4. Saves both the question and answer to the conversation history.

    If no conversation_id is provided, a new conversation is created
    using the first few words of the question as the title.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    # Override retrieval settings for this query
    pipeline.config.retriever.method = request.retrieval_method
    pipeline.config.retriever.top_k = request.top_k
    pipeline.retriever.config.method = request.retrieval_method
    pipeline.retriever.config.top_k = request.top_k
    # Reset BM25 cache when switching methods
    pipeline.retriever._bm25_index = None
    pipeline.retriever._bm25_chunks = None

    # Run the RAG pipeline
    try:
        result = pipeline.query(request.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}\n{traceback.format_exc()}",
        )

    # ── Persist to database ──
    db = await get_db()
    try:
        # Create or reuse conversation
        if request.conversation_id:
            conv = await get_conversation_by_id(db, request.conversation_id)
            if not conv:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation {request.conversation_id} not found.",
                )
            conv_id = request.conversation_id
        else:
            # Auto-generate title from question (first 50 chars)
            title = request.question[:50] + ("..." if len(request.question) > 50 else "")
            conv_id = await create_conversation(db, title=title)

        # Save the user's question
        await insert_message(
            db,
            conversation_id=conv_id,
            role="user",
            content=request.question,
        )

        # Prepare citations for storage (strip the full text to save space)
        citation_data = [
            {
                "source_file": c["source_file"],
                "page_number": c.get("page_number", -1),
                "chunk_index": c.get("chunk_index", -1),
                "score": round(c.get("score", 0), 4),
                "text": c.get("text", "")[:200],  # Truncate for DB storage
            }
            for c in result.get("citations", [])
        ]

        # Save the assistant's answer
        msg_id = await insert_message(
            db,
            conversation_id=conv_id,
            role="assistant",
            content=result["answer"],
            model=result.get("model"),
            retrieval_method=request.retrieval_method,
            latency_seconds=result.get("latency_seconds"),
            citations=citation_data,
        )
    finally:
        await db.close()

    # Build response
    citations = [
        Citation(
            source_file=c["source_file"],
            page_number=c.get("page_number", -1),
            chunk_index=c.get("chunk_index", -1),
            score=round(c.get("score", 0), 4),
            text=c.get("text", ""),
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        conversation_id=conv_id,
        message_id=msg_id,
        answer=result["answer"],
        model=result.get("model", "unknown"),
        retrieval_method=request.retrieval_method,
        latency_seconds=result.get("latency_seconds", 0),
        citations=citations,
    )


# ══════════════════════════════════════════════════════════════
# Conversation Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/conversations", response_model=list[ConversationSummary])
async def list_conversations():
    """List all conversations (most recent first)."""
    db = await get_db()
    try:
        conversations = await get_all_conversations(db)
    finally:
        await db.close()

    return [ConversationSummary(**conv) for conv in conversations]


@app.get("/api/conversations/{conv_id}", response_model=ConversationDetail)
async def get_conversation(conv_id: int):
    """Get a conversation with all its messages."""
    db = await get_db()
    try:
        conv = await get_conversation_by_id(db, conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        messages = await get_conversation_messages(db, conv_id)
    finally:
        await db.close()

    return ConversationDetail(
        id=conv["id"],
        title=conv.get("title"),
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=[
            MessageResponse(
                id=msg["id"],
                role=msg["role"],
                content=msg["content"],
                model=msg.get("model"),
                retrieval_method=msg.get("retrieval_method"),
                latency_seconds=msg.get("latency_seconds"),
                citations=[
                    Citation(**c) for c in msg["citations"]
                ] if msg.get("citations") else None,
                created_at=msg["created_at"],
            )
            for msg in messages
        ],
    )


# ══════════════════════════════════════════════════════════════
# System Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system-wide statistics."""
    db = await get_db()
    try:
        docs = await get_all_documents(db)
        conversations = await get_all_conversations(db)

        # Count messages across all conversations
        total_messages = 0
        for conv in conversations:
            msgs = await get_conversation_messages(db, conv["id"])
            total_messages += len(msgs)
    finally:
        await db.close()

    # Vector store chunk count
    vector_chunks = None
    if pipeline:
        try:
            info = pipeline.vector_store.client.get_collection(
                pipeline.config.qdrant.collection_name
            )
            vector_chunks = info.points_count
        except Exception:
            pass

    return StatsResponse(
        total_documents=len(docs),
        processed_documents=sum(1 for d in docs if d["status"] == "processed"),
        failed_documents=sum(1 for d in docs if d["status"] == "failed"),
        total_conversations=len(conversations),
        total_messages=total_messages,
        vector_store_chunks=vector_chunks,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if all system components are operational."""
    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Check vector store
    vector_ok = False
    if pipeline:
        try:
            pipeline.vector_store.client.get_collections()
            vector_ok = True
        except Exception:
            pass

    # Check database
    db_ok = False
    try:
        db = await get_db()
        await db.execute("SELECT 1")
        await db.close()
        db_ok = True
    except Exception:
        pass

    overall = "healthy" if (ollama_ok and vector_ok and db_ok) else "degraded"

    return HealthResponse(
        status=overall,
        ollama_connected=ollama_ok,
        vector_store_ready=vector_ok,
        database_ready=db_ok,
    )


@app.get("/api/experiments")
async def get_experiments():
    """Return saved evaluation and experiment results for the frontend."""
    import json

    data_dir = PROJECT_ROOT / "data"
    results = {}

    eval_path = data_dir / "eval_results.json"
    if eval_path.exists():
        results["evaluation"] = json.loads(eval_path.read_text(encoding="utf-8"))

    exp_path = data_dir / "experiment_results.json"
    if exp_path.exists():
        results["experiments"] = json.loads(exp_path.read_text(encoding="utf-8"))

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No experiment results found. Run test_milestone6.py and test_milestone7.py first.",
        )

    return results
