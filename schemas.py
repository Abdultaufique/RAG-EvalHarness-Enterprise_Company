"""
Schemas — Pydantic models for API request/response validation.

WHY PYDANTIC:
FastAPI uses Pydantic models to:
1. Validate incoming request data (type checking, required fields).
2. Serialize outgoing response data (consistent JSON structure).
3. Auto-generate OpenAPI/Swagger documentation.

These models define the "contract" between the API and the frontend.
"""

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
# Request Models (what the client sends)
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint.

    question: The user's natural language question.
    conversation_id: Optional — if provided, the message is added to an
                     existing conversation. If omitted, a new conversation
                     is created automatically.
    retrieval_method: Override the default retrieval strategy for this query.
    top_k: Override the number of chunks to retrieve.
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about the uploaded documents.",
    )
    conversation_id: int | None = Field(
        default=None,
        description="Existing conversation ID to append to. "
                    "If null, a new conversation is created.",
    )
    retrieval_method: str = Field(
        default="dense",
        description="Retrieval strategy: 'dense', 'bm25', or 'hybrid'.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (1-20).",
    )


class IngestRequest(BaseModel):
    """Request body for triggering re-ingestion of all documents."""
    recreate_collection: bool = Field(
        default=True,
        description="If true, wipe existing vector data before re-ingesting.",
    )


# ══════════════════════════════════════════════════════════════
# Response Models (what the API returns)
# ══════════════════════════════════════════════════════════════

class Citation(BaseModel):
    """A single source citation from the retriever."""
    source_file: str
    page_number: int = -1
    chunk_index: int = -1
    score: float
    text: str


class ChatResponse(BaseModel):
    """Response body from the /api/chat endpoint."""
    conversation_id: int
    message_id: int
    answer: str
    model: str
    retrieval_method: str
    latency_seconds: float
    citations: list[Citation]


class DocumentResponse(BaseModel):
    """Response body for a single document."""
    id: int
    filename: str
    file_type: str
    file_size_bytes: int
    num_sections: int
    num_chunks: int
    status: str
    error_message: str | None = None
    uploaded_at: str
    processed_at: str | None = None


class DocumentListResponse(BaseModel):
    """Response body for the document listing endpoint."""
    documents: list[DocumentResponse]
    total: int


class ConversationSummary(BaseModel):
    """Brief summary of a conversation for listing."""
    id: int
    title: str | None = None
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """A single message in a conversation."""
    id: int
    role: str
    content: str
    model: str | None = None
    retrieval_method: str | None = None
    latency_seconds: float | None = None
    citations: list[Citation] | None = None
    created_at: str


class ConversationDetail(BaseModel):
    """Full conversation with all messages."""
    id: int
    title: str | None = None
    created_at: str
    updated_at: str
    messages: list[MessageResponse]


class UploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: int
    filename: str
    status: str
    message: str


class IngestResponse(BaseModel):
    """Response after ingesting documents into the vector store."""
    documents_loaded: int
    sections_extracted: int
    chunks_created: int
    embeddings_generated: int
    embedding_dimension: int
    message: str


class StatsResponse(BaseModel):
    """System statistics."""
    total_documents: int
    processed_documents: int
    failed_documents: int
    total_conversations: int
    total_messages: int
    vector_store_chunks: int | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    vector_store_ready: bool
    database_ready: bool
