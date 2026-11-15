"""
Database — SQLite persistence layer for the RAG application.

WHY SQLITE:
- Zero-configuration: no database server to install or manage.
- Single file: easy to backup, move, and version.
- Good enough for a single-user research application.
- We'll migrate to PostgreSQL if multi-user support is needed later.

WHAT WE STORE:
- Documents: metadata about uploaded files (name, size, chunk count, upload time).
- Conversations: groups of related Q&A messages.
- Messages: individual questions and answers with citations and timing.
- This enables the frontend to show chat history, document inventory,
  and performance metrics for research analysis.

WHY ASYNC:
- FastAPI is async-native. Using aiosqlite prevents database I/O
  from blocking the event loop, keeping the API responsive while
  queries are being processed by the LLM.
"""

import aiosqlite
import json
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT

# Database file lives in the data/ directory alongside Qdrant storage
DB_PATH = PROJECT_ROOT / "data" / "rag_enterprise.db"


async def get_db() -> aiosqlite.Connection:
    """Get an async database connection.

    Used as a FastAPI dependency so each request gets its own connection.
    Row factory is set to sqlite3.Row so we can access columns by name.
    """
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
    await db.execute("PRAGMA foreign_keys=ON")    # Enforce FK relationships
    return db


async def init_db() -> None:
    """Create all tables if they don't exist.

    Called once at application startup. All tables use IF NOT EXISTS
    so it's safe to call multiple times.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        # ── Documents table ──
        # Tracks every uploaded file and its processing status.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                num_sections INTEGER DEFAULT 0,
                num_chunks INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                uploaded_at TEXT NOT NULL,
                processed_at TEXT
            )
        """)

        # ── Conversations table ──
        # Groups related messages together. Each "chat session" is one conversation.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # ── Messages table ──
        # Stores every question + answer pair with full metadata.
        # Citations are stored as JSON for flexibility.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                model TEXT,
                retrieval_method TEXT,
                latency_seconds REAL,
                citations TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        # ── Indices for common queries ──
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(status)
        """)

        await db.commit()
        print("  Database initialized at:", DB_PATH)


# ══════════════════════════════════════════════════════════════
# CRUD Operations
# ══════════════════════════════════════════════════════════════

def _now() -> str:
    """ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


# ── Documents ──

async def insert_document(
    db: aiosqlite.Connection,
    filename: str,
    file_type: str,
    file_size_bytes: int,
) -> int:
    """Insert a new document record. Returns the new document ID."""
    cursor = await db.execute(
        """INSERT INTO documents (filename, file_type, file_size_bytes, uploaded_at)
           VALUES (?, ?, ?, ?)""",
        (filename, file_type, file_size_bytes, _now()),
    )
    await db.commit()
    return cursor.lastrowid


async def update_document_status(
    db: aiosqlite.Connection,
    doc_id: int,
    status: str,
    num_sections: int = 0,
    num_chunks: int = 0,
    error_message: str | None = None,
) -> None:
    """Update document processing status after ingestion."""
    await db.execute(
        """UPDATE documents
           SET status = ?, num_sections = ?, num_chunks = ?,
               error_message = ?, processed_at = ?
           WHERE id = ?""",
        (status, num_sections, num_chunks, error_message, _now(), doc_id),
    )
    await db.commit()


async def get_all_documents(db: aiosqlite.Connection) -> list[dict]:
    """Fetch all documents ordered by upload time (newest first)."""
    cursor = await db.execute(
        "SELECT * FROM documents ORDER BY uploaded_at DESC"
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_document_by_id(db: aiosqlite.Connection, doc_id: int) -> dict | None:
    """Fetch a single document by ID."""
    cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def delete_document(db: aiosqlite.Connection, doc_id: int) -> bool:
    """Delete a document by ID. Returns True if a row was deleted."""
    cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    await db.commit()
    return cursor.rowcount > 0


# ── Conversations ──

async def create_conversation(
    db: aiosqlite.Connection,
    title: str | None = None,
) -> int:
    """Create a new conversation. Returns the conversation ID."""
    now = _now()
    cursor = await db.execute(
        """INSERT INTO conversations (title, created_at, updated_at)
           VALUES (?, ?, ?)""",
        (title, now, now),
    )
    await db.commit()
    return cursor.lastrowid


async def get_all_conversations(db: aiosqlite.Connection) -> list[dict]:
    """Fetch all conversations ordered by last update (newest first)."""
    cursor = await db.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC"
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_conversation_by_id(
    db: aiosqlite.Connection, conv_id: int
) -> dict | None:
    """Fetch a single conversation by ID."""
    cursor = await db.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


# ── Messages ──

async def insert_message(
    db: aiosqlite.Connection,
    conversation_id: int,
    role: str,
    content: str,
    model: str | None = None,
    retrieval_method: str | None = None,
    latency_seconds: float | None = None,
    citations: list[dict] | None = None,
) -> int:
    """Insert a message into a conversation. Returns the message ID."""
    now = _now()
    citations_json = json.dumps(citations) if citations else None

    cursor = await db.execute(
        """INSERT INTO messages
           (conversation_id, role, content, model, retrieval_method,
            latency_seconds, citations, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (conversation_id, role, content, model, retrieval_method,
         latency_seconds, citations_json, now),
    )

    # Update conversation's updated_at timestamp
    await db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    await db.commit()
    return cursor.lastrowid


async def get_conversation_messages(
    db: aiosqlite.Connection, conversation_id: int
) -> list[dict]:
    """Fetch all messages in a conversation, ordered chronologically."""
    cursor = await db.execute(
        """SELECT * FROM messages
           WHERE conversation_id = ?
           ORDER BY created_at ASC""",
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    messages = []
    for row in rows:
        msg = dict(row)
        # Parse citations JSON back into a list
        if msg.get("citations"):
            msg["citations"] = json.loads(msg["citations"])
        messages.append(msg)
    return messages
