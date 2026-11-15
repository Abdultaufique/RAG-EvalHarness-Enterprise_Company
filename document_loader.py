"""
Document Loader — extracts text from PDF, DOCX, and TXT files.

WHY THIS MODULE EXISTS:
Before we can chunk and embed text, we need to extract it from various
file formats. Each format needs different handling:
  - TXT: Read directly (simplest case).
  - PDF: Use PyMuPDF (fitz) to extract text page by page.
  - DOCX: Use python-docx to extract text paragraph by paragraph.

We also preserve SOURCE METADATA (filename, page number) so that
retrieved chunks can be traced back to their exact origin — this is
critical for citations in the final RAG answers.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    """Represents a piece of text with its source metadata.

    text: The actual text content.
    source_file: Name of the original file.
    page_number: Page number (1-indexed) or None for formats without pages.
    chunk_index: Position of this chunk in the document (set during chunking).
    """
    text: str
    source_file: str
    page_number: int | None = None
    chunk_index: int | None = None

    def to_metadata(self) -> dict:
        """Convert to a flat dictionary for storage in Qdrant."""
        return {
            "source_file": self.source_file,
            "page_number": self.page_number if self.page_number is not None else -1,
            "chunk_index": self.chunk_index if self.chunk_index is not None else -1,
            "text": self.text,
        }


def load_txt(file_path: Path) -> list[DocumentChunk]:
    """Load a plain text file.

    TXT files don't have pages, so page_number is None.
    We return the entire file as a single DocumentChunk — chunking
    happens in the next step.
    """
    text = file_path.read_text(encoding="utf-8")
    return [DocumentChunk(text=text, source_file=file_path.name, page_number=None)]


def load_pdf(file_path: Path) -> list[DocumentChunk]:
    """Load a PDF file, extracting text page by page.

    WHY PAGE-BY-PAGE:
    - Preserves page numbers for citations.
    - Keeps each page as a separate chunk input, so the chunker
      knows which page each piece of text came from.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(file_path))
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():  # Skip empty pages
            chunks.append(DocumentChunk(
                text=text,
                source_file=file_path.name,
                page_number=page_num + 1,  # 1-indexed for human readability
            ))
    doc.close()
    return chunks


def load_docx(file_path: Path) -> list[DocumentChunk]:
    """Load a DOCX file, extracting text from paragraphs.

    DOCX files don't have true page numbers accessible via python-docx,
    so we set page_number to None. We concatenate all paragraphs into
    a single chunk — the chunker handles splitting.
    """
    from docx import Document

    doc = Document(str(file_path))
    full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    if not full_text.strip():
        return []
    return [DocumentChunk(text=full_text, source_file=file_path.name, page_number=None)]


def load_document(file_path: str | Path) -> list[DocumentChunk]:
    """Load a document based on its file extension.

    This is the main entry point — it detects the format and
    calls the appropriate loader.

    Args:
        file_path: Path to the document file.

    Returns:
        List of DocumentChunks (one per page for PDFs, one for TXT/DOCX).

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()
    loaders = {
        ".txt": load_txt,
        ".pdf": load_pdf,
        ".docx": load_docx,
    }

    loader = loaders.get(extension)
    if loader is None:
        raise ValueError(
            f"Unsupported file format: '{extension}'. "
            f"Supported formats: {', '.join(loaders.keys())}"
        )

    chunks = loader(path)
    print(f"  Loaded {path.name}: {len(chunks)} section(s), "
          f"{sum(len(c.text) for c in chunks)} chars total")
    return chunks
