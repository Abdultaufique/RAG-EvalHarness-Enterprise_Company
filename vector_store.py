"""
Vector Store — manages Qdrant for storing and searching document embeddings.

WHY A VECTOR STORE:
After generating embeddings for all chunks, we need a way to efficiently
find the most similar ones when a user asks a question. A vector store
(also called a vector database) does exactly this — it indexes vectors
and supports fast nearest-neighbor search.

QDRANT specifically:
- Runs locally without a server (file-based storage mode).
- Supports cosine similarity, dot product, and Euclidean distance.
- Stores metadata (source file, page number, text) alongside vectors.
- Scales well if we later need to handle thousands of documents.
"""

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config import QdrantConfig, EmbeddingConfig
from document_loader import DocumentChunk


class VectorStore:
    """Manages a Qdrant collection for document chunk embeddings."""

    def __init__(self, qdrant_config: QdrantConfig, embedding_config: EmbeddingConfig):
        """Initialize Qdrant client with local file storage.

        No Qdrant server needed — data is stored in the specified path.
        """
        self.config = qdrant_config
        self.embedding_dim = embedding_config.dimension

        print(f"  Initializing Qdrant at: {qdrant_config.path}")
        self.client = QdrantClient(path=qdrant_config.path)

    def create_collection(self, recreate: bool = False) -> None:
        """Create the vector collection if it doesn't exist.

        Args:
            recreate: If True, delete and recreate the collection.
                      Useful during development and experiments.
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if self.config.collection_name in collections:
            if recreate:
                print(f"  Recreating collection: {self.config.collection_name}")
                # Delete via API instead of nuking the directory — avoids
                # Windows file-locking errors on Qdrant's internal SQLite.
                self.client.delete_collection(self.config.collection_name)
            else:
                print(f"  Collection '{self.config.collection_name}' already exists.")
                return

        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"  Created collection '{self.config.collection_name}' "
              f"(dim={self.embedding_dim}, distance=cosine)")

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Store document chunks and their embeddings in Qdrant.

        Each chunk becomes a "point" in Qdrant with:
          - id: A unique integer identifier.
          - vector: The embedding array.
          - payload: Metadata (source file, page, text, chunk index).

        Args:
            chunks: List of DocumentChunks to store.
            embeddings: numpy array of shape (len(chunks), dimension).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Get current point count to generate unique IDs
        collection_info = self.client.get_collection(self.config.collection_name)
        start_id = collection_info.points_count

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=start_id + i,
                vector=embedding.tolist(),
                payload=chunk.to_metadata(),
            ))

        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )
        print(f"  Stored {len(points)} chunks in Qdrant "
              f"(IDs {start_id} to {start_id + len(points) - 1})")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for the most similar chunks to a query vector.

        Args:
            query_vector: The embedding of the user's question.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: 'text', 'source_file', 'page_number',
            'chunk_index', 'score'.
        """
        results = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector.tolist(),
            limit=top_k * 2,  # Fetch extra to allow for dedup
            with_payload=True,
        )

        # Deduplicate by (source_file, chunk_index) — keeps highest-scored
        seen = set()
        deduped = []
        for point in results.points:
            key = (point.payload["source_file"], point.payload.get("chunk_index", -1))
            if key not in seen:
                seen.add(key)
                deduped.append(point)
            if len(deduped) >= top_k:
                break

        return [
            {
                "text": point.payload["text"],
                "source_file": point.payload["source_file"],
                "page_number": point.payload.get("page_number", -1),
                "chunk_index": point.payload.get("chunk_index", -1),
                "score": point.score,
            }
            for point in deduped
        ]

    def get_all_chunks(self) -> list[dict]:
        """Retrieve all stored chunks (used for BM25 indexing).

        Returns:
            List of dicts with metadata and text for every stored chunk.
        """
        collection_info = self.client.get_collection(self.config.collection_name)
        total = collection_info.points_count

        if total == 0:
            return []

        results = self.client.scroll(
            collection_name=self.config.collection_name,
            limit=total,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": point.id,
                "text": point.payload["text"],
                "source_file": point.payload["source_file"],
                "page_number": point.payload.get("page_number", -1),
                "chunk_index": point.payload.get("chunk_index", -1),
            }
            for point in results[0]
        ]

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self.client.close()
