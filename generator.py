"""
Generator — sends retrieved context + question to an LLM via Ollama.

WHY THIS MODULE EXISTS:
This is the "G" in RAG (Retrieval-Augmented Generation). After retrieving
relevant chunks, we need to:

1. Construct a PROMPT that includes the retrieved evidence and the question.
2. Send it to the LLM (via Ollama's REST API).
3. Parse the response and return the answer with citations.

KEY DESIGN DECISIONS:
- We use Ollama's HTTP API directly (via httpx) instead of a Python SDK
  to minimize dependencies and keep full control over the request.
- The prompt explicitly instructs the LLM to cite sources and abstain
  when evidence is insufficient — this is critical for trustworthy RAG.
- We track which source chunks were used so the UI can display citations.
"""

import json
import time
import httpx

from config import LLMConfig


# The system prompt instructs the LLM how to behave.
# This is separate from the user's question — it sets the "role" of the assistant.
SYSTEM_PROMPT = """You are a precise, evidence-based assistant. Your job is to answer questions using ONLY the provided source documents.

RULES:
1. Base your answer ONLY on the provided source chunks. Do not use outside knowledge.
2. After each claim, cite the source using [Source: filename, Chunk #N] format.
3. If the provided sources do not contain enough information to answer the question, say: "I cannot answer this question based on the available documents."
4. Be concise but thorough. Aim for 2-4 sentences unless more detail is needed.
5. If sources provide conflicting information, mention the conflict and cite both sources."""


def build_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM.

    Each chunk is labeled with its source and index so the LLM can
    reference them in citations.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk["source_file"]
        chunk_idx = chunk.get("chunk_index", i)
        page = chunk.get("page_number", -1)
        page_info = f", Page {page}" if page and page > 0 else ""

        context_parts.append(
            f"--- Source: {source}{page_info}, Chunk #{chunk_idx} ---\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(context_parts)


def build_user_prompt(question: str, context: str) -> str:
    """Combine the question and context into the user prompt."""
    return (
        f"SOURCE DOCUMENTS:\n\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Provide a detailed answer based on the source documents above. "
        f"Cite each source you use."
    )


class Generator:
    """Generates answers using an LLM via Ollama's REST API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_url = f"{config.base_url}/api/chat"

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> dict:
        """Generate an answer to the question using retrieved context.

        Args:
            question: The user's natural language question.
            retrieved_chunks: List of chunk dicts from the retriever.

        Returns:
            Dict with keys:
              - 'answer': The LLM's response text.
              - 'model': Which model generated it.
              - 'citations': The source chunks that were provided as context.
              - 'latency_seconds': How long generation took.
              - 'prompt': The full prompt sent (for debugging/evaluation).
        """
        context = build_context(retrieved_chunks)
        user_prompt = build_user_prompt(question, context)

        # Ollama chat API payload
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        start_time = time.time()

        try:
            # Increased timeout for CPU inference (can be slow)
            with httpx.Client(timeout=300.0) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError:
            return {
                "answer": "ERROR: Cannot connect to Ollama. "
                          "Make sure Ollama is running (run 'ollama serve' in a terminal).",
                "model": self.config.model,
                "citations": retrieved_chunks,
                "latency_seconds": time.time() - start_time,
                "prompt": user_prompt,
            }
        except httpx.HTTPStatusError as e:
            return {
                "answer": f"ERROR: Ollama returned status {e.response.status_code}. "
                          f"Is model '{self.config.model}' pulled?",
                "model": self.config.model,
                "citations": retrieved_chunks,
                "latency_seconds": time.time() - start_time,
                "prompt": user_prompt,
            }

        latency = time.time() - start_time
        result = response.json()
        answer = result.get("message", {}).get("content", "No response from model.")

        return {
            "answer": answer,
            "model": self.config.model,
            "citations": retrieved_chunks,
            "latency_seconds": round(latency, 2),
            "prompt": user_prompt,
        }
