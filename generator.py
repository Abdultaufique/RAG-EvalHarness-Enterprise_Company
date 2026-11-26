"""
Generator — sends retrieved context + question to an LLM.

DUAL-MODE OPERATION:
  1. LOCAL mode  → Uses Ollama (Llama 3.1 8B) at localhost:11434
  2. CLOUD mode  → Uses Google Gemini 1.5 Flash API (free tier)

The system auto-detects which to use:
  - If GEMINI_API_KEY is set in env/secrets → Cloud (Gemini)
  - Otherwise → Local (Ollama)

This allows the same codebase to run:
  - Locally on your machine (Ollama, full privacy)
  - On Streamlit Cloud (Gemini API, free tier)
"""

import json
import os
import time
import httpx

from config import LLMConfig


# ── Shared prompt templates ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, evidence-based assistant. Your job is to answer questions using ONLY the provided source documents.

RULES:
1. Base your answer ONLY on the provided source chunks. Do not use outside knowledge.
2. After each claim, cite the source using [Source: filename, Chunk #N] format.
3. If the provided sources do not contain enough information to answer the question, say: "I cannot answer this question based on the available documents."
4. Be concise but thorough. Aim for 2-4 sentences unless more detail is needed.
5. If sources provide conflicting information, mention the conflict and cite both sources."""


def build_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
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


# ── Ollama backend ───────────────────────────────────────────────────────────

def _generate_ollama(config: LLMConfig, user_prompt: str) -> tuple[str, str]:
    """Call Ollama local API. Returns (answer_text, model_name)."""
    api_url = f"{config.base_url}/api/chat"
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }
    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()

    result = response.json()
    answer = result.get("message", {}).get("content", "No response from model.")
    return answer, config.model


# ── Gemini backend ───────────────────────────────────────────────────────────

def _generate_gemini(user_prompt: str, api_key: str) -> tuple[str, str]:
    """Call Google Gemini API. Returns (answer_text, model_name)."""
    # Try models in order until one works
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-002",
    ]

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
        },
    }

    last_error = None
    for model_name in models_to_try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 404:
                    last_error = f"Model {model_name} not found, trying next..."
                    continue
                response.raise_for_status()

            data = response.json()
            try:
                answer = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                answer = "No response from Gemini model."
            return answer, f"gemini/{model_name}"

        except httpx.HTTPStatusError as e:
            last_error = f"{model_name}: HTTP {e.response.status_code}"
            continue
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


# ── Generator class ──────────────────────────────────────────────────────────

class Generator:
    """
    Generates answers using an LLM — auto-selects Ollama or Gemini.

    Priority:
      1. GEMINI_API_KEY env var / Streamlit secret → use Gemini Flash
      2. OLLAMA_BASE_URL reachable             → use Ollama (Llama 3.1)
      3. Neither available                     → return helpful error
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._gemini_key: str | None = self._get_gemini_key()
        self._mode: str = "gemini" if self._gemini_key else "ollama"

    @staticmethod
    def _get_gemini_key() -> str | None:
        """Check env var, then Streamlit secrets."""
        # 1. Environment variable (works locally & on Streamlit Cloud)
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if key:
            return key

        # 2. Streamlit secrets (only available inside a Streamlit app)
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "").strip()
            if key:
                return key
        except Exception:
            pass

        return None

    @property
    def mode(self) -> str:
        """Return 'gemini' or 'ollama'."""
        return self._mode

    def generate(self, question: str, retrieved_chunks: list[dict]) -> dict:
        """
        Generate an answer using the best available LLM.

        Returns dict with:
          answer, model, citations, latency_seconds, prompt, llm_mode
        """
        context     = build_context(retrieved_chunks)
        user_prompt = build_user_prompt(question, context)
        start       = time.time()

        # ── Try Gemini first (if key present) ──
        if self._gemini_key:
            try:
                answer, model = _generate_gemini(user_prompt, self._gemini_key)
                return {
                    "answer":          answer,
                    "model":           model,
                    "citations":       retrieved_chunks,
                    "latency_seconds": round(time.time() - start, 2),
                    "prompt":          user_prompt,
                    "llm_mode":        "gemini",
                }
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    answer = (
                        "⚠️ Gemini API rate limit hit. "
                        "Please wait a moment and try again."
                    )
                else:
                    answer = f"⚠️ Gemini API error {e.response.status_code}: {e.response.text[:200]}"
                return {
                    "answer": answer, "model": "gemini/error",
                    "citations": retrieved_chunks,
                    "latency_seconds": round(time.time() - start, 2),
                    "prompt": user_prompt, "llm_mode": "gemini_error",
                }
            except Exception as e:
                answer = f"⚠️ Gemini error: {e}"
                return {
                    "answer": answer, "model": "gemini/error",
                    "citations": retrieved_chunks,
                    "latency_seconds": round(time.time() - start, 2),
                    "prompt": user_prompt, "llm_mode": "gemini_error",
                }

        # ── Fall back to Ollama ──
        try:
            answer, model = _generate_ollama(self.config, user_prompt)
            return {
                "answer":          answer,
                "model":           model,
                "citations":       retrieved_chunks,
                "latency_seconds": round(time.time() - start, 2),
                "prompt":          user_prompt,
                "llm_mode":        "ollama",
            }
        except httpx.ConnectError:
            return {
                "answer": (
                    "❌ No LLM available.\n\n"
                    "**Local mode**: Make sure Ollama is running (`ollama serve`).\n"
                    "**Cloud mode**: Set `GEMINI_API_KEY` in Streamlit secrets."
                ),
                "model": "unavailable",
                "citations": retrieved_chunks,
                "latency_seconds": round(time.time() - start, 2),
                "prompt": user_prompt,
                "llm_mode": "none",
            }
        except httpx.HTTPStatusError as e:
            return {
                "answer": (
                    f"❌ Ollama returned status {e.response.status_code}. "
                    f"Is model '{self.config.model}' pulled? Run: `ollama pull llama3.1`"
                ),
                "model": self.config.model,
                "citations": retrieved_chunks,
                "latency_seconds": round(time.time() - start, 2),
                "prompt": user_prompt,
                "llm_mode": "ollama_error",
            }
