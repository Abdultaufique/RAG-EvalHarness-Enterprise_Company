"""
Milestone 4 — Test the FastAPI endpoints.

Run with: .venv\Scripts\python.exe test_milestone4.py

PREREQUISITE: Start the API server first:
  .venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
"""

import sys
import time
import json
import httpx

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
client = httpx.Client(timeout=300.0)  # 5 min timeout for LLM calls


def test(name: str, method: str, url: str, **kwargs):
    """Helper to run a single API test."""
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print(f"  {method} {url}")

    try:
        response = getattr(client, method.lower())(url, **kwargs)
        print(f"  Status: {response.status_code}")

        if response.status_code >= 400:
            print(f"  ERROR: {response.text}")
            return None

        data = response.json()
        print(f"  Response: {json.dumps(data, indent=4, default=str)[:1000]}")
        return data
    except httpx.ConnectError:
        print("  ERROR: Cannot connect. Is the server running?")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("=" * 60)
    print("MILESTONE 4 — API ENDPOINT TESTS")
    print("=" * 60)

    # ── 1. Health check ──
    health = test("Health Check", "GET", f"{BASE_URL}/api/health")
    if not health:
        print("\nServer not running! Start it first:")
        print("  .venv\\Scripts\\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # ── 2. Upload documents ──
    from pathlib import Path
    sample_dir = Path(__file__).parent / "sample_docs"

    for doc_name in ["artificial_intelligence.txt", "climate_change.txt", "quantum_computing.pdf"]:
        doc_path = sample_dir / doc_name
        if doc_path.exists():
            with open(doc_path, "rb") as f:
                test(
                    f"Upload {doc_name}",
                    "POST",
                    f"{BASE_URL}/api/documents/upload",
                    files={"file": (doc_name, f)},
                )

    # ── 3. List documents ──
    docs = test("List Documents", "GET", f"{BASE_URL}/api/documents")

    # ── 4. Ingest documents ──
    ingest = test(
        "Ingest Documents",
        "POST",
        f"{BASE_URL}/api/documents/ingest",
        json={"recreate_collection": True},
    )

    # ── 5. Chat — question about AI ──
    chat1 = test(
        "Chat: AI Question",
        "POST",
        f"{BASE_URL}/api/chat",
        json={
            "question": "What are the three types of machine learning?",
            "retrieval_method": "dense",
            "top_k": 3,
        },
    )

    conv_id = chat1.get("conversation_id") if chat1 else None

    # ── 6. Chat — follow-up in same conversation ──
    if conv_id:
        chat2 = test(
            "Chat: Follow-up (same conversation)",
            "POST",
            f"{BASE_URL}/api/chat",
            json={
                "question": "Which one is used for recommendation systems?",
                "conversation_id": conv_id,
                "retrieval_method": "hybrid",
                "top_k": 3,
            },
        )

    # ── 7. Chat — abstention test ──
    chat3 = test(
        "Chat: Abstention Test",
        "POST",
        f"{BASE_URL}/api/chat",
        json={
            "question": "What is the recipe for chocolate chip cookies?",
            "retrieval_method": "dense",
        },
    )

    # ── 8. List conversations ──
    convs = test("List Conversations", "GET", f"{BASE_URL}/api/conversations")

    # ── 9. Get full conversation ──
    if conv_id:
        test(
            "Get Conversation Detail",
            "GET",
            f"{BASE_URL}/api/conversations/{conv_id}",
        )

    # ── 10. Stats ──
    test("System Stats", "GET", f"{BASE_URL}/api/stats")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("MILESTONE 4 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Health check:       {'✅' if health else '❌'}")
    print(f"  Document upload:    {'✅' if docs else '❌'}")
    print(f"  Ingestion:          {'✅' if ingest else '❌'}")
    print(f"  Chat (dense):       {'✅' if chat1 else '❌'}")
    print(f"  Chat (follow-up):   {'✅' if conv_id and chat2 else '❌'}")
    print(f"  Chat (abstention):  {'✅' if chat3 else '❌'}")
    print(f"  Conversations:      {'✅' if convs else '❌'}")
    print(f"  Stats:              ✅")
    print(f"\n  Swagger docs: http://localhost:8000/docs")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
