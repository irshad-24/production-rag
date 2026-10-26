
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.api import app
import httpx

# Do not enter the lifespan context: that would load the real Qdrant store.
client = TestClient(app)


def test_health_endpoint():
    app.state.ready = True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ready": True}


def test_empty_question_rejected():
    app.state.ready = True

    response = client.post(
        "/query",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_missing_question_rejected():
    app.state.ready = True

    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_returns_503_when_not_ready():
    app.state.ready = False

    response = client.post(
        "/query",
        json={"question": "What is RAG?"}
    )

    assert response.status_code == 503
    

def test_query_success_with_mocks():
    app.state.ready = True
    app.state.chunks = [
        {
            "page_content": "RAG combines retrieval with generation.",
            "metadata": {
                "chunk_id": 1,
                "page": 1,
                "source_type": "pdf",
            },
        }
    ]
    app.state.vector_store = object()
    app.state.bm25 = object()
    app.state.reranker = object()

    fake_chunk = app.state.chunks[0]

    with (
        patch("app.api.decompose_query", return_value=["What is RAG?"]),
        patch(
            "app.api.hybrid_search",
            return_value=([fake_chunk], None, None, None),
        ),
        patch(
            "app.api.rerank_documents",
            return_value=[(fake_chunk, 0.99)],
        ),
        patch(
            "app.api.expand_hybrid_context",
            return_value=[fake_chunk],
        ),
        patch(
            "app.api.generate_answer",
            return_value="RAG retrieves relevant information before generating an answer.",
        ),
    ):
        response = client.post(
            "/query",
            json={"question": "What is RAG?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "RAG retrieves relevant information" in data["answer"]
    assert data["sources"][0]["chunk_id"] == "1"
    assert data["sources"][0]["page"] == 1
    assert data["sources"][0]["source_type"] == "pdf"
    assert data["latency_seconds"] >= 0


def test_query_handles_groq_rate_limit():
    from groq import RateLimitError

    app.state.ready = True
    app.state.chunks = [
        {
            "page_content": "Sample context",
            "metadata": {"chunk_id": 1, "page": 1, "source_type": "pdf"},
        }
    ]
    app.state.vector_store = object()
    app.state.bm25 = object()
    app.state.reranker = object()

    fake_chunk = app.state.chunks[0]

    with (
        patch("app.api.decompose_query", return_value=["Test question"]),
        patch("app.api.hybrid_search", return_value=([fake_chunk], None, None, None)),
        patch("app.api.rerank_documents", return_value=[(fake_chunk, 0.99)]),
        patch("app.api.expand_hybrid_context", return_value=[fake_chunk]),
        patch(
            "app.api.generate_answer",
            side_effect=RateLimitError(
                "Rate limit reached",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://api.groq.com"),
                ),
                body={"error": "rate_limit_exceeded"},
            ),
        ),
    ):
        response = client.post("/query", json={"question": "Test question"})

    assert response.status_code == 503
    assert "quota exhausted" in response.json()["detail"].lower()
