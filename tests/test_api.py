from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import app as api_module

client = TestClient(api_module.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_retrieved_chunks(monkeypatch):
    def fake_retrieve_documents(query, k):
        assert query == "hipertrofia"
        assert k == 2
        return [
            (
                Document(
                    page_content="Treino de hipertrofia com técnica.",
                    metadata={"source": r"docs\guia.pdf", "page": 4},
                ),
                0.12,
            )
        ]

    monkeypatch.setattr(api_module, "retrieve_documents", fake_retrieve_documents)

    response = client.post(
        "/search",
        json={"query": " hipertrofia ", "k": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "content": "Treino de hipertrofia com técnica.",
                "source": "guia.pdf",
                "page": 4,
                "distance": 0.12,
            }
        ]
    }


def test_search_rejects_invalid_result_count():
    response = client.post("/search", json={"query": "hipertrofia", "k": 0})

    assert response.status_code == 422


def test_chat_invokes_trainer_chain(monkeypatch):
    class FakeChain:
        def invoke(self, question):
            assert question == "Como treino pernas?"
            return "Treine com técnica e progressão."

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_module, "get_trainer_chain", lambda: FakeChain())

    response = client.post("/chat", json={"question": "  Como treino pernas?  "})

    assert response.status_code == 200
    assert response.json() == {"answer": "Treine com técnica e progressão."}


def test_chat_rejects_blank_question():
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_reports_upstream_failure(monkeypatch):
    class FailingChain:
        def invoke(self, question):
            raise RuntimeError("model unavailable")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_module, "get_trainer_chain", lambda: FailingChain())

    response = client.post("/chat", json={"question": "Pergunta de teste"})

    assert response.status_code == 502
    assert "Falha ao consultar" in response.json()["detail"]
