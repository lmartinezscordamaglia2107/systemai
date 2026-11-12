import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.retriever import retrieve_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "llm" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

from src.agent import create_personal_trainer_chain

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Personal Trainer API", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    content: str
    source: str
    page: int | str | None
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


@lru_cache(maxsize=1)
def get_trainer_chain():
    return create_personal_trainer_chain()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="A consulta não pode ficar vazia.")

    try:
        matches = retrieve_documents(query, k=request.k)
    except Exception as exc:
        logger.exception("Falha ao pesquisar na base vetorial.")
        raise HTTPException(
            status_code=502,
            detail="Falha ao pesquisar na base vetorial. Verifique a configuração e os logs do servidor.",
        ) from exc

    results = [
        SearchResult(
            content=document.page_content,
            source=os.path.basename(str(document.metadata.get("source", "Unknown"))),
            page=document.metadata.get("page"),
            distance=float(distance),
        )
        for document, distance in matches
    ]
    return SearchResponse(results=results)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="A pergunta não pode ficar vazia.")
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY não está configurada no ambiente do servidor.",
        )

    try:
        answer = get_trainer_chain().invoke(question)
    except Exception as exc:
        logger.exception("Falha ao processar pergunta no treinador RAG.")
        raise HTTPException(
            status_code=502,
            detail="Falha ao consultar o treinador. Verifique a configuração e os logs do servidor.",
        ) from exc

    if not isinstance(answer, str) or not answer.strip():
        logger.error("A cadeia RAG retornou uma resposta vazia ou inválida.")
        raise HTTPException(
            status_code=502,
            detail="O treinador não retornou uma resposta válida.",
        )

    return ChatResponse(answer=answer)
