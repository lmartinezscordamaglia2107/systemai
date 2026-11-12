import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "llm" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DB_DIR = Path(
    os.getenv(
        "CHROMA_DB_DIR",
        PROJECT_ROOT / "data" / "processed" / "chroma_db_multilingual",
    )
).expanduser().resolve()


def get_vector_store():
    """Loads the persisted Chroma vector database."""
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Database directory not found at {DB_DIR}. Please run ingestion.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    return vector_store


def retrieve_documents(query: str, k: int = 3):
    """Returns the top 'k' matching chunks and their similarity distances."""
    return get_vector_store().similarity_search_with_score(query, k=k)


def search_documents(query: str, k: int = 3):
    """Searches the vector database for the top 'k' most relevant chunks."""
    results = retrieve_documents(query, k=k)

    print(f"\n--- Search Results for Query: '{query}' ---")
    for i, (doc, score) in enumerate(results, 1):
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", "N/A")

        print(f"\n[Result {i}] (Score/Distance: {score:.4f})")
        print(f"Source: {source_file} | Page: {page_num}")
        print(f"Snippet: {doc.page_content.strip()}")
        print("-" * 60)

    return results


if __name__ == "__main__":
    sample_query = "What is the optimal weekly training volume and frequency for muscle hypertrophy?"
    search_documents(sample_query, k=3)