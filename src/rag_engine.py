from src.rag.engine import create_personal_trainer_chain
from src.rag.retriever import get_vector_store, search_documents

__all__ = ["create_personal_trainer_chain", "get_vector_store", "search_documents"]


def initialize_vector_store():
    return get_vector_store().as_retriever(search_kwargs={"k": 3})
