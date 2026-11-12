import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .retriever import get_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "llm" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)


def format_docs(docs):
    """Helper function to concatenate retrieved document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_personal_trainer_chain():
    """Initializes the vector store retriever and builds the RAG chain for the AI Personal Trainer."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your environment or project .env file."
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=api_key,
    )

    template = """You are an expert AI Personal Trainer specializing in advanced hypertrophy, biomechanics, and personalized workout structuring.
Your mission is to provide safe, highly efficient, and tailored training strategies based strictly on the provided context.

Context from Knowledge Base & User Profile:
{context}

Student/User Question:
{question}

Instructions:
- Answer with technical authority, a motivational tone, and keep it straight to the point.
- Strictly adhere to user constraints and biomechanical restrictions mentioned in the context (e.g., equipment alternatives).
- Respond in Portuguese (or match the language of the user's question) while keeping the technical code and architecture in English.
"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    print("=== Initializing AI Personal Trainer Agent ===")
    trainer_ai = create_personal_trainer_chain()

    test_question = (
        "Monte uma estratégia de treino de pernas focada em quadríceps"
        " respeitando minhas restrições."
    )
    print(f"\n[User]: {test_question}\n")

    response = trainer_ai.invoke(test_question)
    print(f"[AI Personal Trainer]:\n{response}")