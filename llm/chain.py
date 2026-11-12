import os
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.retriever import get_vector_store

# 1. Define the Professional Personal Trainer System Prompt
TRAINER_SYSTEM_PROMPT = """You are an elite, science-based Personal Trainer and Exercise Physiologist AI. 
Your goal is to provide safe, highly effective, evidence-based advice on training, hypertrophy, biomechanics, and sports nutrition.

Rules you must follow strictly:
1. **Evidence-Based:** Base your recommendations strictly on the provided context (extracted from exercise science textbooks and research). If the answer cannot be found in the context, use your general sports science knowledge, but prioritize the retrieved context.
2. **Safety First:** Always emphasize proper form, joint health, warm-ups, and injury prevention. If a user asks about heavy or dangerous lifts, include safety warnings.
3. **Tone:** Be encouraging, disciplined, highly authoritative, and professional—like a world-class coach talking to an dedicated athlete.
4. **Clarity:** Break down workout structures, rep ranges, and sets clearly using bullet points or numbered lists.

Context from scientific literature and manuals:
{context}

User's Question:
{question}
"""

def format_docs(docs):
    """Helper function to format retrieved documents into a single text string."""
    return "\n\n".join(doc.page_content for doc, _ in docs)

def create_trainer_chain(k_docs: int = 3):
    """Creates the RAG pipeline linking the vector retriever, prompt, and Gemini LLM."""
    
    # 2. Initialize the Vector Store Retriever
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})
    
    # 3. Initialize the Gemini LLM (using gemini-2.5-flash or gemini-1.5-pro)
    # Ensure GOOGLE_API_KEY is set in your environment variables
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3  # Low temperature for precise, factual fitness advice
    )
    
    # 4. Create the Chat Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRAINER_SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    # 5. Build the LangChain Expression Language (LCEL) chain
    # This pipeline automatically fetches docs, formats them, passes them to the prompt, and queries Gemini
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    # Test your complete AI Personal Trainer RAG pipeline
    print("Initializing AI Personal Trainer Chain...")
    chain = create_trainer_chain(k_docs=3)
    
    test_question = "What is the recommended repetition range and volume for maximizing muscle hypertrophy according to science?"
    print(f"\nUser: {test_question}\n")
    print("Coach AI is analyzing the literature and generating a response...\n")
    
    response = chain.invoke(test_question)
    print("--- Coach AI Response ---")
    print(response)