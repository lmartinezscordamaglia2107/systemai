from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .retriever import DB_DIR, EMBEDDING_MODEL, PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONTEXT_FILE = PROJECT_ROOT / "data" / "context.txt"

def ingest_documents():
    print("1. Loading documents...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    if any(RAW_DATA_DIR.glob("*.pdf")):
        documents.extend(PyPDFDirectoryLoader(str(RAW_DATA_DIR)).load())

    for path in sorted((*RAW_DATA_DIR.glob("*.txt"), *RAW_DATA_DIR.glob("*.md"))):
        documents.extend(TextLoader(str(path), encoding="utf-8").load())

    if CONTEXT_FILE.exists() and not any(
        document.metadata.get("source") == str(CONTEXT_FILE)
        for document in documents
    ):
        documents.extend(TextLoader(str(CONTEXT_FILE), encoding="utf-8").load())

    if not documents:
        print(
            "No documents found. Add PDFs/text files to data/raw or provide "
            "data/context.txt."
        )
        return

    print(f"-> {len(documents)} pages/documents successfully loaded.")

    print("2. Applying chunking (text splitting)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Documents split into {len(chunks)} text chunks.")

    print("3. Initializing Embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("4. Creating and persisting Vector Database (ChromaDB)...")
    DB_DIR.parent.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    
    print(f"Ingestion process completed! Vector database persisted at: {DB_DIR}")

if __name__ == "__main__":
    ingest_documents()