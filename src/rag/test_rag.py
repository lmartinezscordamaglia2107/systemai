import os
import sys

# Adiciona o diretório raiz ao path do Python para encontrar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_engine import initialize_vector_store


def test_context_file_exists():
  """Verifies if the context.txt file exists in the data directory."""
  context_path = os.path.join("data", "context.txt")
  assert os.path.exists(context_path), (
      "The context.txt file is missing in the data/ folder!"
  )


def test_vector_store_initialization():
  """Tests if the retriever initializes successfully and returns results."""
  try:
    retriever = initialize_vector_store()
    assert retriever is not None, "Retriever failed to initialize."

    results = retriever.invoke("Quantas séries de panturrilha devo fazer?")
    assert len(results) > 0, "Retriever did not find any chunks for the query."
    assert any("Standing Calf Raises" in doc.page_content for doc in results), (
        "Retriever did not return the calf exercise and set-count context."
    )
  except Exception as e:
    assert False, (
        f"Vector store initialization raised an exception: {e}"
    )