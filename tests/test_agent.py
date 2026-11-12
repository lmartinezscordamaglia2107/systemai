import os
import sys
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

# Adiciona o diretório raiz ao path do Python para encontrar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import create_personal_trainer_chain


def test_agent_chain_creation(monkeypatch):
  """Tests if the personal trainer RAG chain builds without errors."""
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")
  monkeypatch.setattr(
      "src.rag.engine.get_vector_store",
      lambda: SimpleNamespace(
          as_retriever=lambda **kwargs: RunnableLambda(lambda _: [])
      ),
  )

  chain = create_personal_trainer_chain()
  assert chain is not None, "Agent chain is None."