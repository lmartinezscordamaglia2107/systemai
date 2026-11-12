import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "llm" / ".env"):
  if env_path.exists():
    load_dotenv(env_path, override=False)

# Verify that the OpenAI API key is configured.
if not os.getenv("OPENAI_API_KEY"):
  print(
      "[Error]: OPENAI_API_KEY is missing. Set it in a .env file or system environment."
  )
  sys.exit(1)

# Import the personal trainer agent chain from the src package
from src.agent import create_personal_trainer_chain


def main():
  print("=" * 60)
  print("🚀 INITIALIZING AI PERSONAL TRAINER (RAG-ENABLED)")
  print("=" * 60)
  print("Loading knowledge base and building RAG pipeline...\n")

  try:
    # Initialize the RAG agent chain
    trainer_ai = create_personal_trainer_chain()
    print("✅ AI Personal Trainer is ready!")
    print(
        "Type your questions below (e.g., 'Monte meu treino de pernas' or"
        " 'type exit to quit').\n"
    )
    print("-" * 60)

    # Interactive chat loop in terminal
    while True:
      user_input = input("\n[You]: ").strip()

      if not user_input:
        continue

      if user_input.lower() in ["exit", "quit", "sair"]:
        print(
            "\n[AI Personal Trainer]: Keep pushing hard! Goodbye and good"
            " training! 💪"
        )
        break

      print("\n[AI Personal Trainer thinking...]")
      try:
        # Invoke the RAG chain with user input
        response = trainer_ai.invoke(user_input)
        print(f"\n[AI Personal Trainer]:\n{response}")
        print("-" * 60)
      except Exception as e:
        print(f"\n[Error processing request]: {e}")

  except Exception as e:
    print(f"\n[Initialization Error]: {e}")
    print(
        "Make sure you have run your processors.py and populated data/processed/"
        " before running main.py."
    )


if __name__ == "__main__":
  main()