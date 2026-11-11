# 🏋️‍♂️ AI Personal Trainer — RAG System

Um assistente virtual especializado em Educação Física, Biomecânica, Hipertrofia e Nutrição Esportiva. O sistema utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para consultar uma base vetorial de literatura científica (livros, diretrizes de treinamento e manuais de fisiologia) e responder a dúvidas de alunos sem sofrer alucinações.

---

## 🚀 Arquitetura & Principais Funcionalidades

- **Base de Conhecimento Científica (RAG):** Leitura, fatiamento (*chunking*) e indexação de PDFs teóricos (ex.: ACSM, NSCA, estudos de hipertrofia) para embasamento rigoroso das respostas.
- **Banco Vetorial Local:** Armazenamento e busca por similaridade semântica de alta velocidade utilizando **ChromaDB**.
- **Modelos de Linguagem & Embeddings:** Integração com **Google Gemini** (`gemini-2.5-flash`) via LangChain e embeddings locais via **HuggingFace** (`all-MiniLM-L6-v2`).
- **Prompt Especializado:** System prompt configurado para atuar como um treinador de alto nível, priorizando a segurança articulares, execução correta dos exercícios e periodização eficiente.

---

## 🛠️ Tecnologias Utilizadas

| Componente | Tecnologia / Biblioteca |
| :--- | :--- |
| **Linguagem** | Python 3.10+ |
| **Framework LLM & RAG** | LangChain / `langchain-google-genai` |
| **Banco Vetorial** | ChromaDB |
| **Modelos de Embeddings** | HuggingFace (`sentence-transformers`) |
| **Modelo da IA (LLM)** | Google Gemini API (`gemini-2.5-flash`) |
| **Leitor de Documentos** | `PyPDFDirectoryLoader` |

---

## 📂 Estrutura do Projeto

```text
personal-trainer-ai/
├── data/
│   ├── raw/                  # Coloque aqui os PDFs brutos (livros, artigos, manuais)
│   └── processed/            # Banco vetorial ChromaDB persistido (.gitignored)
│
├── config/
│   ├── settings.py           # Configurações globais e caminhos
│   └── prompts.py            # Templates de System Prompts do treinador
│
├── src/
│   ├── __init__.py
│   ├── rag/                  # Módulo de RAG
│   │   ├── __init__.py
│   │   ├── ingestion.py      # Processamento de PDFs e geração dos embeddings
│   │   └── retriever.py      # Mecanismo de busca vetorial por similaridade
│   │
│   ├── llm/                  # Pipeline com LLM
│   │   ├── __init__.py
│   │   └── chain.py          # Encadeamento LangChain (Retriever + Prompt + Gemini)
│   │
│   └── agent/                # Memória e ferramentas de cálculo (TDEE, 1RM, etc.)
│       ├── __init__.py
│       └── tools.py
│
├── .env.example              # Modelo de variáveis de ambiente
├── .gitignore
├── requirements.txt          # Dependências do Python
└── README.md                 # Documentação do projeto
