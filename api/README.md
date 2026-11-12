# API HTTP do treinador RAG

Instale as dependências na raiz do projeto:

```powershell
python -m pip install -r config/requirements.txt
```

Crie o índice vetorial multilíngue antes de usar a busca ou o chat:

```powershell
python -m src.rag.ingestion
```

O índice novo fica em `data/processed/chroma_db_multilingual`; o índice antigo é preservado.

Configure `OPENAI_API_KEY` no `.env` local e inicie o servidor apenas na máquina local:

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

- `GET http://127.0.0.1:8000/health` verifica se a API está ativa.
- `POST http://127.0.0.1:8000/chat` recebe `{"question":"..."}` e devolve `{"answer":"..."}`.
- `POST http://127.0.0.1:8000/search` recebe `{"query":"...","k":3}` e devolve trechos que mudam conforme a consulta, além de arquivo, página e distância de similaridade.
- Use `/chat` para uma resposta do treinador; `/search` retorna os trechos de referência sem gerar uma resposta.
- `http://127.0.0.1:8000/docs` mostra a documentação interativa.

Não publique o servidor na rede sem adicionar autenticação e controles de acesso.
