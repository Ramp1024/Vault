# Vault

Vault uses:
- Backend API (FastAPI)
- Frontend UI (Vite + React)
- Qdrant (Docker)
- Ollama running natively in WSL

## Prerequisites

- Docker and Docker Compose
- Ollama installed and running in WSL

### Install Ollama in WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Ollama Base URL

The backend uses `OLLAMA_BASE_URL`, defaulting to `http://localhost:11434`.

Optional model overrides:

- `EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `GENERATION_MODEL` (default: `llama3.1:8b`)

## Start the stack

Start Ollama in WSL first:

```bash
ollama serve
```

Then run the application stack. Docker Compose starts only the application services and does not manage Ollama:

```bash
docker-compose up --build
```
