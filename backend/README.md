# Vault Backend

## Prerequisites

- Ollama installed and running in WSL
- Required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

Set environment variables as needed:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
export GENERATION_MODEL=llama3.1:8b
```

## Setup with uv

Create or refresh the virtual environment and install dependencies:

```bash
uv sync
```

Run the API in development mode:

```bash
uv run uvicorn app.main:app --reload
```

Synchronize Notion documents into Qdrant explicitly:

```bash
uv run python -m app.cli.sync
```

The command exits with `0` after a successful sync, `1` when synchronization
fails, and `130` when cancelled with Ctrl+C. Failures are logged to the console.

Call the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```