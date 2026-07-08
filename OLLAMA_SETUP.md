# Ollama Embedding Model Setup

## Prerequisites

Ollama runs in Docker. To pull the embedding model:

### Step 1: Start Docker Containers
```bash
docker-compose up -d ollama
```

### Step 2: Pull the Embedding Model
Once the Ollama container is running, pull the `nomic-embed-text` model:

```bash
# Option A: Run directly in container
docker exec vault-ollama-1 ollama pull nomic-embed-text

# Option B: If ollama CLI is available locally (after installing Ollama)
ollama pull nomic-embed-text
```

### Step 3: Verify the Model
```bash
docker exec vault-ollama-1 ollama list
```

Should show `nomic-embed-text` in the list.

## Usage

The `EmbeddingService` will:
- Connect to Ollama at `OLLAMA_BASE_URL` (default: http://localhost:11434)
- Use the `nomic-embed-text` model for embeddings
- Provide methods to embed text and chunks

## Notes

- `nomic-embed-text` is a lightweight embedding model (~274M parameters)
- Runs efficiently on CPU
- Produces 768-dimensional embeddings
- Part of the RAG pipeline after chunking
