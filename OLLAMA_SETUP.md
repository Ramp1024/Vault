# Ollama Embedding Model Setup

## Prerequisites

Install Ollama in WSL and ensure the service is running.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 1: Start Ollama
```bash
ollama serve
```

### Step 2: Pull Required Models
Once Ollama is running, pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

### Step 3: Verify the Model
```bash
ollama list
```

Should show both `nomic-embed-text` and `llama3.1:8b` in the list.

### Step 4: Configure `OLLAMA_BASE_URL`

Use the default local endpoint unless you have a non-standard setup:

- `http://localhost:11434`

## Usage

The `EmbeddingService` will:
- Connect to Ollama at `OLLAMA_BASE_URL`
- Use the `nomic-embed-text` model for embeddings
- Provide methods to embed text and chunks

## Notes

- `nomic-embed-text` is a lightweight embedding model (~274M parameters)
- Runs efficiently on CPU
- Produces 768-dimensional embeddings
- Part of the RAG pipeline after chunking
