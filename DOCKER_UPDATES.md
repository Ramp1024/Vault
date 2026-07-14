# Docker Configuration Updates

## Changes Made

### 1. Backend Dockerfile (`backend/Dockerfile`)

**Before:**
```dockerfile
COPY app ./app
COPY core ./core
COPY models ./models
COPY services ./services
```

**After:**
```dockerfile
# Copy entire app directory with all subdirectories
COPY app ./app
COPY config ./config
COPY .env .env 2>/dev/null || true
```

**Why:** The new project structure has all code under `app/` directory with subdirectories:
- `app/core/`
- `app/models/`
- `app/services/`
- `app/processors/` (newly added)
- `app/connectors/`
- `app/api/`

### 2. Docker Compose (`docker-compose.yml`)

**Replaced Chroma with Qdrant:**

**Before:**
```yaml
chroma:
  image: chromadb/chroma:latest
  ports:
    - "8001:8000"
  volumes:
    - chroma_data:/chroma/chroma
  environment:
    IS_PERSISTENT: "TRUE"
```

**After:**
```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
  volumes:
    - qdrant_data:/qdrant/storage
  environment:
    QDRANT_API_KEY: ${QDRANT_API_KEY:-vault-dev-key}
```

**Updated API Service:**

**Before:**
```yaml
environment:
  OLLAMA_BASE_URL: <docker-managed ollama endpoint>
  CHROMA_HOST: chroma
  CHROMA_PORT: 8000
depends_on:
  - ollama
  - chroma
```

**After:**
```yaml
environment:
  OLLAMA_BASE_URL: http://localhost:11434
  QDRANT_HOST: qdrant
  QDRANT_PORT: 6333
depends_on:
  - qdrant
```

Ollama is no longer managed by Docker Compose. Run Ollama natively in WSL before starting the stack.

## New Directory Structure

The Docker files now properly handle:

```
backend/
├── app/                    # All application code
│   ├── core/              # Config and settings
│   ├── models/            # Document and Chunk models
│   ├── services/          # Ollama, Qdrant, Embedding services
│   ├── processors/        # Chunker (new)
│   ├── connectors/        # Notion connector
│   ├── api/               # FastAPI endpoints
│   └── main.py
├── config/                # Configuration files (YAML, etc.)
├── Dockerfile             # Updated
├── pyproject.toml         # Dependencies
└── uv.lock               # Locked dependencies
```

## Usage

### Start Services
```bash
docker-compose up -d qdrant api ui
```

### Build and Run
```bash
docker-compose up --build
```

### Pull Required Models
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Services

- **API** (8000): FastAPI backend
- **Ollama** (11434): External local WSL service
- **Qdrant** (6333): Vector database
- **UI** (5173): Frontend (Vite/React)
