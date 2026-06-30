# Vault Backend

## Setup with uv

Create or refresh the virtual environment and install dependencies:

```bash
uv sync
```

Run the API in development mode:

```bash
uv run uvicorn app.main:app --reload
```

Call the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```