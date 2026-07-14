from fastapi import FastAPI

from app.api import router as api_router
from app.services.ollama import assert_ollama_reachable

app = FastAPI(title="Vault Backend")


@app.on_event("startup")
async def startup_checks() -> None:
    assert_ollama_reachable()


app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Vault Backend is running"}
    