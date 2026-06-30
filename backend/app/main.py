from fastapi import FastAPI

from app.api import router as api_router

app = FastAPI(title="Vault Backend")

app.include_router(api_router)

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Vault Backend is running"}
    