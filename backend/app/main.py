from fastapi import FastAPI

app = FastAPI(title="Vault Backend")



@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Vault Backend is running"}
    