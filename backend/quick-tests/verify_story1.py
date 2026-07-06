import httpx

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
OLLAMA_URL = "http://localhost:11434"


def check(name: str, url: str):
    try:
        response = httpx.get(url, timeout=5)

        if response.status_code == 200:
            print(f"✅ {name}")
        else:
            print(f"❌ {name} (HTTP {response.status_code})")

    except Exception as e:
        print(f"❌ {name} ({e})")


print("=" * 50)
print("VAULT - STORY 1 VERIFICATION")
print("=" * 50)

check("Frontend Running", FRONTEND_URL)
check("Backend Running", f"{BACKEND_URL}/health")
check("Ollama Running", f"{OLLAMA_URL}/api/tags")

print("=" * 50)
print("Verify ChromaDB manually or add a health endpoint if exposed.")
print("=" * 50)
