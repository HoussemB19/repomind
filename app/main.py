from fastapi import FastAPI

app = FastAPI(title="RepoMind", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "phase": "0 - setup"}