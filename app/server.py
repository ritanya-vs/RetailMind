"""RetailMind web UI.  Run from the repository root:

    python app/server.py            # http://127.0.0.1:8000
    RETAILMIND_MODE=demo python app/server.py   # force the sample-catalogue demo
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.requests import Request  # noqa: E402

from demo_catalog import EXAMPLES  # noqa: E402
from engine import load_engine, load_results  # noqa: E402

app = FastAPI(title="RetailMind", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.middleware("http")
async def secure_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "mode": engine.mode}

engine = load_engine()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    run: str = "A6"
    top: int = Field(default=10, ge=1, le=30)


@app.get("/api/status")
def status() -> dict:
    return {
        "mode": engine.mode,
        "llm": engine.llm_ok,
        "examples": EXAMPLES,
        "runs": {n: {"description": r.description, "components": r.components}
                 for n, r in engine.runs.items()},
    }  # fmt: skip


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    if req.run not in engine.runs:
        raise HTTPException(400, f"unknown run {req.run!r}")
    try:
        return engine.search(req.query.strip(), req.run, req.top)
    except Exception as error:  # noqa: BLE001 - surface the reason to the UI
        raise HTTPException(500, str(error)) from error


@app.get("/api/results")
def results() -> dict:
    return load_results()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
