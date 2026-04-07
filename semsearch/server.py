"""FastAPI HTTP server for semantic search."""

import json
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from semsearch.config import DATA_DIR, DEFAULT_K, SERVER_HOST, SERVER_PORT
from semsearch.models import SearchResult
from semsearch.search import SearchEngine

engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SearchEngine()
    yield


app = FastAPI(title="semsearch", description="Semantic search for Coq libraries", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


class SearchRequest(BaseModel):
    query: str
    k: int = DEFAULT_K
    library: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total: int


class StatsResponse(BaseModel):
    total_declarations: int
    libraries: dict[str, int]


@app.get("/search")
def search_get(
    q: str = Query(..., description="Search query"),
    k: int = Query(DEFAULT_K, description="Number of results"),
    library: str | None = Query(None, description="Filter by library"),
) -> SearchResponse:
    assert engine is not None
    results = engine.search(q, k=k, library=library)
    return SearchResponse(results=results, query=q, total=len(results))


@app.post("/search")
def search_post(req: SearchRequest) -> SearchResponse:
    assert engine is not None
    results = engine.search(req.query, k=req.k, library=req.library)
    return SearchResponse(results=results, query=req.query, total=len(results))


@app.get("/libraries")
def list_libraries() -> list[str]:
    assert engine is not None
    return sorted({m["library"] for m in engine.metadata})


@app.get("/stats")
def stats() -> StatsResponse:
    assert engine is not None
    counts = Counter(m["library"] for m in engine.metadata)
    return StatsResponse(
        total_declarations=len(engine.metadata),
        libraries=dict(sorted(counts.items())),
    )


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run semsearch HTTP server")
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
