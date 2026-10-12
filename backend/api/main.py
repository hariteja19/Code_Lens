"""
FastAPI REST API for CodeLens.

Endpoints:
- POST /api/index        — start a background indexing job
- GET  /api/index/{id}  — poll indexing job status
- POST /api/query       — run the full query pipeline
- GET  /api/repos       — list indexed repositories
- GET  /api/health      — health check
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="CodeLens API",
    description="RAG-powered code intelligence — ask questions about any codebase",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    """Request to start indexing a GitHub repository."""
    github_url: str = Field(
        ...,
        description="Full GitHub URL (HTTPS)",
        examples=["https://github.com/expressjs/express"]
    )

class QueryRequest(BaseModel):
    """Request to query an indexed codebase."""
    repo_id: str = Field(..., description="Repository identifier (from indexing response)")
    question: str = Field(..., description="Natural language question about the codebase")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to retrieve")

class IndexStatusResponse(BaseModel):
    """Status of an indexing job."""
    job_id: str
    status: str  # queued | cloning | indexing | complete | error
    progress: int = 0
    repo_id: Optional[str] = None
    total_chunks: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-Memory Job Store
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Indexing Background Task
# ---------------------------------------------------------------------------

def run_indexing(job_id: str, github_url: str, repo_id: str):
    """Background task: clone repo → AST chunk → embed → upsert into Qdrant."""
    try:
        from backend.indexer.cloner import RepoCloner
        from backend.indexer.pipeline import IndexingPipeline

        jobs[job_id]["status"] = "cloning"
        jobs[job_id]["progress"] = 10

        cloner = RepoCloner()
        clone_path, actual_repo_id = cloner.clone(
            github_url,
            repo_id=repo_id,
            force_reclone=True
        )

        jobs[job_id]["status"] = "indexing"
        jobs[job_id]["progress"] = 20

        qdrant_url = os.getenv("QDRANT_URL", "local")
        pipeline = IndexingPipeline(qdrant_url=qdrant_url)
        pipeline.create_collection()

        def progress_callback(status: str, progress: int):
            scaled = 20 + int(progress * 0.75)
            jobs[job_id]["status"] = "indexing"
            jobs[job_id]["progress"] = min(scaled, 95)
            jobs[job_id]["detail"] = status

        total_chunks = pipeline.index_repo(
            clone_path,
            repo_id,
            progress_callback=progress_callback
        )

        jobs[job_id] = {
            "status": "complete",
            "progress": 100,
            "repo_id": repo_id,
            "total_chunks": total_chunks,
        }

    except Exception as e:
        jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "repo_id": repo_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "codelens-api"}


@app.post("/api/index", response_model=IndexStatusResponse)
async def start_indexing(req: IndexRequest, background_tasks: BackgroundTasks):
    """
    Start indexing a GitHub repository.
    Returns a job_id to poll via GET /api/index/{job_id}.
    """
    job_id = str(uuid.uuid4())

    from backend.indexer.cloner import RepoCloner
    cloner = RepoCloner()
    repo_id = cloner.extract_repo_id(req.github_url)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "repo_id": repo_id
    }

    background_tasks.add_task(run_indexing, job_id, req.github_url, repo_id)

    return IndexStatusResponse(
        job_id=job_id,
        repo_id=repo_id,
        status="queued",
        progress=0
    )


@app.get("/api/index/{job_id}", response_model=IndexStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status of an indexing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return IndexStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        repo_id=job.get("repo_id"),
        total_chunks=job.get("total_chunks"),
        error=job.get("error"),
    )


@app.post("/api/query")
async def query_codebase(req: QueryRequest):
    """
    Query an indexed codebase with a natural language question.
    Pipeline: hybrid search → cross-encoder rerank → LLM answer generation.
    """
    from backend.retriever.searcher import HybridSearcher
    from backend.retriever.reranker import Reranker
    from backend.retriever.generator import Generator

    qdrant_url = os.getenv("QDRANT_URL", "local")

    try:
        searcher = HybridSearcher(qdrant_url, "codelens_chunks")
        candidates = searcher.search(req.question, req.repo_id, top_k=20)

        if not candidates:
            return {
                "answer": "No relevant code chunks found. The repository may not be indexed yet.",
                "citations": [],
                "model_used": "none",
                "chunks_retrieved": 0,
                "query": req.question,
            }

        reranker = Reranker()
        top_chunks = reranker.rerank(req.question, candidates, top_k=req.top_k)

        generator = Generator()
        return generator.generate(req.question, top_chunks)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Repository '{req.repo_id}' not found. Please index it first via POST /api/index"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/repos")
async def list_repos():
    """List all indexed repositories."""
    from backend.indexer.pipeline import METADATA_DIR
    from pathlib import Path
    import json as _json

    repos = []
    metadata_path = Path(METADATA_DIR)
    if metadata_path.exists():
        for file_path in metadata_path.rglob("*_vocab.json"):
            rel_path = file_path.relative_to(metadata_path)
            parts = list(rel_path.parts)
            parts[-1] = parts[-1][:-11]  # strip "_vocab.json"
            repo_id = "/".join(parts)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    vocab = _json.load(f)
                vocab_size = len(vocab)
            except Exception:
                vocab_size = 0

            repos.append({
                "repo_id": repo_id,
                "status": "indexed",
                "vocab_size": vocab_size,
            })

    return {"repos": repos, "total": len(repos)}


@app.get("/api/stats")
async def get_stats():
    """System statistics — indexed repos, collection info."""
    from backend.indexer.pipeline import METADATA_DIR
    from pathlib import Path

    repo_count = 0
    if os.path.exists(METADATA_DIR):
        repo_count = sum(1 for _ in Path(METADATA_DIR).rglob("*_vocab.json"))

    collection_info = None
    try:
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv("QDRANT_URL", "local")
        if qdrant_url == "local":
            client = QdrantClient(path="qdrant_data")
        else:
            client = QdrantClient(url=qdrant_url, timeout=3)
        info = client.get_collection("codelens_chunks")
        collection_info = {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
    except Exception:
        collection_info = {"status": "unavailable"}

    return {
        "indexed_repos": repo_count,
        "active_jobs": len([j for j in jobs.values() if j.get("status") not in ("complete", "error")]),
        "collection": collection_info,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
