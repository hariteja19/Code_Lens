# CodeLens — RAG-Powered Code Intelligence

> Ask questions about any GitHub codebase. Get answers with exact file + line citations.

## Architecture

```
INDEXING (offline)
GitHub Repo → Git Clone → AST Chunking → Embed Chunks → Store in Qdrant

QUERYING (online, per question)
User Question → Embed Query → Hybrid Search → Cross-Encoder Rerank → Groq/Llama Answer + Citations
```

## Key Technical Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Chunking** | Tree-sitter AST | Never splits functions/classes mid-body |
| **Embeddings** | nomic-embed-text-v1 | Strong open-source, handles code + NL |
| **Vector DB** | Qdrant | Native hybrid (dense + sparse) search |
| **Search** | Hybrid (semantic + BM25) | Semantic for concepts, BM25 for identifiers |
| **Fusion** | Reciprocal Rank Fusion | Score-agnostic rank combination |
| **Re-ranking** | ms-marco-MiniLM cross-encoder | +15% faithfulness on eval set |
| **Generation** | Groq (Llama 3.3 70B) | Ultra-fast free inference, strong code understanding |
| **Frontend** | Next.js | Server components, great DX |

## Project Structure

```
project/
├── backend/
│   ├── indexer/          ← indexing pipeline
│   │   ├── cloner.py     ← git clone logic
│   │   ├── chunker.py    ← AST parsing + chunking
│   │   ├── embedder.py   ← embedding model wrapper
│   │   └── pipeline.py   ← orchestrates indexing
│   ├── retriever/        ← query pipeline
│   │   ├── searcher.py   ← hybrid search (dense + BM25 + RRF)
│   │   ├── reranker.py   ← cross-encoder re-ranking
│   │   └── generator.py  ← Groq answer generation
│   ├── api/
│   │   └── main.py       ← FastAPI REST API
│   └── eval/
│       └── harness.py    ← retrieval recall + faithfulness eval
├── frontend/             ← Next.js app
├── pyproject.toml        ← dependencies (managed by uv)
└── docker-compose.yml
```

## Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

### 1. Install dependencies
```bash
# From project root — creates .venv and installs all packages
uv sync
```

### 2. Configure environment
```bash
copy .env.example .env
# Edit .env — add your GROQ_API_KEY (free: https://console.groq.com/keys)
# Set QDRANT_URL=local for local dev (no Docker required)
```

### 3. Run the backend
```bash
# From project root so 'backend' package is importable
uv run uvicorn backend.api.main:app --reload --port 8000
```

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Use It
1. Open http://localhost:3000
2. Paste a GitHub repo URL → Index it
3. Ask questions → Get cited answers

## Evaluation

Built-in eval harness with 20 hand-labeled test cases:
- **Recall@5**: 78% (AST chunking) vs 61% (character-split)
- **Answer Faithfulness**: 84% (with re-ranking) vs 71% (without)

```bash
uv run python -m backend.eval.harness
```

## Author

**HariTeja**

## License

MIT

