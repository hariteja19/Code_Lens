"""
Indexing Pipeline — orchestrates the complete indexing flow.

Offline phase: GitHub Repo → Clone → Walk Files → AST Chunk → Embed → Store in Qdrant

Creates a Qdrant collection with dual vector support:
- Dense vectors: from embedding model, capture semantic meaning
- Sparse vectors: BM25 term frequencies, capture keyword overlap
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector
)
import numpy as np
import json
import re
import os
from pathlib import Path
from typing import Callable, Optional
from collections import Counter

from backend.indexer.chunker import chunk_file, chunk_file_with_fallback, should_skip_path, Chunk
from backend.indexer.embedder import Embedder
from backend.indexer.cloner import RepoCloner


METADATA_DIR = os.path.join(os.path.expanduser("~"), ".codelens", "metadata")


def get_vocab_path(repo_id: str) -> str:
    """Get the path to a repo's stored vocabulary file."""
    parts = repo_id.split('/')
    filename = f"{parts[-1]}_vocab.json"
    return os.path.join(METADATA_DIR, *parts[:-1], filename)


class IndexingPipeline:
    """
    Orchestrates the full indexing pipeline for a code repository.

    Steps: create collection → chunk files → build vocabulary → embed → upsert → save vocab
    """

    def __init__(self, qdrant_url: str = "local"):
        if qdrant_url == "local":
            self.client = QdrantClient(path="qdrant_data")
        else:
            self.client = QdrantClient(url=qdrant_url)

        self.embedder = Embedder()
        self.collection_name = "codelens_chunks"
        self.vector_size = self.embedder.dimension
        os.makedirs(METADATA_DIR, exist_ok=True)

    def create_collection(self):
        """
        Create a Qdrant collection with both dense and sparse vector support.
        Drops and recreates the collection if it already exists.
        """
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            }
        )
        print(f"Created collection '{self.collection_name}' (dense: {self.vector_size}d, sparse: BM25)")

    def tokenize(self, text: str) -> list[str]:
        """Simple BM25 tokenizer — splits on non-alphanumeric characters and lowercases."""
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())

    def compute_bm25_vector(self, tokens: list[str], vocab: dict[str, int]) -> SparseVector:
        """
        Convert tokenized text to a sparse BM25 vector.
        Only non-zero entries are stored (indices + values for matching terms).
        """
        term_freq = Counter(tokens)

        indices = []
        values = []
        for term, freq in term_freq.items():
            if term in vocab:
                indices.append(vocab[term])
                values.append(float(freq))

        return SparseVector(indices=indices, values=values)

    def index_repo(
        self,
        repo_path: str,
        repo_id: str,
        progress_callback: Callable | None = None
    ) -> int:
        """
        Full indexing pipeline for a repository.

        Args:
            repo_path: Local path to the cloned repository
            repo_id: Unique identifier for the repository
            progress_callback: Optional function(status, progress%) for progress updates

        Returns:
            Total number of chunks indexed
        """
        def report(status: str, progress: int = 0):
            if progress_callback:
                progress_callback(status, progress)
            print(f"[{repo_id}] {status} ({progress}%)")

        # Step 1: Collect source files
        report("Collecting source files...", 5)
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.html'}
        source_files = []
        repo = Path(repo_path)

        for file_path in repo.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix not in source_extensions:
                continue
            rel_path = str(file_path.relative_to(repo))
            if should_skip_path(rel_path):
                continue
            source_files.append(file_path)

        report(f"Found {len(source_files)} source files", 10)

        # Step 2: Chunk all files
        report("Parsing ASTs and chunking files...", 15)
        all_chunks: list[Chunk] = []

        for file_path in source_files:
            try:
                source_code = file_path.read_text(encoding='utf-8', errors='ignore')
                ext = file_path.suffix
                rel_path = str(file_path.relative_to(repo))
                file_chunks = chunk_file_with_fallback(rel_path, source_code, ext)
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"  Skipping {file_path}: {e}")

        if not all_chunks:
            report("No chunks found — repo may be empty or unsupported", 100)
            return 0

        report(f"Created {len(all_chunks)} chunks", 30)

        # Step 3: Build BM25 vocabulary
        report("Building BM25 vocabulary...", 35)
        tokenized_corpus = [
            self.tokenize(c.context_header + " " + c.raw_text)
            for c in all_chunks
        ]

        vocab: dict[str, int] = {}
        for tokens in tokenized_corpus:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        report(f"Vocabulary: {len(vocab)} unique terms", 40)

        # Steps 4+5+6: Embed, compute BM25, upsert in batches
        batch_size = 16
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_idx:batch_idx + batch_size]
            batch_tokens = tokenized_corpus[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1
            chunks_done = batch_idx + len(batch)

            progress = 40 + int(50 * (batch_idx / len(all_chunks)))
            report(f"Embedding & indexing batch {current_batch}/{total_batches} ({chunks_done}/{len(all_chunks)} chunks)...", progress)

            texts = [
                f"search_document: {c.context_header}\n\n{c.raw_text}"
                for c in batch
            ]
            dense_vectors = self.embedder.embed_batch(texts, show_progress=False)

            points = []
            for j, (chunk, tokens, dense_vec) in enumerate(zip(batch, batch_tokens, dense_vectors)):
                sparse_vec = self.compute_bm25_vector(tokens, vocab)

                point = PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": sparse_vec,
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "node_type": chunk.node_type,
                        "parent_class": chunk.parent_class,
                        "language": chunk.language,
                        "raw_text": chunk.raw_text,
                        "context_header": chunk.context_header,
                        "repo_id": repo_id,
                        "char_count": chunk.char_count,
                    }
                )
                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)

        # Step 7: Save vocabulary for query-time use
        vocab_path = get_vocab_path(repo_id)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)

        report(f"Indexing complete! {len(all_chunks)} chunks indexed", 100)
        return len(all_chunks)

    def get_vocab_path(self, repo_id: str) -> str:
        """Get the path to a repo's stored vocabulary file."""
        return get_vocab_path(repo_id)

    def collection_exists(self) -> bool:
        """Check if the Qdrant collection already exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False
