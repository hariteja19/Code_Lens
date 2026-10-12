"""
Hybrid Search with Reciprocal Rank Fusion (RRF).

Combines dense semantic search and BM25 keyword search:
- Dense: finds semantically related code even without exact keywords
- BM25: exact term match for function names, identifiers

RRF formula: score(doc) = Σ 1/(k + rank_i)
where k=60 prevents top-ranked results from dominating.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, Filter, FieldCondition, MatchValue
)
import json
import os
import re
from collections import Counter

from backend.indexer.embedder import Embedder
from backend.indexer.pipeline import METADATA_DIR, get_vocab_path


class HybridSearcher:
    """
    Performs hybrid search combining dense (semantic) and sparse (BM25)
    retrieval with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        qdrant_url: str = "local",
        collection_name: str = "codelens_chunks",
        rrf_k: int = 60
    ):
        if qdrant_url == "local":
            self.client = QdrantClient(path="qdrant_data")
        else:
            self.client = QdrantClient(url=qdrant_url)

        self.collection_name = collection_name
        self.embedder = Embedder()
        self.rrf_k = rrf_k
        self._vocab_cache: dict[str, dict] = {}

    def load_vocab(self, repo_id: str) -> dict:
        """Load (and cache) the BM25 vocabulary for a repository."""
        if repo_id not in self._vocab_cache:
            vocab_path = get_vocab_path(repo_id)
            with open(vocab_path, "r", encoding="utf-8") as f:
                self._vocab_cache[repo_id] = json.load(f)
        return self._vocab_cache[repo_id]

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenizer — matches the one used during indexing."""
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())

    def query_to_sparse(self, query: str, vocab: dict) -> SparseVector:
        """Convert query text to a sparse BM25 vector using the stored vocabulary."""
        tokens = self.tokenize(query)
        term_freq = Counter(tokens)

        indices = []
        values = []
        for term, freq in term_freq.items():
            if term in vocab:
                indices.append(vocab[term])
                values.append(float(freq))

        return SparseVector(indices=indices, values=values)

    def reciprocal_rank_fusion(
        self,
        dense_results: list,
        sparse_results: list,
        k: int = 60,
        top_n: int = 20
    ) -> list:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.
        Documents appearing in both lists score higher than those in just one.
        """
        scores: dict = {}

        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.id
            if doc_id not in scores:
                scores[doc_id] = {
                    "rrf": 0.0,
                    "payload": result.payload,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "dense_score": result.score,
                }
            scores[doc_id]["rrf"] += 1.0 / (k + rank)

        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.id
            if doc_id not in scores:
                scores[doc_id] = {
                    "rrf": 0.0,
                    "payload": result.payload,
                    "dense_rank": None,
                    "dense_score": 0.0,
                }
            scores[doc_id]["rrf"] += 1.0 / (k + rank)
            scores[doc_id]["sparse_rank"] = rank

        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf"],
            reverse=True
        )

        return sorted_docs[:top_n]

    def search(
        self,
        query: str,
        repo_id: str,
        top_k: int = 20,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> list:
        """
        Full hybrid search: dense + sparse + RRF fusion.
        Results are filtered to only include chunks from the specified repo.
        """
        vocab = self.load_vocab(repo_id)

        repo_filter = Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=repo_id)
                )
            ]
        )

        query_embedding = self.embedder.embed_query(query)
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            using="dense",
            query_filter=repo_filter,
            limit=top_k,
            with_payload=True,
        ).points

        sparse_query = self.query_to_sparse(query, vocab)
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_query,
            using="sparse",
            query_filter=repo_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return self.reciprocal_rank_fusion(dense_results, sparse_results, top_n=top_k)
