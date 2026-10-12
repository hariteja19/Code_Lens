"""
Cross-Encoder Re-ranking.

Rescores the top candidates from hybrid search using a cross-encoder model.
Unlike bi-encoders (which compare vector pairs), cross-encoders pass the query
and document together through a transformer — much more accurate but slower.

We only run it on the top 20 RRF candidates, keeping latency to ~100ms on CPU.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder re-ranker using ms-marco-MiniLM-L-6-v2.
    Trained on MS MARCO passage ranking — ideal for query-document relevance scoring.
    """

    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        max_length: int = 512
    ):
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.model_name = model_name

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """
        Re-score candidates with the cross-encoder and return the top_k.

        Args:
            query: The user's question
            candidates: List of (doc_id, {rrf, payload, ...}) tuples from RRF
            top_k: Number of top results to return

        Returns:
            List of dicts with payload, rerank_score, rrf_score, and ranks
        """
        if not candidates:
            return []

        pairs = []
        for candidate in candidates:
            doc_id, doc_data = candidate
            payload = doc_data["payload"]
            document_text = f"{payload['context_header']}\n\n{payload['raw_text']}"
            pairs.append((query, document_text))

        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        results = []
        for rank, (candidate, score) in enumerate(scored[:top_k], start=1):
            doc_id, doc_data = candidate
            results.append({
                "payload": doc_data["payload"],
                "rerank_score": float(score),
                "rrf_score": doc_data["rrf"],
                "dense_rank": doc_data.get("dense_rank"),
                "sparse_rank": doc_data.get("sparse_rank"),
                "final_rank": rank,
            })

        return results
