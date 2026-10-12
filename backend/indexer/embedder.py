"""
Embedding Model Wrapper.

Converts code chunks and queries into dense vectors using nomic-embed-text-v1.5.

Key design decisions:
- Contextual embedding: embed context_header + raw_text together for location awareness
- Asymmetric: queries use "search_query:" prefix, documents use "search_document:"
- Normalized: L2-normalized so cosine similarity == dot product
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Wraps nomic-ai/nomic-embed-text-v1.5 for both document and query embedding.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.dimension = self.model.get_embedding_dimension()

    def embed_chunk(self, chunk) -> np.ndarray:
        """
        Embed a single code chunk with its context header.
        Embeds context_header + raw_text so the vector captures WHERE the code
        lives in the codebase, not just WHAT it contains.
        """
        text_to_embed = f"search_document: {chunk.context_header}\n\n{chunk.raw_text}"
        return self.model.encode(text_to_embed, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query using the 'search_query:' prefix.
        nomic-embed uses asymmetric embedding: queries are trained to be close
        to relevant documents, not to other queries.
        """
        return self.model.encode(f"search_query: {query}", normalize_embeddings=True)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed multiple texts in batches for efficiency."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )

    def embed_texts_for_indexing(self, chunks: list) -> np.ndarray:
        """Embed a list of Chunk objects for indexing (with search_document: prefix)."""
        texts = [
            f"search_document: {c.context_header}\n\n{c.raw_text}"
            for c in chunks
        ]
        return self.embed_batch(texts)
