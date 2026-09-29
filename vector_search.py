"""
Vector Search: Semantic search using FAISS + BGE-Large from Task 2.
Optimized for faster query encoding on CPU.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    CHUNKS_PATH,
    BGE_CORPUS_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_QUERY_PREFIX,
    TOP_K_VECTOR,
)


class VectorSearch:
    """FAISS vector search using BGE-Large embeddings."""

    def __init__(self, chunks=None):
        if chunks is None:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]

        # Load cached BGE-Large embeddings from Task 2
        print(f"  Loading cached embeddings from {BGE_CORPUS_PATH}")
        self.corpus_embeddings = np.load(BGE_CORPUS_PATH).astype(np.float32)

        # Build FAISS index
        dimension = self.corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        normed = self.corpus_embeddings.copy()
        faiss.normalize_L2(normed)
        self.index.add(normed)
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dimension}")

        # Load the embedding model for encoding queries
        print(f"  Loading embedding model for queries...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.model.max_seq_length = 128  # Queries are short, no need for 512

        # Warm up the model with a dummy query (first call is always slow)
        print(f"  Warming up model...")
        _ = self.model.encode(["warm up query"], show_progress_bar=False)
        print(f"  Vector search ready!")

    def _embed_query(self, query):
        """Embed a single query using BGE-Large with query prefix."""
        text = EMBEDDING_QUERY_PREFIX + query
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            batch_size=1,
            normalize_embeddings=True,
        )[0].astype(np.float32)
        return embedding

    def _embed_queries_batch(self, queries):
        """Embed multiple queries at once (faster than one by one)."""
        texts = [EMBEDDING_QUERY_PREFIX + q for q in queries]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=len(texts),
            normalize_embeddings=True,
        ).astype(np.float32)
        return embeddings

    def search(self, query, top_k=TOP_K_VECTOR):
        """Search for the top_k most semantically similar chunks."""
        query_vec = self._embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def search_multiple(self, queries, top_k=TOP_K_VECTOR):
        """Search with multiple queries at once (batch embedding = faster)."""
        all_results = {}

        # Embed all queries in one batch instead of one by one
        embeddings = self._embed_queries_batch(queries)

        for i, query_vec in enumerate(embeddings):
            query_vec = query_vec.reshape(1, -1)
            scores, indices = self.index.search(query_vec, top_k)

            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[idx]
                    if chunk_id not in all_results or score > all_results[chunk_id]:
                        all_results[chunk_id] = float(score)

        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_chunk_text(self, chunk_id):
        """Get the text of a chunk by its ID."""
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk["text"]
        return None


# ──────────────────────────────────────────────
# Test it
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("VECTOR SEARCH TEST")
    print("=" * 60)

    vsearch = VectorSearch()

    test_queries = [
        "machine learning algorithms",
        "backpropagation neural network",
        "German automotive industry"
    ]

    for q in test_queries:
        results = vsearch.search(q, top_k=3)
        print(f"\n  Query: {q}")
        for chunk_id, score in results:
            text = vsearch.get_chunk_text(chunk_id)
            preview = text[:80] + "..." if text else "N/A"
            print(f"    [{score:.4f}] {chunk_id}: {preview}")
