"""
Content Reranker: Takes the top 10 chunks from hybrid search
and rescores them using a cross-encoder model.

Why reranking?
  BM25 and FAISS score chunks INDEPENDENTLY — they look at
  the chunk alone without reading the question carefully.

  A cross-encoder reads the QUESTION and CHUNK TOGETHER as one
  text, like a human reading both side by side. This gives a
  much more accurate relevance score.

  Think of it like:
    - BM25/FAISS = scanning a library shelf by title
    - Reranker = actually reading each book's first page
      to see if it answers your question

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~80MB, runs locally)
"""

import json
from sentence_transformers import CrossEncoder
from config import CHUNKS_PATH, TOP_K_RERANK


class Reranker:
    """Cross-encoder reranker for search results."""

    def __init__(self):
        """Load the cross-encoder model (downloads ~80MB first time)."""
        print("  Loading cross-encoder reranker model...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        print("  Reranker ready!")

        # Load chunks for text lookup
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

    def rerank(self, query, search_results, top_k=TOP_K_RERANK):
        """
        Rerank search results using cross-encoder.

        Args:
            query: the user's question
            search_results: list of (chunk_id, score) from hybrid search
            top_k: how many to keep after reranking

        Returns:
            list of (chunk_id, rerank_score) sorted by new score
        """
        if not search_results:
            return []

        # Build pairs of (question, chunk_text) for the cross-encoder
        pairs = []
        chunk_ids = []
        for chunk_id, _score in search_results:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                pairs.append((query, chunk["text"]))
                chunk_ids.append(chunk_id)

        if not pairs:
            return []

        # Cross-encoder scores each (question, chunk) pair
        scores = self.model.predict(pairs)

        # Combine chunk_ids with new scores and sort
        reranked = list(zip(chunk_ids, [float(s) for s in scores]))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]

    def get_chunk_text(self, chunk_id):
        """Get chunk text by ID."""
        chunk = self.chunk_map.get(chunk_id)
        return chunk["text"] if chunk else None

    def get_chunk_details(self, chunk_id):
        """Get full chunk details."""
        return self.chunk_map.get(chunk_id)


# ──────────────────────────────────────────────
# Test it
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from hybrid_search import HybridSearch

    print("=" * 60)
    print("RERANKER TEST")
    print("=" * 60)

    hybrid = HybridSearch()
    reranker = Reranker()

    query = "How is AI used in medicine?"

    # Get hybrid search results
    search_result = hybrid.search(query)
    hybrid_results = search_result["results"]

    print(f"\n  Query: {query}")
    print(f"\n  Before reranking (hybrid scores):")
    for chunk_id, score in hybrid_results[:5]:
        text = hybrid.get_chunk_text(chunk_id)
        preview = text[:70] + "..." if text else "N/A"
        print(f"    [{score:.4f}] {preview}")

    # Rerank
    reranked = reranker.rerank(query, hybrid_results)

    print(f"\n  After reranking (cross-encoder scores):")
    for chunk_id, score in reranked:
        text = reranker.get_chunk_text(chunk_id)
        preview = text[:70] + "..." if text else "N/A"
        print(f"    [{score:.4f}] {preview}")
