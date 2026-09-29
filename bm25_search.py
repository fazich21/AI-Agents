"""
BM25 Search: Traditional keyword-based search.

BM25 finds chunks that contain the EXACT words in your query.
It's good at finding specific terms, names, and numbers that
vector search might miss.

Example:
  Query: "backpropagation gradient descent"
  BM25 finds chunks containing those exact words.
  Vector search might also return chunks about "training neural networks"
  which is related but doesn't contain those exact words.
"""

import json
import re
from rank_bm25 import BM25Okapi
from config import CHUNKS_PATH, TOP_K_BM25


class BM25Search:
    """BM25 keyword search over document chunks."""

    def __init__(self, chunks=None):
        """
        Build the BM25 index from chunks.
        Chunks are loaded from Task 2's chunks.json if not provided.
        """
        if chunks is None:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]

        # Tokenize each chunk's text into words
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]

        # Build the BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  BM25 index built over {len(chunks)} chunks")

    def _tokenize(self, text):
        """
        Simple tokenization: lowercase, split on non-alphanumeric,
        remove very short words.
        """
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Remove very short tokens (like "a", "is", "to")
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    def search(self, query, top_k=TOP_K_BM25):
        """
        Search for the top_k most relevant chunks by keyword matching.

        Returns: list of (chunk_id, score) tuples, sorted by score descending.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return chunks with non-zero score
                results.append((self.chunk_ids[idx], float(scores[idx])))

        return results

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
    print("BM25 SEARCH TEST")
    print("=" * 60)

    bm25 = BM25Search()

    test_queries = [
        "machine learning algorithms",
        "backpropagation neural network",
        "German automotive industry",
    ]

    for q in test_queries:
        results = bm25.search(q, top_k=3)
        print(f"\n  Query: {q}")
        for chunk_id, score in results:
            text = bm25.get_chunk_text(chunk_id)
            preview = text[:80] + "..." if text else "N/A"
            print(f"    [{score:.2f}] {chunk_id}: {preview}")
