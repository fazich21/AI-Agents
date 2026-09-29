"""
Hybrid Search: Combines BM25 (keyword) + FAISS (vector) results
using Reciprocal Rank Fusion (RRF).

Why hybrid? Each search type has strengths:
  - BM25 catches EXACT word matches (names, numbers, specific terms)
  - FAISS catches MEANING matches (synonyms, related concepts)
  - Together they cover more relevant chunks than either alone

How RRF works:
  A chunk ranked #1 by BM25 gets score 1/(60+1) = 0.0164
  A chunk ranked #3 by FAISS gets score 1/(60+3) = 0.0159
  If a chunk appears in BOTH lists, its scores ADD UP.
  So chunks found by both methods rank highest.
"""

import json
from config import (
    CHUNKS_PATH,
    TOP_K_BM25,
    TOP_K_VECTOR,
    TOP_K_HYBRID,
    RRF_K,
)
from bm25_search import BM25Search
from vector_search import VectorSearch
from query_rewriter import rewrite_query
from multi_query import generate_multi_queries


class HybridSearch:
    """
    Combines BM25 keyword search + FAISS vector search.
    Includes query rewriting and multi-query retrieval.
    """

    def __init__(self):
        """Initialize both search engines."""
        print("=" * 60)
        print("Initializing Hybrid Search Engine")
        print("=" * 60)

        # Load chunks once, share between both engines
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  Loaded {len(self.chunks)} chunks")

        # Build both search indexes
        self.bm25 = BM25Search(self.chunks)
        self.vector = VectorSearch(self.chunks)

        print("\n  Hybrid search engine ready!")

    def reciprocal_rank_fusion(self, result_lists, k=RRF_K):
        """
        Combine multiple ranked lists into one using RRF.

        For each chunk, its RRF score = sum of 1/(k + rank) across all lists.
        Chunks that appear in multiple lists get higher combined scores.

        Args:
            result_lists: list of lists, each containing (chunk_id, score) tuples
            k: RRF constant (default 60, standard value from the original paper)

        Returns:
            list of (chunk_id, rrf_score) sorted by score descending
        """
        rrf_scores = {}

        for results in result_lists:
            for rank, (chunk_id, _score) in enumerate(results):
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                # RRF formula: 1 / (k + rank + 1)
                # rank+1 because rank starts at 0
                rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(self, user_query, use_rewrite=True, use_multi_query=True, top_k=TOP_K_HYBRID):
        """
        Full hybrid search pipeline:
          1. Rewrite the query (optional)
          2. Generate multi-queries (optional)
          3. Run BM25 on all queries
          4. Run FAISS on all queries
          5. Fuse results with RRF
          6. Return top-k

        Args:
            user_query: the user's original question
            use_rewrite: whether to rewrite the query with Gemini
            use_multi_query: whether to generate multiple query variants
            top_k: number of results to return

        Returns:
            dict with:
                - results: list of (chunk_id, score) tuples
                - original_query: what the user typed
                - rewritten_query: what Gemini improved it to
                - multi_queries: the variant queries generated
                - bm25_results: raw BM25 results
                - vector_results: raw FAISS results
        """
        search_info = {
            "original_query": user_query,
            "rewritten_query": None,
            "multi_queries": [],
            "bm25_results": [],
            "vector_results": [],
        }

        # Step 1: Query rewriting
        if use_rewrite:
            rewritten = rewrite_query(user_query)
            search_info["rewritten_query"] = rewritten
            primary_query = rewritten
        else:
            primary_query = user_query

        # Step 2: Multi-query generation
        if use_multi_query:
            multi_queries = generate_multi_queries(primary_query)
            search_info["multi_queries"] = multi_queries
            all_queries = [primary_query] + multi_queries
        else:
            all_queries = [primary_query]

        # Step 3: BM25 search with all queries
        bm25_result_lists = []
        for q in all_queries:
            results = self.bm25.search(q, top_k=TOP_K_BM25)
            bm25_result_lists.append(results)

        # Fuse BM25 results from all queries
        bm25_fused = self.reciprocal_rank_fusion(bm25_result_lists)[:TOP_K_BM25]
        search_info["bm25_results"] = bm25_fused

        # Step 4: Vector search with all queries
        vector_results = self.vector.search_multiple(all_queries, top_k=TOP_K_VECTOR)
        search_info["vector_results"] = vector_results

        # Step 5: Hybrid fusion (combine BM25 + Vector)
        hybrid_fused = self.reciprocal_rank_fusion([bm25_fused, vector_results])
        final_results = hybrid_fused[:top_k]

        search_info["results"] = final_results
        return search_info

    def search_simple(self, user_query, top_k=TOP_K_HYBRID):
        """
        Simple search without rewriting or multi-query.
        Useful for comparison or when Gemini API is unavailable.
        """
        return self.search(user_query, use_rewrite=False, use_multi_query=False, top_k=top_k)

    def get_chunk_text(self, chunk_id):
        """Get the text of a chunk by its ID."""
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk["text"]
        return None

    def get_chunk_details(self, chunk_id):
        """Get full chunk details including file name and language."""
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None


# ──────────────────────────────────────────────
# Test it
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n")
    hybrid = HybridSearch()

    test_queries = [
        "How is AI used in medicine?",
        "What is backpropagation?",
        "German car industry",
    ]

    for user_query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"USER QUERY: {user_query}")
        print(f"{'=' * 60}")

        result = hybrid.search(user_query)

        print(f"\n  Rewritten: {result['rewritten_query']}")
        print(f"  Multi-queries:")
        for i, mq in enumerate(result['multi_queries']):
            print(f"    {i+1}. {mq}")

        print(f"\n  BM25 results ({len(result['bm25_results'])} chunks):")
        for chunk_id, score in result['bm25_results'][:3]:
            text = hybrid.get_chunk_text(chunk_id)
            preview = text[:70] + "..." if text else "N/A"
            print(f"    [{score:.4f}] {preview}")

        print(f"\n  Vector results ({len(result['vector_results'])} chunks):")
        for chunk_id, score in result['vector_results'][:3]:
            text = hybrid.get_chunk_text(chunk_id)
            preview = text[:70] + "..." if text else "N/A"
            print(f"    [{score:.4f}] {preview}")

        print(f"\n  HYBRID results ({len(result['results'])} chunks):")
        for chunk_id, score in result['results'][:5]:
            text = hybrid.get_chunk_text(chunk_id)
            preview = text[:70] + "..." if text else "N/A"
            print(f"    [{score:.4f}] {preview}")
