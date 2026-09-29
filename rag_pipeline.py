"""
Complete RAG Pipeline: Ties everything together.

Full flow:
  User question
    → Query rewriting (Gemini)
    → Multi-query generation (Gemini)
    → BM25 search (keywords)
    → FAISS search (vectors)
    → Hybrid fusion (RRF)
    → Reranking (cross-encoder)
    → Answer generation (Gemini)
    → Citation generation (Gemini)
    → Final answer with sources

Run directly:
    python rag_pipeline.py
"""

import time
import json
from hybrid_search import HybridSearch
from reranker import Reranker
from answer_generator import AnswerGenerator


class RAGPipeline:
    """Complete RAG pipeline from question to cited answer."""

    def __init__(self):
        """Initialize all components."""
        print("\n" + "=" * 60)
        print("  INITIALIZING RAG PIPELINE")
        print("=" * 60)

        self.hybrid = HybridSearch()
        self.reranker = Reranker()
        self.generator = AnswerGenerator()

        print("\n" + "=" * 60)
        print("  RAG PIPELINE READY!")
        print("=" * 60)

    def ask(self, question, use_rewrite=True, use_multi_query=True, verbose=True):
        """
        Ask a question and get an answer with citations.

        Args:
            question: the user's question
            use_rewrite: whether to rewrite the query
            use_multi_query: whether to generate multi-queries
            verbose: whether to print step-by-step progress

        Returns:
            dict with complete results from every step
        """
        result = {
            "question": question,
            "timings": {},
        }

        # Step 1: Hybrid search (includes query rewriting + multi-query)
        if verbose:
            print(f"\n  Step 1: Searching...")
        start = time.time()
        search_result = self.hybrid.search(
            question,
            use_rewrite=use_rewrite,
            use_multi_query=use_multi_query,
        )
        result["timings"]["search"] = round(time.time() - start, 2)
        result["search"] = search_result

        if verbose:
            print(f"    Rewritten: {search_result.get('rewritten_query', 'N/A')}")
            print(f"    Found {len(search_result['results'])} hybrid results")

        # Step 2: Reranking
        if verbose:
            print(f"\n  Step 2: Reranking...")
        start = time.time()
        reranked = self.reranker.rerank(question, search_result["results"])
        result["timings"]["rerank"] = round(time.time() - start, 2)
        result["reranked"] = reranked

        if verbose:
            print(f"    Kept top {len(reranked)} chunks after reranking")

        # Step 3: Answer generation with citations
        if verbose:
            print(f"\n  Step 3: Generating answer...")
        start = time.time()
        answer_result = self.generator.generate_answer(question, reranked)
        result["timings"]["answer"] = round(time.time() - start, 2)
        result["answer"] = answer_result["answer"]
        result["sources"] = answer_result["sources"]

        # Total time
        result["timings"]["total"] = round(
            sum(result["timings"].values()), 2
        )

        if verbose:
            print(f"\n  Done! Total time: {result['timings']['total']}s")

        return result

    def format_answer(self, result):
        """Format the result for nice display."""
        output = []
        output.append(f"\nQuestion: {result['question']}")

        search = result.get("search", {})
        if search.get("rewritten_query"):
            output.append(f"Rewritten: {search['rewritten_query']}")

        output.append(f"\n{'─' * 50}")
        output.append(f"ANSWER:\n")
        output.append(result["answer"])
        output.append(f"\n{'─' * 50}")
        output.append(f"SOURCES:")

        for src in result.get("sources", []):
            output.append(
                f"  [{src['source_number']}] {src['file_name']} "
                f"(score: {src['relevance_score']})"
            )

        timings = result.get("timings", {})
        output.append(f"\n{'─' * 50}")
        output.append(f"TIMINGS:")
        output.append(f"  Search: {timings.get('search', 0)}s")
        output.append(f"  Rerank: {timings.get('rerank', 0)}s")
        output.append(f"  Answer: {timings.get('answer', 0)}s")
        output.append(f"  Total:  {timings.get('total', 0)}s")

        return "\n".join(output)


# ──────────────────────────────────────────────
# Interactive mode
# ──────────────────────────────────────────────

if __name__ == "__main__":
    rag = RAGPipeline()

    print("\n" + "=" * 60)
    print("  RAG SYSTEM READY — Type your questions!")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        print()
        question = input("  Your question: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        result = rag.ask(question)
        formatted = rag.format_answer(result)
        print(formatted)
