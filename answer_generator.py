"""
Answer & Citation Generator: Takes the top reranked chunks and
generates a proper answer with citations using Groq (fast LLM).
"""

import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, CHUNKS_PATH


def clean_response(text):
    """Remove <think> tags and any reasoning from model output."""
    text = text.strip()
    if "<think>" in text:
        parts = text.split("</think>")
        text = parts[-1].strip() if len(parts) > 1 else ""
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


class AnswerGenerator:
    """Generates answers with citations using Groq."""

    def __init__(self):
        """Initialize Groq."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=GROQ_API_KEY)
        print("  Answer generator ready (Groq)!")

        # Load chunks for source info
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

    def generate_answer(self, query, reranked_results):
        """Generate an answer with citations from the top chunks."""
        if not reranked_results:
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "sources": [],
                "chunks_used": [],
            }

        top_results = reranked_results[:5]

        context_parts = []
        sources = []
        chunks_used = []

        for i, (chunk_id, score) in enumerate(top_results):
            chunk = self.chunk_map.get(chunk_id, {})
            text = chunk.get("text", "")
            file_name = chunk.get("file_name", "unknown")
            language = chunk.get("language", "unknown")

            short_text = text[:300] if len(text) > 300 else text
            context_parts.append(f"[{i+1}] ({file_name}):\n{short_text}")

            sources.append({
                "source_number": i + 1,
                "chunk_id": chunk_id,
                "file_name": file_name,
                "language": language,
                "relevance_score": round(score, 4),
                "text_preview": text[:150] + "..." if len(text) > 150 else text,
            })

            chunks_used.append({
                "chunk_id": chunk_id,
                "text": text,
                "file_name": file_name,
            })

        context = "\n\n".join(context_parts)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. Answer questions using ONLY the provided sources. Add [1], [2], [3], [4], [5] citations after each claim. Write 3-5 sentences. Never make up information. Do NOT include any thinking, reasoning, or tags in your response. Just give the answer directly."
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
                }
            ],
            max_tokens=800,
            temperature=0.3,
        )

        answer = clean_response(response.choices[0].message.content)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": chunks_used,
        }

    def format_output(self, query, answer_result, search_info=None):
        """Format the complete output for display."""
        output = []
        output.append(f"Question: {query}")

        if search_info:
            if search_info.get("rewritten_query"):
                output.append(f"Rewritten: {search_info['rewritten_query']}")

        output.append(f"\nAnswer:\n{answer_result['answer']}")

        if answer_result["sources"]:
            output.append(f"\nSources:")
            for src in answer_result["sources"]:
                output.append(
                    f"  [{src['source_number']}] {src['file_name']} "
                    f"(chunk: {src['chunk_id']}, score: {src['relevance_score']})"
                )

        return "\n".join(output)


if __name__ == "__main__":
    from hybrid_search import HybridSearch
    from reranker import Reranker

    print("=" * 60)
    print("ANSWER + CITATION GENERATOR TEST (Groq)")
    print("=" * 60)

    hybrid = HybridSearch()
    reranker = Reranker()
    generator = AnswerGenerator()

    test_queries = [
        "How is AI used in medicine?",
        "What is backpropagation?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        search_result = hybrid.search(query)
        reranked = reranker.rerank(query, search_result["results"])
        answer_result = generator.generate_answer(query, reranked)
        formatted = generator.format_output(query, answer_result, search_result)
        print(formatted)