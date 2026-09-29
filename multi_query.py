"""
Multi-Query Retrieval: Takes one question and generates multiple
different versions to search with, using Groq (fast LLM).
"""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, NUM_MULTI_QUERIES


def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=GROQ_API_KEY)


def clean_response(text):
    """Remove <think> tags and any reasoning from model output."""
    text = text.strip()
    if "<think>" in text:
        parts = text.split("</think>")
        text = parts[-1].strip() if len(parts) > 1 else ""
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def generate_multi_queries(user_query, num_queries=NUM_MULTI_QUERIES):
    """Generate multiple search queries from one user question."""
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"Generate exactly {num_queries} different search queries for the given question. One per line. No numbering, no bullets, no explanations, no thinking, no reasoning. Always write in English."
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        max_tokens=500,
        temperature=0.5,
    )

    result = clean_response(response.choices[0].message.content)
    lines = result.split("\n")

    queries = []
    for line in lines:
        cleaned = line.strip().strip("-").strip("•").strip()
        if cleaned and len(cleaned) > 5:
            parts = cleaned.split(".", 1)
            if parts[0].strip().isdigit() and len(parts) > 1:
                cleaned = parts[1].strip()
            parts = cleaned.split(")", 1)
            if parts[0].strip().isdigit() and len(parts) > 1:
                cleaned = parts[1].strip()
            queries.append(cleaned.strip('"').strip("'"))

    if len(queries) > num_queries:
        queries = queries[:num_queries]
    elif len(queries) < num_queries:
        queries.append(user_query)

    return queries[:num_queries]


if __name__ == "__main__":
    test_queries = [
        "How is AI used in medicine?",
        "What is deep learning?",
        "Tell me about German universities",
    ]

    print("=" * 60)
    print("MULTI-QUERY GENERATOR TEST (Groq)")
    print("=" * 60)

    for q in test_queries:
        print(f"\n  Original: {q}")
        multi = generate_multi_queries(q)
        for i, mq in enumerate(multi):
            print(f"    Query {i+1}: {mq}")