"""
Query Rewriter: Takes a user's messy question and rewrites it
into a clean, effective search query using Groq (fast LLM).
"""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "Run: $env:GROQ_API_KEY='your-key-here'"
        )
    return Groq(api_key=GROQ_API_KEY)


def clean_response(text):
    """Remove <think> tags and any reasoning from model output."""
    text = text.strip()
    if "<think>" in text:
        parts = text.split("</think>")
        text = parts[-1].strip() if len(parts) > 1 else ""
    # Remove any remaining tags
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip().strip('"').strip("'")


def rewrite_query(user_query):
    """Rewrite a user question into a better search query."""
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a search query optimizer. Return ONLY the rewritten query as a single line. No thinking, no reasoning, no explanations, no quotes, no tags. Make it specific. Always write in English even if the user asks in another language."
            },
            {
                "role": "user",
                "content": f"Rewrite this into a clear search query: {user_query}"
            }
        ],
        max_tokens=300,
        temperature=0.3,
    )

    result = response.choices[0].message.content
    return clean_response(result)


if __name__ == "__main__":
    test_queries = [
        "how does AI help doctors?",
        "what is ML?",
        "tell me about german cars",
        "was ist das Bildungssystem in Deutschland?",
    ]

    print("=" * 60)
    print("QUERY REWRITER TEST (Groq)")
    print("=" * 60)

    for q in test_queries:
        rewritten = rewrite_query(q)
        print(f"\n  Original:  {q}")
        print(f"  Rewritten: {rewritten}")