"""
Summarization Agent — Condenses retrieved chunks for better answers.
=====================================================================

Purpose in the RAG pipeline:
  After retrieval + reranking, you have 5-10 chunks of raw text.
  These chunks often contain:
    - Redundant information (same fact in multiple chunks)
    - Irrelevant sentences (the chunk has extra context)
    - Long passages when only 1-2 sentences matter

  The Summarization Agent reads all chunks, understands the user's
  question, and produces a FOCUSED summary that:
    1. Keeps only information relevant to the question
    2. Removes redundancy across chunks
    3. Preserves key facts, numbers, dates, names
    4. Stays faithful to the source (no hallucination)

  This summary then goes to the Answer Generator instead of raw chunks,
  resulting in:
    - Better answers (less noise for the LLM to filter)
    - Lower token usage (shorter context = cheaper API calls)
    - Faster responses (less text to process)

How it works:
  1. Receives: list of chunk texts + user query
  2. Builds a summarization prompt with the query as focus
  3. Calls Groq LLM (same one used for answer generation)
  4. Returns: condensed summary + metadata

Usage:
    from agents import SummarizationAgent

    agent = SummarizationAgent()

    result = agent.run(
        chunks=["chunk1 text...", "chunk2 text..."],
        query="What is the annual leave policy?",
        max_summary_length=500,    # optional, default 500 words
        mode="query_focused",       # or "general"
    )

    if result["success"]:
        summary = result["summary"]
        # Pass this summary to AnswerGenerator instead of raw chunks
"""

import os
import time
from typing import Any, Optional

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Summarization prompts
# ---------------------------------------------------------------------------

QUERY_FOCUSED_PROMPT = """You are a precise summarization agent in a RAG pipeline.

USER QUESTION: {query}

RETRIEVED DOCUMENTS:
{chunks_text}

TASK:
Summarize the above documents with FOCUS on answering the user's question.

Rules:
1. Include ONLY information relevant to the question.
2. Preserve exact numbers, dates, names, and specific details.
3. Remove redundant information (if multiple chunks say the same thing, state it once).
4. Do NOT add information that is not in the documents.
5. Do NOT say "according to the documents" — just state the facts.
6. Keep the summary under {max_length} words.
7. If the documents don't contain relevant information, say "No relevant information found in the retrieved documents."

FOCUSED SUMMARY:"""

GENERAL_SUMMARY_PROMPT = """You are a precise summarization agent.

DOCUMENTS:
{chunks_text}

TASK:
Provide a comprehensive summary of all the documents above.

Rules:
1. Cover all key points from every document.
2. Preserve exact numbers, dates, names, and specific details.
3. Remove redundant information across documents.
4. Organize the summary logically by topic.
5. Do NOT add information that is not in the documents.
6. Keep the summary under {max_length} words.

SUMMARY:"""

CHUNK_SUMMARY_PROMPT = """Summarize this single text passage in 2-3 sentences,
preserving key facts, numbers, and names:

TEXT:
{chunk_text}

SUMMARY:"""


class SummarizationAgent(BaseAgent):
    """
    Agent that summarizes retrieved chunks before answer generation.

    Supports three modes:
      - "query_focused": Summarizes with the user's question as focus (default)
      - "general":       General summary of all chunks
      - "per_chunk":     Summarizes each chunk individually (useful for long chunks)

    Uses Groq API (same as your AnswerGenerator) for fast inference.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        super().__init__(
            name="SummarizationAgent",
            description="Condenses retrieved chunks into focused summaries",
        )
        # Use provided key; only fall back to env if None (not empty string)
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GROQ_API_KEY", "")

        # Use provided model, or read from project config, or fallback
        if model is not None:
            self.model = model
        else:
            try:
                from config import GROQ_MODEL
                self.model = GROQ_MODEL
            except ImportError:
                self.model = "openai/gpt-oss-120b"

        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy-load the Groq client."""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: pip install groq"
                )
        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call Groq LLM and return the response text."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise summarization assistant. "
                               "You only state facts from the provided text. "
                               "You never add information that isn't there."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _format_chunks(self, chunks: list[str]) -> str:
        """Format chunks into numbered text for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Document {i}]\n{chunk.strip()}\n")
        return "\n".join(parts)

    # ----- Main run method -----
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Summarize retrieved chunks.

        Args:
            chunks: list of chunk text strings
            query:  the user's original question (required for query_focused mode)
            mode:   "query_focused" (default), "general", or "per_chunk"
            max_summary_length: max words for the summary (default 500)

        Returns:
            {
                "agent": "SummarizationAgent",
                "success": True/False,
                "summary": "the condensed text...",
                "mode": "query_focused",
                "input_chunks": 5,
                "input_words": 2340,
                "output_words": 180,
                "compression_ratio": 0.077,
                "time_taken": 1.23,
            }
        """
        chunks = kwargs.get("chunks", [])
        query = kwargs.get("query", "")
        mode = kwargs.get("mode", "query_focused")
        max_length = kwargs.get("max_summary_length", 500)

        # Validation
        if not chunks:
            return self._error_result("No chunks provided to summarize")

        if mode == "query_focused" and not query:
            return self._error_result(
                "query_focused mode requires a 'query' parameter"
            )

        if not self.api_key:
            return self._error_result(
                "No API key. Set GROQ_API_KEY environment variable."
            )

        start_time = time.time()

        try:
            if mode == "query_focused":
                summary = self._summarize_query_focused(chunks, query, max_length)
            elif mode == "general":
                summary = self._summarize_general(chunks, max_length)
            elif mode == "per_chunk":
                summary = self._summarize_per_chunk(chunks)
            else:
                return self._error_result(f"Unknown mode: {mode}")

            elapsed = round(time.time() - start_time, 2)

            # Calculate compression stats
            input_text = " ".join(chunks)
            input_words = len(input_text.split())
            output_words = len(summary.split())
            compression = round(output_words / max(input_words, 1), 3)

            return self._build_result(
                success=True,
                summary=summary,
                mode=mode,
                input_chunks=len(chunks),
                input_words=input_words,
                output_words=output_words,
                compression_ratio=compression,
                time_taken=elapsed,
            )

        except Exception as e:
            return self._error_result(str(e))

    # ----- Summarization strategies -----

    def _summarize_query_focused(
        self, chunks: list[str], query: str, max_length: int
    ) -> str:
        """Summarize chunks with the user's question as the focus."""
        chunks_text = self._format_chunks(chunks)
        prompt = QUERY_FOCUSED_PROMPT.format(
            query=query,
            chunks_text=chunks_text,
            max_length=max_length,
        )
        return self._call_llm(prompt)

    def _summarize_general(self, chunks: list[str], max_length: int) -> str:
        """General summary of all chunks without a specific question focus."""
        chunks_text = self._format_chunks(chunks)
        prompt = GENERAL_SUMMARY_PROMPT.format(
            chunks_text=chunks_text,
            max_length=max_length,
        )
        return self._call_llm(prompt)

    def _summarize_per_chunk(self, chunks: list[str]) -> str:
        """Summarize each chunk individually, then combine."""
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk.strip())
            chunk_summary = self._call_llm(prompt)
            summaries.append(f"[{i}] {chunk_summary}")
        return "\n\n".join(summaries)
