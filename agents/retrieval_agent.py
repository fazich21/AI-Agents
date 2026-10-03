import os
import json
import time
from typing import Any, Optional

from agents.base_agent import BaseAgent


class RetrievalAgent(BaseAgent):
    """
    Agent that scores and filters retrieved chunks by relevance.

    Takes retrieved chunks + user query and:
      - Scores each chunk's relevance (0.0 to 1.0)
      - Filters out chunks below the threshold
      - Detects near-duplicate chunks
      - Returns a clean, ranked list

    Uses Groq API for relevance scoring.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        super().__init__(
            name="RetrievalAgent",
            description="Scores and filters chunks by relevance to the query",
        )
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GROQ_API_KEY", "")

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
                raise ImportError("groq package not installed. Run: pip install groq")
        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call Groq LLM and return the response text."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a relevance scoring assistant. "
                               "You always return valid JSON arrays. "
                               "You never add text outside the JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _format_chunks(self, chunks: list[dict]) -> str:
        """Format chunks for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")[:500]  # Truncate long chunks for scoring
            parts.append(f"[Chunk {i}]\n{text.strip()}\n")
        return "\n".join(parts)

    def _parse_scores(self, raw: str, num_chunks: int) -> list[dict]:
        """Parse LLM response into score list."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            scores = json.loads(text)
            if isinstance(scores, list):
                return scores
        except json.JSONDecodeError:
            pass

        # Fallback: give all chunks a default score
        return [
            {"chunk_index": i, "relevance_score": 0.5, "reason": "scoring failed"}
            for i in range(num_chunks)
        ]

    def _detect_duplicates(self, chunks: list[dict]) -> set:
        """
        Simple duplicate detection based on text overlap.
        Returns set of chunk indices that are near-duplicates.
        """
        duplicates = set()
        texts = [c.get("text", "").lower().split() for c in chunks]

        for i in range(len(texts)):
            if i in duplicates:
                continue
            for j in range(i + 1, len(texts)):
                if j in duplicates:
                    continue
                # Calculate word overlap ratio
                set_i = set(texts[i])
                set_j = set(texts[j])
                if not set_i or not set_j:
                    continue
                overlap = len(set_i & set_j) / min(len(set_i), len(set_j))
                if overlap > 0.8:  # 80% word overlap = near-duplicate
                    duplicates.add(j)  # Keep the earlier chunk

        return duplicates

    # ----- Main run method -----
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Score and filter retrieved chunks.

        Args:
            chunks:               list of chunk dicts with "chunk_id", "text", "score"
            query:                the user's original question
            relevance_threshold:  minimum score to keep (default 0.3)
            remove_duplicates:    drop near-duplicate chunks (default True)

        Returns:
            {
                "agent": "RetrievalAgent",
                "success": True/False,
                "filtered_chunks": [{"chunk_id": ..., "relevance_score": ..., "reason": ...}],
                "total_input": 5,
                "total_output": 3,
                "removed_low_relevance": 1,
                "removed_duplicates": 1,
                "time_taken": 1.23,
            }
        """
        chunks = kwargs.get("chunks", [])
        query = kwargs.get("query", "")
        threshold = kwargs.get("relevance_threshold", 0.3)
        remove_dupes = kwargs.get("remove_duplicates", True)

        if not chunks:
            return self._error_result("No chunks provided to evaluate")

        if not query:
            return self._error_result("No query provided")

        if not self.api_key:
            return self._error_result("No API key. Set GROQ_API_KEY environment variable.")

        start_time = time.time()

        try:
            # Step 1: Detect duplicates (no LLM needed)
            duplicate_indices = set()
            if remove_dupes:
                duplicate_indices = self._detect_duplicates(chunks)

            # Step 2: Score relevance with LLM
            chunks_text = self._format_chunks(chunks)
            prompt = RELEVANCE_PROMPT.format(query=query, chunks_text=chunks_text)
            raw_response = self._call_llm(prompt)
            scores = self._parse_scores(raw_response, len(chunks))

            # Step 3: Build scored chunk list
            scored_chunks = []
            removed_low = 0
            removed_dupes = 0

            for score_entry in scores:
                idx = score_entry.get("chunk_index", -1)
                if idx < 0 or idx >= len(chunks):
                    continue

                relevance = score_entry.get("relevance_score", 0.0)
                reason = score_entry.get("reason", "")

                # Skip duplicates
                if idx in duplicate_indices:
                    removed_dupes += 1
                    continue

                # Skip low relevance
                if relevance < threshold:
                    removed_low += 1
                    continue

                chunk = chunks[idx]
                scored_chunks.append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                    "text": chunk.get("text", ""),
                    "original_score": chunk.get("score", 0.0),
                    "relevance_score": round(relevance, 3),
                    "reason": reason,
                })

            # Sort by relevance score (highest first)
            scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

            elapsed = round(time.time() - start_time, 2)

            return self._build_result(
                success=True,
                filtered_chunks=scored_chunks,
                total_input=len(chunks),
                total_output=len(scored_chunks),
                removed_low_relevance=removed_low,
                removed_duplicates=removed_dupes,
                relevance_threshold=threshold,
                time_taken=elapsed,
            )

        except Exception as e:
            return self._error_result(str(e))
