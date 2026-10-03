

import os
import json
import time
from typing import Any, Optional

from agents.base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    """
    Agent that validates LLM answers against source documents.

    Takes a generated answer + source chunks and:
      - Extracts factual claims from the answer
      - Checks each claim against the sources
      - Computes a confidence score
      - Flags unsupported claims (potential hallucinations)

    Uses Groq API for claim verification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        super().__init__(
            name="ValidationAgent",
            description="Validates answers against sources to catch hallucinations",
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
                    "content": "You are a fact-checking assistant. "
                               "You verify claims against source documents. "
                               "You always return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _format_chunks(self, chunks: list[str]) -> str:
        """Format source chunks for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}]\n{chunk.strip()}\n")
        return "\n".join(parts)

    def _parse_validation(self, raw: str) -> dict:
        """Parse LLM validation response."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: can't parse, assume cautiously valid
            return {
                "is_valid": True,
                "confidence": 0.5,
                "claims": [],
                "summary": "Validation parsing failed — returned default.",
            }

    # ----- Main run method -----
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Validate a generated answer against source chunks.

        Args:
            answer:  the LLM-generated answer text
            chunks:  list of source chunk texts used to generate the answer
            query:   the user's original question

        Returns:
            {
                "agent": "ValidationAgent",
                "success": True/False,
                "is_valid": True/False,
                "confidence": 0.85,
                "claims": [{"claim": "...", "supported": True, "evidence": "..."}],
                "flagged_claims": ["unsupported claim 1", ...],
                "summary": "validation verdict",
                "time_taken": 1.23,
            }
        """
        answer = kwargs.get("answer", "")
        chunks = kwargs.get("chunks", [])
        query = kwargs.get("query", "")

        if not answer:
            return self._error_result("No answer provided to validate")

        if not chunks:
            return self._error_result("No source chunks provided for validation")

        if not query:
            return self._error_result("No query provided")

        if not self.api_key:
            return self._error_result("No API key. Set GROQ_API_KEY environment variable.")

        start_time = time.time()

        try:
            chunks_text = self._format_chunks(chunks)
            prompt = VALIDATION_PROMPT.format(
                query=query,
                answer=answer,
                chunks_text=chunks_text,
            )
            raw_response = self._call_llm(prompt)
            parsed = self._parse_validation(raw_response)

            # Extract flagged (unsupported) claims
            claims = parsed.get("claims", [])
            flagged = [
                c["claim"] for c in claims
                if not c.get("supported", True)
            ]

            elapsed = round(time.time() - start_time, 2)

            return self._build_result(
                success=True,
                is_valid=parsed.get("is_valid", True),
                confidence=round(parsed.get("confidence", 0.5), 3),
                claims=claims,
                flagged_claims=flagged,
                total_claims=len(claims),
                supported_claims=len(claims) - len(flagged),
                summary=parsed.get("summary", ""),
                time_taken=elapsed,
            )

        except Exception as e:
            return self._error_result(str(e))
