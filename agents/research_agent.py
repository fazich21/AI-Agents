import os
import json
import time
from typing import Any, Optional

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """You are a research planning agent in a RAG system.

USER QUESTION: {query}

TASK:
Analyze this question and produce a research plan to find the best documents.

Return a JSON object with these fields:
{{
    "rewritten_query": "A clearer, more specific version of the question",
    "sub_questions": ["sub-question 1", "sub-question 2", ...],
    "keywords": ["keyword1", "keyword2", "keyword3", ...],
    "related_concepts": ["concept1", "concept2", ...]
}}

Rules:
1. The rewritten query should be self-contained and specific.
2. Break complex questions into 2-4 simpler sub-questions.
3. Extract 3-6 important keywords for text search.
4. List 2-4 related concepts that might appear in relevant documents.
5. For simple questions, sub_questions can have just 1 item.
6. Return ONLY valid JSON, no other text.

JSON:"""


class ResearchAgent(BaseAgent):
    """
    Agent that expands user queries into research plans.

    Takes a short user question and produces:
      - rewritten_query:   clearer version of the question
      - sub_questions:     complex query broken into parts
      - keywords:          key terms for BM25 search
      - related_concepts:  broader terms to expand recall

    Uses Groq API for fast inference.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        super().__init__(
            name="ResearchAgent",
            description="Expands queries into research plans for better retrieval",
        )
        # API key: None → use env var, "" → truly empty
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GROQ_API_KEY", "")

        # Model: provided → use it, else config, else fallback
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
                    "content": "You are a research planning assistant. "
                               "You always return valid JSON. "
                               "You never add text outside the JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM response into structured dict, handling markdown fences."""
        text = raw.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]  # drop first line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: return raw text as rewritten query
            return {
                "rewritten_query": text[:500],
                "sub_questions": [],
                "keywords": [],
                "related_concepts": [],
            }

    # ----- Main run method -----
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Generate a research plan from a user query.

        Args:
            query: the user's original question

        Returns:
            {
                "agent": "ResearchAgent",
                "success": True/False,
                "rewritten_query": "improved version...",
                "sub_questions": ["q1", "q2"],
                "keywords": ["kw1", "kw2", "kw3"],
                "related_concepts": ["concept1", "concept2"],
                "time_taken": 1.23,
            }
        """
        query = kwargs.get("query", "")

        if not query:
            return self._error_result("No query provided")

        if not self.api_key:
            return self._error_result("No API key. Set GROQ_API_KEY environment variable.")

        start_time = time.time()

        try:
            prompt = RESEARCH_PROMPT.format(query=query)
            raw_response = self._call_llm(prompt)
            parsed = self._parse_response(raw_response)

            elapsed = round(time.time() - start_time, 2)

            return self._build_result(
                success=True,
                rewritten_query=parsed.get("rewritten_query", query),
                sub_questions=parsed.get("sub_questions", []),
                keywords=parsed.get("keywords", []),
                related_concepts=parsed.get("related_concepts", []),
                original_query=query,
                time_taken=elapsed,
            )

        except Exception as e:
            return self._error_result(str(e))
