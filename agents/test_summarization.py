
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from agents import SummarizationAgent
from agents.base_agent import BaseAgent


# Sample chunks (simulating retrieved RAG chunks)
SAMPLE_CHUNKS = [
    """Annual Leave Policy: All permanent employees are entitled to 20 days
    of annual leave per calendar year. Leave must be approved by the
    department head at least 3 days in advance. Unused leave can be
    carried forward up to a maximum of 5 days to the next year.""",

    """Employee Benefits - Leave: The organization provides 20 working days
    of paid annual leave. In addition, employees get 10 days of sick leave
    and 3 days of casual leave per year. Maternity leave is 90 days for
    female employees. Paternity leave is 15 days.""",

    """Leave Application Process: To apply for leave, employees must submit
    a request through the HR portal. The request goes to the line manager
    for approval. For leave exceeding 5 consecutive days, approval from
    the department head is also required. Emergency leave can be applied
    retroactively within 2 working days.""",
]

SAMPLE_QUERY = "What is the annual leave policy?"


class TestSummarizationAgentStructure(unittest.TestCase):
    """Test agent creation and interface."""

    def test_inherits_base_agent(self):
        """SummarizationAgent should inherit from BaseAgent."""
        agent = SummarizationAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_has_name(self):
        """Agent should have a name."""
        agent = SummarizationAgent()
        self.assertEqual(agent.name, "SummarizationAgent")

    def test_has_description(self):
        """Agent should have a description."""
        agent = SummarizationAgent()
        self.assertIn("summar", agent.description.lower())

    def test_repr(self):
        """Agent should have a readable repr."""
        agent = SummarizationAgent()
        self.assertIn("SummarizationAgent", repr(agent))


class TestSummarizationInputValidation(unittest.TestCase):
    """Test input validation without making LLM calls."""

    def test_no_chunks_returns_error(self):
        """Should return error when no chunks provided."""
        agent = SummarizationAgent()
        result = agent.run(query="test")
        self.assertFalse(result["success"])
        self.assertIn("No chunks", result["error"])

    def test_no_query_in_focused_mode(self):
        """Should return error when query_focused mode has no query."""
        agent = SummarizationAgent(api_key="fake")
        result = agent.run(chunks=["some text"], mode="query_focused")
        self.assertFalse(result["success"])
        self.assertIn("query", result["error"].lower())

    def test_no_api_key(self):
        """Should return error when no API key is set."""
        agent = SummarizationAgent(api_key="")
        result = agent.run(chunks=["some text"], query="test")
        self.assertFalse(result["success"])
        self.assertIn("API key", result["error"])

    def test_unknown_mode(self):
        """Should return error for unknown summarization mode."""
        agent = SummarizationAgent(api_key="fake-key")
        result = agent.run(chunks=["text"], query="q", mode="nonexistent")
        self.assertFalse(result["success"])
        self.assertIn("Unknown mode", result["error"])


class TestSummarizationWithLLM(unittest.TestCase):
    """Test actual summarization with Groq LLM (requires GROQ_API_KEY)."""

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.environ.get("GROQ_API_KEY", "")
        cls.available = bool(cls.api_key)
        if cls.available:
            cls.agent = SummarizationAgent(api_key=cls.api_key)

    def test_query_focused_summary(self):
        """Query-focused summary should mention annual leave details."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query=SAMPLE_QUERY,
            mode="query_focused",
        )
        self.assertTrue(result["success"])
        self.assertIn("summary", result)
        # Should mention 20 days (the key fact)
        self.assertIn("20", result["summary"])

    def test_general_summary(self):
        """General summary should cover multiple topics."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query="",
            mode="general",
        )
        self.assertTrue(result["success"])
        self.assertIn("summary", result)
        self.assertGreater(len(result["summary"]), 50)

    def test_per_chunk_summary(self):
        """Per-chunk summary should have numbered sections."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query="",
            mode="per_chunk",
        )
        self.assertTrue(result["success"])
        # Should have [1], [2], [3] markers
        self.assertIn("[1]", result["summary"])
        self.assertIn("[2]", result["summary"])

    def test_compression_ratio(self):
        """Summary should be shorter than input (compression < 1.0)."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query=SAMPLE_QUERY,
            mode="query_focused",
        )
        self.assertTrue(result["success"])
        self.assertLess(result["compression_ratio"], 1.0)
        self.assertGreater(result["compression_ratio"], 0.0)

    def test_result_structure(self):
        """Result should have all expected keys."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query=SAMPLE_QUERY,
        )
        self.assertTrue(result["success"])
        self.assertIn("agent", result)
        self.assertIn("summary", result)
        self.assertIn("mode", result)
        self.assertIn("input_chunks", result)
        self.assertIn("input_words", result)
        self.assertIn("output_words", result)
        self.assertIn("compression_ratio", result)
        self.assertIn("time_taken", result)
        self.assertIn("timestamp", result)

    def test_input_chunk_count(self):
        """Result should report correct number of input chunks."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query=SAMPLE_QUERY,
        )
        self.assertEqual(result["input_chunks"], 3)

    def test_preserves_key_facts(self):
        """Summary should preserve specific numbers from chunks."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS,
            query="What types of leave are available and how many days?",
            mode="query_focused",
        )
        self.assertTrue(result["success"])
        summary = result["summary"]
        # Should mention at least some of: 20, 10, 3, 90, 15
        numbers_found = sum(1 for n in ["20", "10", "90"] if n in summary)
        self.assertGreaterEqual(numbers_found, 2,
            "Summary should preserve key numbers from the source")


# ===================================================================
# Built-in Runner
# ===================================================================

def _run_all():
    """Run all tests without pytest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSummarizationAgentStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestSummarizationInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSummarizationWithLLM))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    total = result.testsRun
    failures = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - skipped
    print(f"  Summarization Agent Tests: {passed}/{total} passed, "
          f"{skipped} skipped, {failures} failed")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    _run_all()
