

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from agents.base_agent import BaseAgent
from agents import (
    ResearchAgent,
    RetrievalAgent,
    ValidationAgent,
    SecurityAgent,
)


# ===================================================================
# Sample test data
# ===================================================================

SAMPLE_QUERY = "What is deep learning and how does it work?"

SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1",
        "text": """Deep learning is a subset of machine learning that uses
        artificial neural networks with multiple hidden layers. These networks
        learn hierarchical representations of data, enabling them to perform
        complex tasks like image recognition and natural language processing.
        Deep learning requires large amounts of training data and significant
        computational resources, typically GPUs.""",
        "score": 0.92,
    },
    {
        "chunk_id": "c2",
        "text": """Neural networks consist of layers of interconnected nodes
        called neurons. Each connection has a weight that is adjusted during
        training. The process of training involves forward propagation (computing
        outputs) and backpropagation (adjusting weights based on errors).
        Common activation functions include ReLU, sigmoid, and tanh.""",
        "score": 0.85,
    },
    {
        "chunk_id": "c3",
        "text": """Deep learning is a branch of machine learning using neural
        networks with many hidden layers. It learns from large datasets and
        requires GPU computation. Applications include image recognition,
        speech processing, and natural language understanding.""",
        "score": 0.78,
    },
    {
        "chunk_id": "c4",
        "text": """The university cafeteria serves lunch from 11:30 AM to
        2:00 PM on weekdays. The menu rotates weekly and includes vegetarian
        options. Students can purchase meal plans at a discount.""",
        "score": 0.15,
    },
]

SAMPLE_CHUNK_TEXTS = [c["text"] for c in SAMPLE_CHUNKS[:3]]

SAMPLE_ANSWER = (
    "Deep learning is a subset of machine learning that uses neural networks "
    "with multiple hidden layers. It learns hierarchical representations from "
    "large datasets and requires GPUs for training. The training process uses "
    "backpropagation to adjust connection weights. Applications include image "
    "recognition and natural language processing."
)


# ===================================================================
# 1. Research Agent Tests
# ===================================================================

class TestResearchAgentStructure(unittest.TestCase):
    """Test ResearchAgent creation and interface."""

    def test_inherits_base_agent(self):
        agent = ResearchAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_has_name(self):
        agent = ResearchAgent()
        self.assertEqual(agent.name, "ResearchAgent")

    def test_has_description(self):
        agent = ResearchAgent()
        self.assertIn("research", agent.description.lower())

    def test_repr(self):
        agent = ResearchAgent()
        self.assertIn("ResearchAgent", repr(agent))


class TestResearchAgentValidation(unittest.TestCase):
    """Test input validation without LLM calls."""

    def test_no_query(self):
        agent = ResearchAgent(api_key="fake")
        result = agent.run()
        self.assertFalse(result["success"])
        self.assertIn("No query", result["error"])

    def test_no_api_key(self):
        agent = ResearchAgent(api_key="")
        result = agent.run(query="test question")
        self.assertFalse(result["success"])
        self.assertIn("API key", result["error"])


class TestResearchAgentWithLLM(unittest.TestCase):
    """Test with real LLM calls (requires GROQ_API_KEY)."""

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.environ.get("GROQ_API_KEY", "")
        cls.available = bool(cls.api_key)
        if cls.available:
            cls.agent = ResearchAgent(api_key=cls.api_key)

    def test_basic_research(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        self.assertIn("rewritten_query", result)
        self.assertIn("sub_questions", result)
        self.assertIn("keywords", result)

    def test_has_keywords(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        self.assertIsInstance(result["keywords"], list)
        self.assertGreater(len(result["keywords"]), 0)

    def test_has_sub_questions(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        self.assertIsInstance(result["sub_questions"], list)
        self.assertGreater(len(result["sub_questions"]), 0)

    def test_preserves_original_query(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        self.assertEqual(result["original_query"], SAMPLE_QUERY)

    def test_result_structure(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        for key in ["agent", "rewritten_query", "sub_questions",
                     "keywords", "related_concepts", "time_taken", "timestamp"]:
            self.assertIn(key, result, f"Missing key: {key}")


# ===================================================================
# 2. Retrieval Agent Tests
# ===================================================================

class TestRetrievalAgentStructure(unittest.TestCase):
    """Test RetrievalAgent creation and interface."""

    def test_inherits_base_agent(self):
        agent = RetrievalAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_has_name(self):
        agent = RetrievalAgent()
        self.assertEqual(agent.name, "RetrievalAgent")

    def test_has_description(self):
        agent = RetrievalAgent()
        self.assertIn("relevance", agent.description.lower())


class TestRetrievalAgentValidation(unittest.TestCase):
    """Test input validation without LLM calls."""

    def test_no_chunks(self):
        agent = RetrievalAgent(api_key="fake")
        result = agent.run(query="test")
        self.assertFalse(result["success"])
        self.assertIn("No chunks", result["error"])

    def test_no_query(self):
        agent = RetrievalAgent(api_key="fake")
        result = agent.run(chunks=[{"text": "hello"}])
        self.assertFalse(result["success"])
        self.assertIn("No query", result["error"])

    def test_no_api_key(self):
        agent = RetrievalAgent(api_key="")
        result = agent.run(chunks=[{"text": "hello"}], query="test")
        self.assertFalse(result["success"])
        self.assertIn("API key", result["error"])


class TestRetrievalDuplicateDetection(unittest.TestCase):
    """Test duplicate detection (no LLM needed)."""

    def test_detects_near_duplicates(self):
        agent = RetrievalAgent()
        # Create chunks that are deliberately near-duplicates (>80% word overlap)
        near_duplicate_chunks = [
            {"text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers for complex tasks."},
            {"text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers for complex problems."},
            {"text": "The cafeteria serves lunch daily with a variety of options."},
        ]
        dupes = agent._detect_duplicates(near_duplicate_chunks)
        # Should find chunk 1 as a duplicate of chunk 0
        self.assertGreater(len(dupes), 0)

    def test_no_false_duplicates(self):
        agent = RetrievalAgent()
        unique_chunks = [
            {"text": "Deep learning uses neural networks."},
            {"text": "The cafeteria serves lunch daily."},
        ]
        dupes = agent._detect_duplicates(unique_chunks)
        self.assertEqual(len(dupes), 0)


class TestRetrievalAgentWithLLM(unittest.TestCase):
    """Test with real LLM calls (requires GROQ_API_KEY)."""

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.environ.get("GROQ_API_KEY", "")
        cls.available = bool(cls.api_key)
        if cls.available:
            cls.agent = RetrievalAgent(api_key=cls.api_key)

    def test_basic_filtering(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(chunks=SAMPLE_CHUNKS, query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        self.assertIn("filtered_chunks", result)
        self.assertGreater(len(result["filtered_chunks"]), 0)

    def test_removes_irrelevant(self):
        """Chunk about cafeteria should be removed or scored low."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            chunks=SAMPLE_CHUNKS, query=SAMPLE_QUERY, relevance_threshold=0.4
        )
        self.assertTrue(result["success"])
        # The cafeteria chunk (c4) should not be in filtered results
        filtered_ids = [c["chunk_id"] for c in result["filtered_chunks"]]
        self.assertNotIn("c4", filtered_ids)

    def test_output_counts(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(chunks=SAMPLE_CHUNKS, query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        self.assertEqual(result["total_input"], len(SAMPLE_CHUNKS))
        self.assertLessEqual(result["total_output"], result["total_input"])

    def test_result_structure(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(chunks=SAMPLE_CHUNKS, query=SAMPLE_QUERY)
        self.assertTrue(result["success"])
        for key in ["agent", "filtered_chunks", "total_input", "total_output",
                     "removed_low_relevance", "removed_duplicates", "time_taken"]:
            self.assertIn(key, result, f"Missing key: {key}")


# ===================================================================
# 3. Validation Agent Tests
# ===================================================================

class TestValidationAgentStructure(unittest.TestCase):
    """Test ValidationAgent creation and interface."""

    def test_inherits_base_agent(self):
        agent = ValidationAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_has_name(self):
        agent = ValidationAgent()
        self.assertEqual(agent.name, "ValidationAgent")

    def test_has_description(self):
        agent = ValidationAgent()
        self.assertIn("valid", agent.description.lower())


class TestValidationAgentValidation(unittest.TestCase):
    """Test input validation without LLM calls."""

    def test_no_answer(self):
        agent = ValidationAgent(api_key="fake")
        result = agent.run(chunks=["text"], query="q")
        self.assertFalse(result["success"])
        self.assertIn("No answer", result["error"])

    def test_no_chunks(self):
        agent = ValidationAgent(api_key="fake")
        result = agent.run(answer="some answer", query="q")
        self.assertFalse(result["success"])
        self.assertIn("No source chunks", result["error"])

    def test_no_query(self):
        agent = ValidationAgent(api_key="fake")
        result = agent.run(answer="some answer", chunks=["text"])
        self.assertFalse(result["success"])
        self.assertIn("No query", result["error"])

    def test_no_api_key(self):
        agent = ValidationAgent(api_key="")
        result = agent.run(answer="ans", chunks=["t"], query="q")
        self.assertFalse(result["success"])
        self.assertIn("API key", result["error"])


class TestValidationAgentWithLLM(unittest.TestCase):
    """Test with real LLM calls (requires GROQ_API_KEY)."""

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.environ.get("GROQ_API_KEY", "")
        cls.available = bool(cls.api_key)
        if cls.available:
            cls.agent = ValidationAgent(api_key=cls.api_key)

    def test_valid_answer(self):
        """An answer based on the sources should be marked valid."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            answer=SAMPLE_ANSWER,
            chunks=SAMPLE_CHUNK_TEXTS,
            query=SAMPLE_QUERY,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["is_valid"])
        self.assertGreaterEqual(result["confidence"], 0.5)

    def test_hallucinated_answer(self):
        """A made-up answer should be flagged."""
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        fake_answer = (
            "Deep learning was invented by Albert Einstein in 1923. "
            "It requires quantum computers and runs on Windows 95. "
            "The first neural network had exactly 42 layers."
        )
        result = self.agent.run(
            answer=fake_answer,
            chunks=SAMPLE_CHUNK_TEXTS,
            query=SAMPLE_QUERY,
        )
        self.assertTrue(result["success"])
        # Should have flagged claims
        self.assertGreater(len(result["flagged_claims"]), 0)

    def test_confidence_score(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            answer=SAMPLE_ANSWER,
            chunks=SAMPLE_CHUNK_TEXTS,
            query=SAMPLE_QUERY,
        )
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_result_structure(self):
        if not self.available:
            self.skipTest("GROQ_API_KEY not set")
        result = self.agent.run(
            answer=SAMPLE_ANSWER,
            chunks=SAMPLE_CHUNK_TEXTS,
            query=SAMPLE_QUERY,
        )
        self.assertTrue(result["success"])
        for key in ["agent", "is_valid", "confidence", "claims",
                     "flagged_claims", "total_claims", "supported_claims",
                     "summary", "time_taken", "timestamp"]:
            self.assertIn(key, result, f"Missing key: {key}")


# ===================================================================
# 4. Security Agent Tests
# ===================================================================

class TestSecurityAgentStructure(unittest.TestCase):
    """Test SecurityAgent creation and interface."""

    def test_inherits_base_agent(self):
        agent = SecurityAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_has_name(self):
        agent = SecurityAgent()
        self.assertEqual(agent.name, "SecurityAgent")

    def test_has_description(self):
        agent = SecurityAgent()
        self.assertIn("security", agent.description.lower())


class TestSecurityAgentValidation(unittest.TestCase):
    """Test input validation."""

    def test_no_text(self):
        agent = SecurityAgent()
        result = agent.run()
        self.assertFalse(result["success"])
        self.assertIn("No text", result["error"])

    def test_invalid_check_type(self):
        agent = SecurityAgent()
        result = agent.run(text="hello", check_type="invalid")
        self.assertFalse(result["success"])
        self.assertIn("Unknown check_type", result["error"])


class TestSecurityAgentChecks(unittest.TestCase):
    """Test security checks (no LLM needed — uses rule-based detectors)."""

    @classmethod
    def setUpClass(cls):
        """Try to create agent with SecurityLayer."""
        try:
            from security import SecurityLayer
            cls.security_layer = SecurityLayer(enable_layer2=False)
            cls.available = True
        except ImportError:
            cls.security_layer = None
            cls.available = False

    def test_safe_input(self):
        if not self.available:
            self.skipTest("SecurityLayer not importable")
        agent = SecurityAgent(security_layer=self.security_layer)
        result = agent.run(text="What is deep learning?", check_type="input")
        self.assertTrue(result["success"])
        self.assertTrue(result["safe"])

    def test_safe_chunk(self):
        if not self.available:
            self.skipTest("SecurityLayer not importable")
        agent = SecurityAgent(security_layer=self.security_layer)
        result = agent.run(
            text="Neural networks are computing systems inspired by the brain.",
            check_type="chunk",
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["safe"])

    def test_output_check(self):
        if not self.available:
            self.skipTest("SecurityLayer not importable")
        agent = SecurityAgent(security_layer=self.security_layer)
        result = agent.run(
            text="The answer is: deep learning uses neural networks.",
            check_type="output",
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["safe"])

    def test_pii_detection(self):
        """Output with PII should be flagged and redacted."""
        if not self.available:
            self.skipTest("SecurityLayer not importable")
        agent = SecurityAgent(security_layer=self.security_layer)
        result = agent.run(
            text="Contact John at john@example.com or 555-123-4567",
            check_type="output",
        )
        self.assertTrue(result["success"])
        # If PII detected, should have redacted_text
        if not result["safe"]:
            self.assertIn("redacted_text", result)

    def test_result_structure(self):
        if not self.available:
            self.skipTest("SecurityLayer not importable")
        agent = SecurityAgent(security_layer=self.security_layer)
        result = agent.run(text="Hello world", check_type="input")
        self.assertTrue(result["success"])
        for key in ["agent", "safe", "risk_level", "detections",
                     "check_type", "time_taken", "timestamp"]:
            self.assertIn(key, result, f"Missing key: {key}")


# ===================================================================
# Built-in Runner
# ===================================================================

def _run_all():
    """Run all tests without pytest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Research Agent
    suite.addTests(loader.loadTestsFromTestCase(TestResearchAgentStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestResearchAgentValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestResearchAgentWithLLM))

    # Retrieval Agent
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalAgentStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalAgentValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalDuplicateDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalAgentWithLLM))

    # Validation Agent
    suite.addTests(loader.loadTestsFromTestCase(TestValidationAgentStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationAgentValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationAgentWithLLM))

    # Security Agent
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityAgentStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityAgentValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityAgentChecks))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    total = result.testsRun
    failures = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - skipped
    print(f"  Multi-Agent Tests: {passed}/{total} passed, "
          f"{skipped} skipped, {failures} failed")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    _run_all()
