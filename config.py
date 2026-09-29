"""
Configuration for Task 3: Advanced RAG System.
Reuses Task 2's embeddings and chunks.
"""

import os

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK3_DIR = os.path.join(PROJECT_ROOT, "task3")
TASK2_DIR = os.path.join(PROJECT_ROOT, "task2")

# Task 2 outputs we reuse
CHUNKS_PATH = os.path.join(TASK2_DIR, "data", "chunks.json")
BGE_CORPUS_PATH = os.path.join(TASK2_DIR, "embeddings_cache", "bge-large_corpus.npy")
BGE_QUERIES_PATH = os.path.join(TASK2_DIR, "embeddings_cache", "bge-large_queries.npy")

# Task 3 output directories
DATA_DIR = os.path.join(TASK3_DIR, "data")
RESULTS_DIR = os.path.join(TASK3_DIR, "results")

# ──────────────────────────────────────────────
# GEMINI API
# ──────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.6-flash"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"

# ──────────────────────────────────────────────
# EMBEDDING MODEL (best from Task 2)
# ──────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ──────────────────────────────────────────────
# SEARCH SETTINGS
# ──────────────────────────────────────────────

TOP_K_BM25 = 10          # How many results from BM25
TOP_K_VECTOR = 10        # How many results from FAISS
TOP_K_HYBRID = 10        # Final hybrid results
TOP_K_RERANK = 5         # After reranking
NUM_MULTI_QUERIES = 2    # How many query variants to generate
RRF_K = 60               # Reciprocal Rank Fusion constant

# ──────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────

for d in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
