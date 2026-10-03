"""
FastAPI Backend

Wraps the entire RAG pipeline into a web API.
Start with: uvicorn api:app --reload --port 8000

Endpoints:
  GET  /health          → check if server is running
  POST /ask             → full RAG: answer + citations
  POST /search          → hybrid search only (no Gemini needed)
  GET  /graph/{entity}  → knowledge graph lookup
  GET  /graph/stats     → graph statistics
  POST /api/security/check → test security layer directly
"""

import os
import sys
import json
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ──────────────────────────────────────────────
# Security Layer (Task 4 — Layer 1 + Layer 2)
# ──────────────────────────────────────────────
from security import SecurityLayer

# Set ENABLE_LAYER2=1 to activate ML-based security detectors
# Requires: pip install transformers torch spacy
#           python -m spacy download en_core_web_sm
ENABLE_LAYER2 = os.environ.get("ENABLE_LAYER2", "0") == "1"

security = SecurityLayer(enable_layer2=ENABLE_LAYER2)

# ──────────────────────────────────────────────
# Multi-Agent System (Task 5)
# ──────────────────────────────────────────────
from agents import (
    SummarizationAgent,
    ResearchAgent,
    RetrievalAgent,
    ValidationAgent,
    SecurityAgent,
)

# Initialize all agents once at module level
summarizer = SummarizationAgent()
researcher = ResearchAgent()
retrieval_agent = RetrievalAgent()
validator = ValidationAgent()
security_agent = SecurityAgent(security_layer=security)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Advanced RAG API",
    description="Document retrieval and question answering with hybrid search, reranking, and citations",
    version="2.0.0",
)

# Allow React UI to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React runs on localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Request/Response models
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    use_rewrite: Optional[bool] = True
    use_multi_query: Optional[bool] = False  # Off by default to save API quota
    top_k: Optional[int] = 5
    # Task 5: Multi-Agent toggles (all on by default)
    use_research_agent: Optional[bool] = True
    use_retrieval_agent: Optional[bool] = True
    use_summarizer: Optional[bool] = True
    use_validator: Optional[bool] = True

class SearchRequest(BaseModel):
    query: str
    use_rewrite: Optional[bool] = False
    top_k: Optional[int] = 10

class AskResponse(BaseModel):
    question: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: list
    timings: dict
    security: Optional[dict] = None
    agents_used: Optional[list] = None       # Which agents ran
    validation: Optional[dict] = None        # Validation results

class SearchResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    results: list
    total_results: int

class GraphResponse(BaseModel):
    entity: str
    related: list
    subgraph: Optional[dict] = None

class SecurityCheckRequest(BaseModel):
    text: str
    check_type: str = "input"  # "input", "chunk", "output", "redact"

# ──────────────────────────────────────────────
# Load all components on startup
# ──────────────────────────────────────────────

# These are loaded ONCE when the server starts
hybrid_search = None
reranker = None
answer_gen = None
knowledge_graph = None
chunks_lookup = None


@app.on_event("startup")
async def startup():
    """Load all RAG components when server starts."""
    global hybrid_search, reranker, answer_gen, knowledge_graph, chunks_lookup

    print("\n" + "=" * 60)
    print("  STARTING RAG API SERVER")
    print("=" * 60)

    # Load hybrid search (BM25 + FAISS)
    from hybrid_search import HybridSearch
    hybrid_search = HybridSearch()

    # Load reranker
    from reranker import Reranker
    reranker = Reranker()

    # Load answer generator (needs Gemini key)
    from config import GROQ_API_KEY
    if GROQ_API_KEY:
        from answer_generator import AnswerGenerator
        answer_gen = AnswerGenerator()
    else:
        print("  WARNING: GROQ_API_KEY not set. /ask endpoint won't work.")
        print("  /search and /graph endpoints will still work.")

    # Load knowledge graph
    from knowledge_graph import KnowledgeGraph
    knowledge_graph = KnowledgeGraph(load_existing=True)

    # Build chunks lookup for quick access
    from config import CHUNKS_PATH
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunks_lookup = {c["chunk_id"]: c for c in chunks}

    layer_info = "Layer 1 (Rule-based)"
    if ENABLE_LAYER2:
        layer_info += " + Layer 2 (ML Models)"
    print(f"  Security Layer: LOADED ({layer_info})")

    # Print loaded agents
    agent_names = ["ResearchAgent", "RetrievalAgent",
                   "SummarizationAgent", "ValidationAgent", "SecurityAgent"]
    print(f"  Multi-Agent System: {len(agent_names)} agents loaded")
    for name in agent_names:
        print(f"    ✓ {name}")

    print("\n" + "=" * 60)
    print("  RAG API SERVER READY!")
    print("  Open http://localhost:8000/docs for interactive API docs")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
# Endpoint 1: Health check
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check if the server is running and all components are loaded."""
    return {
        "status": "ok",
        "components": {
            "hybrid_search": hybrid_search is not None,
            "reranker": reranker is not None,
            "answer_generator": answer_gen is not None,
            "knowledge_graph": knowledge_graph is not None,
            "security_layer": True,
            "security_layer2": ENABLE_LAYER2,
        },
        "agents": {
            "research_agent": True,
            "retrieval_agent": True,
            "summarization_agent": True,
            "validation_agent": True,
            "security_agent": True,
        },
        "graph_stats": knowledge_graph.get_stats() if knowledge_graph else None,
    }


# ──────────────────────────────────────────────
# Endpoint 2: Full RAG (ask a question)
# ──────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Full RAG pipeline with multi-agent system:
      1. SecurityAgent   → check user input
      2. ResearchAgent   → expand query into research plan
      3. Hybrid Search   → find relevant chunks
      4. Reranker        → rerank by relevance
      5. RetrievalAgent  → score and filter chunks
      6. SummarizationAgent → condense chunks
      7. Answer Generator    → generate answer
      8. ValidationAgent     → verify answer against sources
      9. SecurityAgent       → check output for PII
    """
    if answer_gen is None:
        raise HTTPException(
            status_code=503,
            detail="Answer generator not available. Set GROQ_API_KEY and restart."
        )

    # Track which agents ran
    agents_used = []

    # ── Step 1: Security Agent — check user input ──
    input_check = security.check_input(request.question)
    agents_used.append("SecurityAgent (input)")
    if not input_check["safe"]:
        categories = [d["category"] for d in input_check["detections"]]
        return AskResponse(
            question=request.question,
            rewritten_query=None,
            answer=f"Your query was blocked by the security layer.\n\n"
                   f"Reason: {', '.join(categories)}\n"
                   f"Risk level: {input_check['risk_level'].upper()}",
            sources=[],
            timings={"security": 0.0},
            security=input_check,
            agents_used=agents_used,
        )

    try:
        timings = {}

        # ── Step 2: Research Agent — expand query ──
        search_query = request.question
        research_result = None
        if request.use_research_agent:
            start = time.time()
            research_result = researcher.run(query=request.question)
            timings["research_agent"] = round(time.time() - start, 2)
            agents_used.append("ResearchAgent")

            if research_result["success"]:
                # Use the improved query for search
                search_query = research_result.get("rewritten_query", request.question)

        # ── Step 3: Hybrid search ──
        start = time.time()
        search_result = hybrid_search.search(
            search_query,
            use_rewrite=request.use_rewrite,
            use_multi_query=request.use_multi_query,
            top_k=request.top_k * 2,  # Get more for reranking
        )
        timings["search"] = round(time.time() - start, 2)

        # ── Step 4: Rerank ──
        start = time.time()
        reranked = reranker.rerank(
            request.question,  # Use original question for reranking
            search_result["results"],
            top_k=request.top_k,
        )
        timings["rerank"] = round(time.time() - start, 2)

        # ── Step 5: Security — check each retrieved chunk ──
        start = time.time()
        safe_chunks = []
        flagged_chunks = []
        for chunk_id, score in reranked:
            chunk_text = chunks_lookup.get(chunk_id, {}).get("text", "")
            chunk_check = security.check_retrieved_chunk(chunk_text)
            if chunk_check["safe"]:
                safe_chunks.append((chunk_id, score))
            else:
                flagged_chunks.append({
                    "chunk_id": chunk_id,
                    "reason": [d["category"] for d in chunk_check["detections"]],
                })
        timings["security_chunk_scan"] = round(time.time() - start, 2)
        agents_used.append("SecurityAgent (chunks)")

        # Use only safe chunks for answer generation
        chunks_for_answer = safe_chunks if safe_chunks else reranked

        # ── Step 6: Retrieval Agent — score and filter chunks ──
        if request.use_retrieval_agent and chunks_for_answer:
            start = time.time()
            chunk_dicts = [
                {
                    "chunk_id": cid,
                    "text": chunks_lookup.get(cid, {}).get("text", ""),
                    "score": score,
                }
                for cid, score in chunks_for_answer
            ]
            retrieval_result = retrieval_agent.run(
                chunks=chunk_dicts,
                query=request.question,
                relevance_threshold=0.3,
            )
            timings["retrieval_agent"] = round(time.time() - start, 2)
            agents_used.append("RetrievalAgent")

            if retrieval_result["success"] and retrieval_result["filtered_chunks"]:
                # Rebuild chunks_for_answer from filtered results
                chunks_for_answer = [
                    (c["chunk_id"], c["relevance_score"])
                    for c in retrieval_result["filtered_chunks"]
                ]

        # ── Step 7: Summarization Agent — condense chunks ──
        summary_text = None
        if request.use_summarizer and chunks_for_answer:
            start = time.time()
            chunk_texts = [
                chunks_lookup.get(cid, {}).get("text", "")
                for cid, score in chunks_for_answer
            ]
            summary_result = summarizer.run(
                chunks=chunk_texts,
                query=request.question,
                mode="query_focused",
            )
            if summary_result["success"]:
                summary_text = summary_result["summary"]
            timings["summarization"] = round(time.time() - start, 2)
            agents_used.append("SummarizationAgent")

        # ── Step 8: Generate answer ──
        start = time.time()
        if summary_text:
            # Inject summary into the first chunk temporarily
            first_cid = chunks_for_answer[0][0]
            original_text = chunks_lookup.get(first_cid, {}).get("text", "")
            chunks_lookup.setdefault(first_cid, {})["text"] = summary_text
            try:
                answer_result = answer_gen.generate_answer(
                    request.question, chunks_for_answer
                )
            finally:
                chunks_lookup[first_cid]["text"] = original_text
        else:
            answer_result = answer_gen.generate_answer(request.question, chunks_for_answer)
        timings["answer"] = round(time.time() - start, 2)

        # ── Step 9: Validation Agent — check for hallucinations ──
        validation_info = None
        if request.use_validator:
            start = time.time()
            chunk_texts_for_val = [
                chunks_lookup.get(cid, {}).get("text", "")
                for cid, score in chunks_for_answer
            ]
            val_result = validator.run(
                answer=answer_result["answer"],
                chunks=chunk_texts_for_val,
                query=request.question,
            )
            timings["validation_agent"] = round(time.time() - start, 2)
            agents_used.append("ValidationAgent")

            if val_result["success"]:
                validation_info = {
                    "is_valid": val_result["is_valid"],
                    "confidence": val_result["confidence"],
                    "total_claims": val_result["total_claims"],
                    "supported_claims": val_result["supported_claims"],
                    "flagged_claims": val_result["flagged_claims"],
                    "summary": val_result["summary"],
                }

                # If answer is invalid, add a warning
                if not val_result["is_valid"]:
                    answer_result["answer"] += (
                        "\n\n⚠️ Note: Some claims in this answer could not be "
                        "fully verified against the source documents."
                    )

        # ── Step 10: Security — check LLM output for PII ──
        start = time.time()
        output_check = security.check_output(answer_result["answer"])
        if not output_check["safe"]:
            answer_result["answer"] = security.redact_output(answer_result["answer"])
        timings["security_output_scan"] = round(time.time() - start, 2)
        agents_used.append("SecurityAgent (output)")

        timings["total"] = round(sum(timings.values()), 2)

        security_info = {
            "input_safe": input_check["safe"],
            "chunks_flagged": len(flagged_chunks),
            "flagged_details": flagged_chunks if flagged_chunks else None,
            "output_pii_redacted": not output_check["safe"],
        }

        return AskResponse(
            question=request.question,
            rewritten_query=search_query if search_query != request.question else
                            search_result.get("rewritten_query"),
            answer=answer_result["answer"],
            sources=answer_result["sources"],
            timings=timings,
            security=security_info,
            agents_used=agents_used,
            validation=validation_info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Endpoint 3: Search only (no Gemini needed)
# ──────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search_chunks(request: SearchRequest):
    """
    Hybrid search + reranking only. Returns relevant chunks
    without generating an answer. No Gemini API needed.
    """
    # ── Security: check search query ──
    input_check = security.check_input(request.query)
    if not input_check["safe"]:
        categories = [d["category"] for d in input_check["detections"]]
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked by security layer: {', '.join(categories)} "
                   f"(risk: {input_check['risk_level']})"
        )

    try:
        # Hybrid search (no rewrite or multi-query by default)
        search_result = hybrid_search.search(
            request.query,
            use_rewrite=request.use_rewrite,
            use_multi_query=False,
            top_k=request.top_k,
        )

        # Rerank
        reranked = reranker.rerank(
            request.query,
            search_result["results"],
            top_k=request.top_k,
        )

        # Build response with chunk details
        results = []
        for chunk_id, score in reranked:
            chunk = chunks_lookup.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "file_name": chunk.get("file_name", "unknown"),
                "language": chunk.get("language", "unknown"),
                "score": round(score, 4),
            })

        return SearchResponse(
            query=request.query,
            rewritten_query=search_result.get("rewritten_query"),
            results=results,
            total_results=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Endpoint 4: Knowledge graph lookup
# ──────────────────────────────────────────────

@app.get("/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    return knowledge_graph.get_stats()


@app.get("/graph/{entity}", response_model=GraphResponse)
async def graph_lookup(entity: str, depth: int = 1, top_k: int = 10):
    """
    Look up an entity in the knowledge graph.
    Returns related entities and a subgraph for visualization.
    """
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")

    related = knowledge_graph.find_related_entities(entity, top_k=top_k)
    subgraph = knowledge_graph.get_subgraph(entity, depth=depth)

    if not related and not subgraph["nodes"]:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity}' not found in knowledge graph"
        )

    return GraphResponse(
        entity=entity,
        related=related,
        subgraph=subgraph,
    )


# ──────────────────────────────────────────────
# Endpoint 5: List available entities
# ──────────────────────────────────────────────

@app.get("/graph/entities/search")
async def search_entities(q: str = Query(..., min_length=2), limit: int = 20):
    """Search for entities by name (for autocomplete in UI)."""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")

    q_lower = q.lower()
    matches = []

    for node in knowledge_graph.graph.nodes():
        if q_lower in node.lower():
            attrs = knowledge_graph.graph.nodes[node]
            matches.append({
                "name": node,
                "type": attrs.get("entity_type", "?"),
                "connections": knowledge_graph.graph.degree(node),
            })

    # Sort by connections (most connected first)
    matches.sort(key=lambda x: x["connections"], reverse=True)
    return matches[:limit]


# ──────────────────────────────────────────────
# Endpoint 6: Security check (test endpoint)
# ──────────────────────────────────────────────

@app.post("/api/security/check")
async def security_check(request: SecurityCheckRequest):
    """
    Test the security layer directly from the UI.
    check_type: "input" | "chunk" | "output" | "redact"
    """
    if request.check_type == "input":
        return security.check_input(request.text)
    elif request.check_type == "chunk":
        return security.check_retrieved_chunk(request.text)
    elif request.check_type == "output":
        return security.check_output(request.text)
    elif request.check_type == "redact":
        return {"redacted": security.redact_output(request.text)}
    else:
        raise HTTPException(status_code=400, detail="Invalid check_type. Use: input, chunk, output, redact")


# ──────────────────────────────────────────────
# Run with: uvicorn api:app --reload --port 8000
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
