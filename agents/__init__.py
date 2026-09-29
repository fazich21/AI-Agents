"""
agents — Multi-Agent System for RAG Pipeline (Task 5)
=====================================================

A modular agent framework where each agent handles a specific
sub-task in the RAG pipeline. Agents communicate through a
shared context dict and can be orchestrated by the Orchestrator.

Available Agents:
    - SummarizationAgent: Condenses retrieved chunks into focused summaries
    - (More agents to be added: QueryPlannerAgent, FactCheckerAgent, etc.)

Usage:
    from agents import SummarizationAgent

    agent = SummarizationAgent()
    result = agent.run(chunks=["chunk1 text...", "chunk2 text..."],
                       query="What is the leave policy?")
"""

from agents.base_agent import BaseAgent
from agents.summarization_agent import SummarizationAgent

__all__ = [
    "BaseAgent",
    "SummarizationAgent",
]
