"""
Base Agent — Abstract interface for all RAG pipeline agents.
============================================================

Every agent in the multi-agent system inherits from BaseAgent.
This gives a consistent interface: each agent has a name, a
description, and a `run()` method that takes keyword arguments
and returns a result dict.

Design principles:
  - Agents are stateless: each `run()` call is independent.
  - Agents return a standard dict with at least:
      {"agent": "<name>", "success": True/False, ...results}
  - Agents handle their own errors gracefully.
  - Agents can be composed (one agent's output feeds another).
"""

from abc import ABC, abstractmethod
from typing import Any
import time


class BaseAgent(ABC):
    """Abstract base class for all pipeline agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            **kwargs: Task-specific inputs (chunks, query, etc.)

        Returns:
            dict with at least {"agent": name, "success": bool}
        """
        pass

    def _build_result(self, success: bool, **data) -> dict[str, Any]:
        """Helper to build a standardized result dict."""
        return {
            "agent": self.name,
            "success": success,
            "timestamp": time.time(),
            **data,
        }

    def _error_result(self, error: str) -> dict[str, Any]:
        """Helper to build an error result."""
        return self._build_result(success=False, error=error)

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"
