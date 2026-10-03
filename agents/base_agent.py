

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
