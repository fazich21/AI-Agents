

import time
from typing import Any, Optional

from agents.base_agent import BaseAgent


class SecurityAgent(BaseAgent):
    """
    Agent wrapper around the Task 4 SecurityLayer.

    Provides a consistent agent interface for security checks:
      - check_type="input":   scan user queries
      - check_type="chunk":   scan retrieved chunks
      - check_type="output":  scan LLM answers + auto-redact PII

    Does NOT use LLM — uses rule-based and ML-based detectors.
    """

    def __init__(self, security_layer=None, enable_layer2: bool = False):
        super().__init__(
            name="SecurityAgent",
            description="Scans text for security threats, PII, and unsafe content",
        )
        # Use provided SecurityLayer or create a new one
        if security_layer is not None:
            self._security = security_layer
        else:
            try:
                from security import SecurityLayer
                self._security = SecurityLayer(enable_layer2=enable_layer2)
            except ImportError:
                self._security = None

    # ----- Main run method -----
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Run a security check on the given text.

        Args:
            text:        the text to check
            check_type:  "input", "chunk", or "output" (default "input")

        Returns:
            {
                "agent": "SecurityAgent",
                "success": True/False,
                "safe": True/False,
                "risk_level": "none" | "low" | "medium" | "high",
                "detections": [...],
                "redacted_text": "..." (only for output check when PII found),
                "check_type": "input",
                "time_taken": 0.01,
            }
        """
        text = kwargs.get("text", "")
        check_type = kwargs.get("check_type", "input")

        if not text:
            return self._error_result("No text provided to check")

        if check_type not in ("input", "chunk", "output"):
            return self._error_result(
                f"Unknown check_type: {check_type}. Use: input, chunk, output"
            )

        if self._security is None:
            return self._error_result(
                "SecurityLayer not available. Make sure security.py is importable."
            )

        start_time = time.time()

        try:
            if check_type == "input":
                result = self._check_input(text)
            elif check_type == "chunk":
                result = self._check_chunk(text)
            else:  # output
                result = self._check_output(text)

            elapsed = round(time.time() - start_time, 4)

            return self._build_result(
                success=True,
                check_type=check_type,
                time_taken=elapsed,
                **result,
            )

        except Exception as e:
            return self._error_result(str(e))

    # ----- Check strategies -----

    def _check_input(self, text: str) -> dict:
        """Check user input for prompt injection, jailbreaks, etc."""
        check = self._security.check_input(text)
        return {
            "safe": check["safe"],
            "risk_level": check.get("risk_level", "none"),
            "detections": check.get("detections", []),
        }

    def _check_chunk(self, text: str) -> dict:
        """Check a retrieved chunk for toxic/unsafe content."""
        check = self._security.check_retrieved_chunk(text)
        return {
            "safe": check["safe"],
            "risk_level": check.get("risk_level", "none"),
            "detections": check.get("detections", []),
        }

    def _check_output(self, text: str) -> dict:
        """Check LLM output for PII leaks and auto-redact if needed."""
        check = self._security.check_output(text)
        result = {
            "safe": check["safe"],
            "risk_level": check.get("risk_level", "none"),
            "detections": check.get("detections", []),
        }
        # Auto-redact PII if found
        if not check["safe"]:
            result["redacted_text"] = self._security.redact_output(text)
        return result
