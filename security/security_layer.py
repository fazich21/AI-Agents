"""
Unified Security Layer (Layer 1 + Layer 2)
============================================

Orchestrates all rule-based AND model-based detectors into a single
SecurityLayer class with four entry points:

  check_input(text)           — user prompt
  check_retrieved_chunk(text) — each RAG chunk before it reaches the LLM
  check_output(text)          — LLM-generated answer before it reaches the user
  check_chat_history(chat_history)  — scan a list of past messages
  check_conversation(chat_history, current_prompt)
                              — combine recent history + current prompt

Layers:
  Layer 1: Rule-based / regex detectors (always available, fast)
  Layer 2: Pretrained model-based detectors (optional, semantic)
           - Prompt injection: DeBERTa v3 classifier
           - Toxicity: DistilBERT toxic-comment-model
           - PII NER: spaCy Named Entity Recognition

Risk-level aggregation:
  HIGH   > MEDIUM > LOW
  The unified result takes the highest risk level across all detections.

Design:
  - Modular: each detector lives in its own function.
  - Layer 2 is opt-in via enable_layer2=True in SecurityLayer().
  - If Layer 2 models fail to load, Layer 1 still works as fallback.
"""

from typing import Any

from security.detectors import (
    detect_prompt_injection,
    detect_indirect_injection,
    detect_jailbreak,
    detect_data_exfiltration,
    detect_toxic_content,
    HIGH, MEDIUM, LOW,
)
from security.pii import detect_pii, redact_pii

# Layer 2 imports (optional — may not be installed)
try:
    from security.layer2_detectors import (
        detect_injection_ml,
        detect_toxicity_ml,
        detect_pii_ner,
        redact_pii_ner,
        preload_all_models,
    )
    _LAYER2_AVAILABLE = True
except ImportError:
    _LAYER2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Risk aggregation
# ---------------------------------------------------------------------------
_RISK_ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2}


def _max_risk(*levels: str) -> str:
    """Return the highest risk level among the inputs."""
    best = LOW
    for lvl in levels:
        if _RISK_ORDER.get(lvl, 0) > _RISK_ORDER[best]:
            best = lvl
    return best


# ---------------------------------------------------------------------------
# Unified result builder
# ---------------------------------------------------------------------------

def _build_result(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge individual detector outputs into a single unified result.

    Returns:
        {
            "safe": True/False,
            "risk_level": "low" | "medium" | "high",
            "detections": [ ... only the positive detections ... ]
        }
    """
    positives = [d for d in detections if d.get("detected")]
    if not positives:
        return {"safe": True, "risk_level": LOW, "detections": []}

    overall_risk = _max_risk(*(d["risk_level"] for d in positives))
    return {
        "safe": False,
        "risk_level": overall_risk,
        "detections": positives,
    }


# ---------------------------------------------------------------------------
# SecurityLayer class
# ---------------------------------------------------------------------------

class SecurityLayer:
    """
    Unified security gate for a RAG pipeline (Layer 1 + optional Layer 2).

    Usage:
        # Layer 1 only (fast, no dependencies)
        sec = SecurityLayer()

        # Layer 1 + Layer 2 (requires: pip install transformers torch spacy)
        sec = SecurityLayer(enable_layer2=True)

        # Check user input
        result = sec.check_input("ignore previous instructions")
        if not result["safe"]:
            print("Blocked:", result)

        # Check each retrieved chunk
        for chunk in retrieved_chunks:
            result = sec.check_retrieved_chunk(chunk)
            if not result["safe"]:
                # skip or flag this chunk
                ...

        # Check LLM output before returning to the user
        result = sec.check_output(llm_response)
    """

    def __init__(self, enable_layer2: bool = False):
        """
        Args:
            enable_layer2: If True, load and use pretrained ML models
                           alongside rule-based detectors. Requires
                           transformers, torch, spacy to be installed.
        """
        self.layer2_enabled = False

        if enable_layer2:
            if not _LAYER2_AVAILABLE:
                print("  [SecurityLayer] WARNING: Layer 2 requested but "
                      "dependencies not installed (transformers, torch, spacy).")
                print("  [SecurityLayer] Falling back to Layer 1 only.")
            else:
                self.layer2_enabled = True
                # Pre-load all models so first request is fast
                preload_all_models()

    # ---- User input ----
    def check_input(self, text: str) -> dict[str, Any]:
        """
        Run all relevant detectors on user input:
          - Direct prompt injection
          - Jailbreak
          - Data exfiltration
          - Toxic content
          - PII
        """
        detections = [
            detect_prompt_injection(text),
            detect_jailbreak(text),
            detect_data_exfiltration(text),
            detect_toxic_content(text),
            detect_pii(text),
        ]
        # Layer 2: ML-based checks (injection + toxicity only)
        # NER PII excluded from pipeline — spaCy's small model flags
        # common terms like "AI", "ML", city names as entities, causing
        # false blocks on normal queries. Regex PII (Layer 1) handles
        # structured PII (emails, phones, CNIC, credit cards) precisely.
        if self.layer2_enabled:
            detections.append(detect_injection_ml(text))
            detections.append(detect_toxicity_ml(text))

        return _build_result(detections)

    # ---- Retrieved RAG chunks ----
    def check_retrieved_chunk(self, text: str) -> dict[str, Any]:
        """
        Scan a single retrieved chunk before it is passed to the LLM:
          - Indirect prompt injection
          - Data-exfiltration indicators (embedded commands)
          - Layer 2: ML injection detection, NER PII
        """
        detections = [
            detect_indirect_injection(text),
            detect_data_exfiltration(text),
        ]
        if self.layer2_enabled:
            detections.append(detect_injection_ml(text))

        return _build_result(detections)

    # ---- LLM output ----
    def check_output(self, text: str) -> dict[str, Any]:
        """
        Scan LLM-generated output before returning it to the user:
          - PII (prevent accidental leakage — regex only, not NER)
          - Toxic content
          - Indirect injection echoed through
          - Layer 2: ML toxicity only

        NOTE: NER PII is NOT used on output because spaCy flags
        common terms (AI, ML, city names in educational text) as
        entities, causing false redactions. Regex PII (emails,
        phone numbers, CNIC, credit cards) is precise enough for
        output scanning.
        """
        detections = [
            detect_pii(text),
            detect_toxic_content(text),
            detect_indirect_injection(text),
        ]
        if self.layer2_enabled:
            detections.append(detect_toxicity_ml(text))
            # NER PII intentionally excluded from output — too noisy

        return _build_result(detections)

    # ---- Chat history scanning ----
    def check_chat_history(
        self,
        chat_history: list[dict[str, str]],
        max_messages: int = 10,
    ) -> dict[str, Any]:
        """
        Scan recent chat messages for injection / jailbreak patterns.

        Args:
            chat_history: list of {"role": "user"|"assistant", "content": str}
            max_messages:  how many recent messages to inspect (default 10)

        Returns a unified result covering ALL messages.
        """
        recent = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history
        all_detections: list[dict[str, Any]] = []

        for msg in recent:
            content = msg.get("content", "")
            if not content:
                continue
            # Run the same checks as input on each message
            all_detections.append(detect_prompt_injection(content))
            all_detections.append(detect_jailbreak(content))

        return _build_result(all_detections)

    # ---- Combined conversation check ----
    def check_conversation(
        self,
        chat_history: list[dict[str, str]],
        current_prompt: str,
        max_history: int = 5,
    ) -> dict[str, Any]:
        """
        Check the current prompt AND recent conversation history together.

        This catches multi-turn injection attacks where individual
        messages look benign but the combined context is malicious.

        Args:
            chat_history:  previous messages [{"role": ..., "content": ...}]
            current_prompt: the user's current input
            max_history:    number of recent history messages to include

        Returns a unified result.
        """
        # 1. Check the current prompt fully (all input checks)
        current_result = self.check_input(current_prompt)

        # 2. Check recent history for injection / jailbreak
        history_result = self.check_chat_history(chat_history, max_messages=max_history)

        # 3. Also check the concatenation of recent user messages + current
        #    prompt for patterns that span multiple messages
        recent = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
        user_texts = [m["content"] for m in recent if m.get("role") == "user" and m.get("content")]
        combined_text = " ".join(user_texts + [current_prompt])
        combined_detections = [
            detect_prompt_injection(combined_text),
            detect_jailbreak(combined_text),
        ]

        # Merge all detections
        all_detections = (
            current_result.get("detections", [])
            + history_result.get("detections", [])
            + [d for d in combined_detections if d.get("detected")]
        )

        if not all_detections:
            return {"safe": True, "risk_level": LOW, "detections": []}

        overall_risk = _max_risk(*(d["risk_level"] for d in all_detections))
        return {
            "safe": False,
            "risk_level": overall_risk,
            "detections": all_detections,
        }

    # ---- Convenience: redact PII in output ----
    def redact_output(self, text: str) -> str:
        """
        Redact any PII found in text (useful for sanitizing LLM output).
        Uses Layer 1 regex redaction ONLY (emails, phones, CNIC, CC, IPs).

        NER redaction is NOT used here because spaCy's NER model
        incorrectly flags common terms in LLM-generated educational
        text (e.g. "AI" as ORG, "ML" as FACILITY, city names as
        LOCATION) causing garbled output.
        """
        return redact_pii(text)
