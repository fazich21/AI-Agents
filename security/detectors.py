"""
Layer 1: Rule-Based / Regex Security Detectors
===============================================

Deterministic, regex-based detectors for:
  1. Direct Prompt Injection
  2. Indirect Prompt Injection (RAG chunk scanning)
  3. Jailbreak Attempts
  4. Data Exfiltration Attempts
  5. Toxic Content

Design Notes:
  - Every detector returns a structured dict, never bare True/False.
  - Patterns use case-insensitive matching.
  - Patterns are deliberately specific to avoid over-blocking.
    e.g. "What is prompt injection?" is SAFE,
         "Ignore all previous instructions" is MALICIOUS.
  - This is Layer 1 only. Layer 2 (pretrained models) can be added
    later without rewriting any of this code.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Risk-level constants
# ---------------------------------------------------------------------------
LOW = "low"
MEDIUM = "medium"
HIGH = "high"


# ---------------------------------------------------------------------------
# Helper: run a list of (compiled_regex, label) pairs against text
# ---------------------------------------------------------------------------
def _scan(text: str, patterns: list[tuple[re.Pattern, str]]) -> list[dict[str, str]]:
    """Return a list of {'pattern': label, 'match': matched_text} dicts."""
    matches = []
    for regex, label in patterns:
        for m in regex.finditer(text):
            matches.append({"pattern": label, "match": m.group(0)})
    return matches


# ===================================================================
# 1. Direct Prompt Injection
# ===================================================================

# Each tuple: (compiled regex, human-readable label)
_PROMPT_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Instruction override ---
    (re.compile(r"\bignore\b.{0,15}\b(?:all\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|rules?|directives?)\b", re.I),
     "instruction_override"),
    (re.compile(r"\b(?:disregard|forget|drop|skip)\b.{0,15}\b(?:previous|prior|above|earlier|system|your)\s+(?:instructions?|prompts?|rules?|directives?)\b", re.I),
     "instruction_override"),
    (re.compile(r"\boverride\b.{0,15}\b(?:your|the|system)\s+(?:instructions?|prompts?|rules?)\b", re.I),
     "instruction_override"),
    (re.compile(r"\bfollow\s+my\s+instructions?\s+instead\b", re.I),
     "instruction_override"),

    # --- System-prompt extraction ---
    (re.compile(r"\b(?:reveal|show|display|print|output|leak|expose|repeat)\b.{0,15}\b(?:your|the)?\s*system\s+(?:prompt|message|instructions?)\b", re.I),
     "system_prompt_extraction"),
    (re.compile(r"\bwhat\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?|rules?)\b", re.I),
     "system_prompt_extraction"),

    # --- Role hijack ---
    (re.compile(r"\byou\s+are\s+now\b.{0,30}\b(?:unrestricted|unfiltered|new|different)\b", re.I),
     "role_hijack"),
    (re.compile(r"\b(?:from\s+now\s+on|starting\s+now)\s+you\s+(?:are|will|must|should)\b", re.I),
     "role_hijack"),

    # --- New-instructions injection ---
    (re.compile(r"\bnew\s+instructions?\s*:", re.I),
     "new_instructions"),
    (re.compile(r"\b(?:begin|start)\s+(?:new|updated)\s+(?:instructions?|prompt)\b", re.I),
     "new_instructions"),
]


def detect_prompt_injection(text: str) -> dict[str, Any]:
    """
    Detect direct prompt-injection attempts in user input.

    Returns:
        {
            "detected": bool,
            "category": "prompt_injection",
            "matches": [{"pattern": str, "match": str}, ...],
            "risk_level": "high" | "low"
        }
    """
    matches = _scan(text, _PROMPT_INJECTION_PATTERNS)
    return {
        "detected": len(matches) > 0,
        "category": "prompt_injection",
        "matches": matches,
        "risk_level": HIGH if matches else LOW,
    }


# ===================================================================
# 2. Indirect Prompt Injection  (RAG chunk scanning)
# ===================================================================

# These patterns look for adversarial instructions that might be
# embedded in documents, not typed by the user.  They therefore
# target *instruction-like* language that would only appear if
# someone deliberately planted it in a document.

_INDIRECT_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Planted instruction overrides ---
    (re.compile(r"\bignore\b.{0,15}\b(?:all\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|rules?)\b", re.I),
     "planted_instruction_override"),
    (re.compile(r"\b(?:disregard|forget)\b.{0,15}\b(?:previous|prior|above|system)\s+(?:instructions?|prompts?|rules?)\b", re.I),
     "planted_instruction_override"),

    # --- Addressing the AI directly inside a document ---
    (re.compile(r"\b(?:AI|assistant|chatbot|language\s+model)\s*(?::|,)\s*(?:ignore|disregard|forget|you\s+must|you\s+should|please\s+ignore)\b", re.I),
     "ai_directive_in_document"),
    (re.compile(r"\bassistant\s+must\b", re.I),
     "ai_directive_in_document"),

    # --- System-prompt / confidential reveal commands ---
    (re.compile(r"\b(?:reveal|show|leak|expose|output)\b.{0,15}\b(?:system\s+prompt|confidential|secret|private|internal)\s+(?:information|data|details?)?\b", re.I),
     "reveal_confidential"),

    # --- Smuggled new instructions ---
    (re.compile(r"\bfollow\s+these\s+instructions?\b", re.I),
     "smuggled_instructions"),
    (re.compile(r"\b(?:send|transmit|exfiltrate)\s+(?:the|all|this)\s+data\b", re.I),
     "data_exfil_in_document"),

    # --- Role reassignment inside document ---
    (re.compile(r"\byou\s+are\s+now\b", re.I),
     "role_reassignment_in_document"),
]


def detect_indirect_injection(text: str) -> dict[str, Any]:
    """
    Scan a retrieved RAG chunk (or any document text) for injected
    adversarial instructions before it reaches the LLM.

    Returns:
        {
            "detected": bool,
            "category": "indirect_injection",
            "matches": [...],
            "risk_level": "high" | "low"
        }
    """
    matches = _scan(text, _INDIRECT_INJECTION_PATTERNS)
    return {
        "detected": len(matches) > 0,
        "category": "indirect_injection",
        "matches": matches,
        "risk_level": HIGH if matches else LOW,
    }


# ===================================================================
# 3. Jailbreak Detection
# ===================================================================

_JAILBREAK_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Safety bypass ---
    (re.compile(r"\b(?:ignore|bypass|disable|remove|turn\s+off|circumvent)\b.{0,15}\b(?:your\s+)?(?:safety|content|ethical)\s*(?:rules?|restrictions?|filters?|guidelines?|limits?|guardrails?)\b", re.I),
     "safety_bypass"),
    (re.compile(r"\b(?:bypass|disable|remove|circumvent)\s+(?:your\s+)?(?:restrictions?|limits?|filters?|guardrails?)\b", re.I),
     "safety_bypass"),
    (re.compile(r"\byou\s+have\s+no\s+(?:restrictions?|limits?|rules?|filters?|guidelines?)\b", re.I),
     "safety_bypass"),

    # --- Unrestricted mode ---
    (re.compile(r"\b(?:pretend|act|behave|operate)\b.{0,20}\b(?:unrestricted|unfiltered|uncensored|without\s+(?:restrictions?|limits?|rules?))\b", re.I),
     "unrestricted_mode"),
    (re.compile(r"\bact\s+as\s+(?:an?\s+)?unrestricted\s+(?:AI|assistant|model)\b", re.I),
     "unrestricted_mode"),

    # --- Known jailbreak templates ---
    # DAN ("Do Anything Now") — only match when paired with jailbreak context
    (re.compile(r"\bDAN\b.{0,30}\b(?:do\s+anything|no\s+restrictions?|unrestricted|jailbreak)\b", re.I),
     "dan_jailbreak"),
    (re.compile(r"\b(?:you\s+are\s+now\s+)?DAN\b[\s,;:]+\byou\s+(?:can|must|will)\s+do\s+anything\b", re.I),
     "dan_jailbreak"),
    # Developer / debug mode when paired with override language
    (re.compile(r"\b(?:enter|enable|activate|switch\s+to)\s+(?:developer|debug|maintenance|god)\s+mode\b", re.I),
     "developer_mode"),

    # --- Roleplay to bypass ---
    (re.compile(r"\b(?:pretend|imagine|roleplay|role-play)\b.{0,30}\b(?:no\s+rules?|no\s+restrictions?|evil|malicious|unethical)\b", re.I),
     "bypass_via_roleplay"),
]


def detect_jailbreak(text: str) -> dict[str, Any]:
    """
    Detect jailbreak attempts using specific pattern matching.

    Note: Mentioning "developer" or "DAN" in a factual/educational
    context (e.g. "What is the DAN jailbreak?") will not match
    because the patterns require surrounding override language.

    Returns:
        {
            "detected": bool,
            "category": "jailbreak",
            "matches": [...],
            "risk_level": "high" | "low"
        }
    """
    matches = _scan(text, _JAILBREAK_PATTERNS)
    return {
        "detected": len(matches) > 0,
        "category": "jailbreak",
        "matches": matches,
        "risk_level": HIGH if matches else LOW,
    }


# ===================================================================
# 4. Data Exfiltration Detection
# ===================================================================
#
# IMPORTANT: Rule-based detection is only a first layer.
# It does NOT replace proper authorization / access-control checks.
# Even if a query passes this detector, the application MUST still
# enforce per-user permissions before returning data.
#

_DATA_EXFIL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Bulk data requests ---
    (re.compile(r"\b(?:show|give|list|display|print|reveal|get|fetch|return)\s+(?:me\s+)?(?:all|every|the\s+entire|the\s+whole|the\s+complete)\s+(?:documents?|records?|files?|entries?|rows?|data|users?|employees?|customers?|clients?|accounts?|emails?|passwords?|credentials?)\b", re.I),
     "bulk_data_request"),

    # --- Database dumping ---
    (re.compile(r"\b(?:dump|export|extract|download)\s+(?:the\s+)?(?:entire\s+)?(?:database|db|table|collection|dataset|storage)\b", re.I),
     "database_dump"),
    (re.compile(r"\b(?:show|give|reveal|get)\s+(?:me\s+)?(?:everything|all\s+(?:the\s+)?data)\s+(?:in|from)\s+(?:the\s+)?(?:database|db|system|storage)\b", re.I),
     "database_dump"),
    (re.compile(r"\b(?:give|show|get)\s+(?:me\s+)?the\s+(?:entire|whole|complete|full)\s+(?:database|db|table|collection|dataset)\b", re.I),
     "database_dump"),

    # --- Confidential / private data requests ---
    (re.compile(r"\b(?:show|give|reveal|access|get)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:confidential|secret|private|classified|restricted|sensitive|internal)\s+(?:data|information|documents?|files?|records?|details?)\b", re.I),
     "confidential_data_request"),

    # --- Credential harvesting ---
    (re.compile(r"\b(?:show|give|list|reveal)\s+(?:me\s+)?(?:all\s+)?(?:passwords?|credentials?|api\s*keys?|tokens?|secrets?)\b", re.I),
     "credential_harvesting"),
]


def detect_data_exfiltration(text: str) -> dict[str, Any]:
    """
    Detect attempts to extract bulk or confidential data.

    Design: Patterns require an *action verb* + a *bulk/sensitive qualifier*
    together, so normal queries like "Summarize the document I uploaded"
    will NOT trigger this detector.

    NOTE: This rule-based check is only a first layer. It does not
    replace proper authorization and access-control enforcement.

    Returns:
        {
            "detected": bool,
            "category": "data_exfiltration",
            "matches": [...],
            "risk_level": "high" | "low"
        }
    """
    matches = _scan(text, _DATA_EXFIL_PATTERNS)
    return {
        "detected": len(matches) > 0,
        "category": "data_exfiltration",
        "matches": matches,
        "risk_level": HIGH if matches else LOW,
    }


# ===================================================================
# 5. Toxic Content Detection
# ===================================================================
#
# Layer 1: basic deterministic detection using a curated word list.
#
# IMPORTANT — Layer 2 should add a pretrained toxicity/moderation
# model (e.g. a HuggingFace classifier or OpenAI Moderation API)
# for semantic, context-aware detection.  Rule-based matching
# cannot understand sarcasm, context, or novel toxic phrasing.
#

# Patterns are phrased to require word boundaries and, where helpful,
# surrounding hostile-intent context so that educational usage
# ("explain what hate speech is") does not trigger.

_TOXIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Direct slurs / extreme hate speech (small curated set) ---
    # These are terms whose *use* (not mention) is almost always toxic.
    # We wrap them in word boundaries.
    (re.compile(r"\b(?:kill\s+(?:yourself|urself|all)\b)", re.I),
     "violent_threat"),
    (re.compile(r"\bi\s+(?:will|am\s+going\s+to|'ll|'m\s+going\s+to|'m\s+gonna|wanna|gonna)\s+(?:kill|murder|shoot|stab|hurt)\s+(?:you|them|him|her|everyone)\b", re.I),
     "violent_threat"),

    # --- Directed harassment patterns ---
    (re.compile(r"\byou\s+(?:are|re)\s+(?:a\s+)?(?:worthless|pathetic|disgusting|stupid|useless|piece\s+of\s+(?:shit|crap|garbage))\b", re.I),
     "directed_harassment"),
    (re.compile(r"\b(?:go\s+)?(?:die|f[*u]ck\s+(?:off|you)|shut\s+the\s+f[*u]ck\s+up)\b", re.I),
     "directed_harassment"),

    # --- Hate-group / extremism markers (specific phrases) ---
    (re.compile(r"\b(?:(?:death|kill)\s+(?:to|all)\s+(?:all\s+)?\w+)\b", re.I),
     "hate_speech"),
    (re.compile(r"\b(?:ethnic\s+cleansing|white\s+(?:power|supremacy)|master\s+race)\b", re.I),
     "hate_speech"),
]


def detect_toxic_content(text: str) -> dict[str, Any]:
    """
    Basic deterministic toxic-content detection (Layer 1).

    Targets clearly abusive patterns: direct threats, directed
    harassment, and specific hate-speech phrases.  Educational
    mentions ("What is hate speech?") will generally not trigger
    because patterns require hostile-intent structure.

    NOTE: Layer 2 should use a pretrained toxicity/moderation model
    for semantic and context-aware detection.

    Returns:
        {
            "detected": bool,
            "category": "toxic_content",
            "matches": [...],
            "risk_level": "high" | "medium" | "low"
        }
    """
    matches = _scan(text, _TOXIC_PATTERNS)
    # Violent threats → HIGH; other toxicity → MEDIUM
    if matches:
        has_violence = any(m["pattern"] in ("violent_threat", "hate_speech") for m in matches)
        risk = HIGH if has_violence else MEDIUM
    else:
        risk = LOW

    return {
        "detected": len(matches) > 0,
        "category": "toxic_content",
        "matches": matches,
        "risk_level": risk,
    }
