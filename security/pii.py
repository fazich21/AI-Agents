"""
Layer 1: PII Detection & Redaction
====================================

Deterministic regex-based detection of personally identifiable
information (PII):
  - Email addresses
  - Pakistani phone numbers  (03xx-xxxxxxx, +923xxxxxxxxx)
  - CNIC numbers             (xxxxx-xxxxxxx-x)
  - Credit-card-like numbers (13-19 digit sequences with optional separators)
  - IPv4 addresses

Also provides redact_pii() to mask detected values in text.

Design Notes:
  - detect_pii() never returns the full sensitive value in its
    output; matches are masked/truncated for safe logging.
  - redact_pii() replaces each value with a labelled placeholder.
  - Layer 2 can later add NER-based PII extraction without
    changing these functions.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# PII regex patterns  (compiled once at import time)
# ---------------------------------------------------------------------------

# Email: standard RFC-5322-ish pattern
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Pakistani phone: local (03xx...) or international (+923xx...)
_PK_PHONE_RE = re.compile(
    r"(?<!\d)"                           # not preceded by a digit
    r"(?:"
        r"\+92[\s\-]?3\d{2}[\s\-]?\d{7}" # +92-3xx-xxxxxxx
        r"|"
        r"03\d{2}[\s\-]?\d{7}"           # 03xx-xxxxxxx / 03xxxxxxxxx
    r")"
    r"(?!\d)"                            # not followed by a digit
)

# CNIC: 5-7-1 format with dashes
_CNIC_RE = re.compile(
    r"\b\d{5}-\d{7}-\d\b"
)

# Credit card: 13-19 digits, optionally separated by spaces or dashes
# in groups of 4.  We avoid matching plain long numbers that are clearly
# not cards by requiring at least 13 digits.
_CREDIT_CARD_RE = re.compile(
    r"(?<!\d)"
    r"(?:\d{4}[\s\-]?){2,4}\d{1,4}"
    r"(?!\d)"
)

# IPv4 address
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)


# Map of (type_label, compiled_regex, redaction_tag)
_PII_DEFINITIONS = [
    ("email",       _EMAIL_RE,       "[EMAIL_REDACTED]"),
    ("phone",       _PK_PHONE_RE,    "[PHONE_REDACTED]"),
    ("cnic",        _CNIC_RE,        "[CNIC_REDACTED]"),
    ("credit_card", _CREDIT_CARD_RE, "[CC_REDACTED]"),
    ("ip_address",  _IPV4_RE,        "[IP_REDACTED]"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_value(value: str, pii_type: str) -> str:
    """Return a safely truncated representation for logging."""
    if pii_type == "email":
        # show first char + domain: a***@example.com
        parts = value.split("@")
        if len(parts) == 2:
            return parts[0][0] + "***@" + parts[1]
    elif pii_type == "phone":
        return value[:4] + "***" + value[-2:]
    elif pii_type == "cnic":
        return value[:6] + "*******" + value[-1:]
    elif pii_type == "credit_card":
        digits_only = re.sub(r"[\s\-]", "", value)
        return "****-****-****-" + digits_only[-4:]
    elif pii_type == "ip_address":
        return value.split(".")[0] + ".*.*.*"
    # Fallback
    return value[:3] + "***"


def _validate_credit_card(raw: str) -> bool:
    """Quick check: must be 13-19 pure digits after stripping separators."""
    digits = re.sub(r"[\s\-]", "", raw)
    return digits.isdigit() and 13 <= len(digits) <= 19


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_pii(text: str) -> dict[str, Any]:
    """
    Scan text for PII patterns.

    Returns:
        {
            "detected": bool,
            "category": "pii",
            "types": ["email", "phone", ...],   # distinct types found
            "matches": [
                {"type": "email", "masked_value": "a***@example.com"},
                ...
            ],
            "risk_level": "high" | "medium" | "low"
        }

    Matched values are masked — the full sensitive value is never
    included in the output to keep logs safe.
    """
    matches: list[dict[str, str]] = []
    types_seen: set[str] = set()

    for pii_type, regex, _tag in _PII_DEFINITIONS:
        for m in regex.finditer(text):
            raw = m.group(0)
            # Extra validation for credit-card patterns
            if pii_type == "credit_card" and not _validate_credit_card(raw):
                continue
            types_seen.add(pii_type)
            matches.append({
                "type": pii_type,
                "masked_value": _mask_value(raw, pii_type),
            })

    # Risk: CNIC or credit card → HIGH; other PII → MEDIUM
    if matches:
        high_risk_types = types_seen & {"cnic", "credit_card"}
        risk = "high" if high_risk_types else "medium"
    else:
        risk = "low"

    return {
        "detected": len(matches) > 0,
        "category": "pii",
        "types": sorted(types_seen),
        "matches": matches,
        "risk_level": risk,
    }


def redact_pii(text: str) -> str:
    """
    Replace all detected PII values with labelled placeholders.

    Example:
        "Ali's email is ali@example.com" →
        "Ali's email is [EMAIL_REDACTED]"
    """
    result = text
    for pii_type, regex, tag in _PII_DEFINITIONS:
        if pii_type == "credit_card":
            # Use a callback to validate before replacing
            def _cc_replace(m: re.Match) -> str:
                return tag if _validate_credit_card(m.group(0)) else m.group(0)
            result = regex.sub(_cc_replace, result)
        else:
            result = regex.sub(tag, result)
    return result
