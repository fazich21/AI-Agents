"""
security — Multi-Layer Security for RAG Pipelines
===================================================

Layer 1: Rule-based / regex detection (always available)
Layer 2: Pretrained model-based detection (optional)

Public API:
    from security import SecurityLayer

    # Layer 1 only
    sec = SecurityLayer()

    # Layer 1 + Layer 2 (requires transformers, torch, spacy)
    sec = SecurityLayer(enable_layer2=True)

    # Individual detectors
    from security.detectors import (
        detect_prompt_injection,
        detect_indirect_injection,
        detect_jailbreak,
        detect_data_exfiltration,
        detect_toxic_content,
    )
    from security.pii import detect_pii, redact_pii

    # Layer 2 detectors (optional)
    from security.layer2_detectors import (
        detect_injection_ml,
        detect_toxicity_ml,
        detect_pii_ner,
        redact_pii_ner,
    )
"""

from security.security_layer import SecurityLayer
from security.detectors import (
    detect_prompt_injection,
    detect_indirect_injection,
    detect_jailbreak,
    detect_data_exfiltration,
    detect_toxic_content,
)
from security.pii import detect_pii, redact_pii

__all__ = [
    "SecurityLayer",
    "detect_prompt_injection",
    "detect_indirect_injection",
    "detect_jailbreak",
    "detect_data_exfiltration",
    "detect_toxic_content",
    "detect_pii",
    "redact_pii",
]

# Conditionally export Layer 2 functions
try:
    from security.layer2_detectors import (
        detect_injection_ml,
        detect_toxicity_ml,
        detect_pii_ner,
        redact_pii_ner,
        preload_all_models,
    )
    __all__ += [
        "detect_injection_ml",
        "detect_toxicity_ml",
        "detect_pii_ner",
        "redact_pii_ner",
        "preload_all_models",
    ]
except ImportError:
    pass
