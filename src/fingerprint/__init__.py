"""
Behavioral Fingerprint Extraction Module
========================================
Deep Learning Model Authentication Using Behavioral Challenge–Response
"""

from src.fingerprint.fingerprint import (
    DEFAULT_CHALLENGE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    extract_fingerprint,
    load_model,
    save_fingerprint,
    verify_reproducibility,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_CHALLENGE_PATH",
    "DEFAULT_OUTPUT_PATH",
    "load_model",
    "extract_fingerprint",
    "save_fingerprint",
    "verify_reproducibility",
]

