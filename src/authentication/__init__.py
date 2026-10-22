"""
Authentication Module
=====================
Behavioral challenge-response model authentication engine.
"""

from src.authentication.authenticator import (
    AuthenticationMetrics,
    AuthenticationReport,
    AuthenticationThresholds,
    FingerprintValidationError,
    ModelAuthenticator,
    authenticate_fingerprints,
    calculate_cosine_similarity,
    calculate_kl_divergence,
    calculate_prediction_agreement,
    validate_fingerprint_alignment,
    validate_fingerprint_schema,
)

__all__ = [
    "AuthenticationMetrics",
    "AuthenticationReport",
    "AuthenticationThresholds",
    "FingerprintValidationError",
    "ModelAuthenticator",
    "authenticate_fingerprints",
    "calculate_cosine_similarity",
    "calculate_kl_divergence",
    "calculate_prediction_agreement",
    "validate_fingerprint_alignment",
    "validate_fingerprint_schema",
]
