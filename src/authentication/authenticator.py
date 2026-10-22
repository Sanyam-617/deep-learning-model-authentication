"""
Authentication Engine
=====================
Member 3 – Behavioral Challenge-Response Model Authentication

This module compares a reference fingerprint with a submitted/test
fingerprint by validating challenge alignment, calculating three independent
behavioral metrics (prediction agreement, cosine similarity, KL divergence),
and applying configurable threshold-based authentication decision rules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ──────────────────────────────────────────────
# 1. Constants & Schema Definition
# ──────────────────────────────────────────────

REQUIRED_FIELDS = [
    "num_challenges",
    "class_names",
    "sample_ids",
    "transform_types",
    "predicted_classes",
    "confidence_scores",
    "probability_vectors",
]

DEFAULT_NUM_CLASSES = 10
EPSILON = 1e-12  # For numerical stability in KL divergence and cosine calculations


class FingerprintValidationError(ValueError):
    """Raised when a fingerprint fails structure, dimension, or alignment checks."""
    pass


# ──────────────────────────────────────────────
# 2. Configurable Thresholds & Result Structures
# ──────────────────────────────────────────────

@dataclass
class AuthenticationThresholds:
    """Configurable baseline thresholds for the authentication engine.

    Note: These baseline values are defaults for development and baseline testing.
    Final thresholds will be determined experimentally in Phase 4.
    """
    cosine_threshold: float = 0.90       # Cosine similarity (higher is better)
    prediction_threshold: float = 0.85   # Prediction agreement in [0, 1] (higher is better)
    kl_threshold: float = 0.15           # KL divergence >= 0 (lower is better)


@dataclass
class AuthenticationMetrics:
    """Raw computed metrics without decision logic."""
    num_challenges: int
    matching_predictions: int
    mismatching_predictions: int
    prediction_agreement: float
    cosine_similarity: float
    kl_divergence: float
    validation_status: str = "VALID"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuthenticationReport:
    """Full evaluation report including metrics and threshold decision."""
    num_challenges: int
    prediction_agreement: float
    cosine_similarity: float
    kl_divergence: float
    cosine_threshold: float
    prediction_threshold: float
    kl_threshold: float
    cosine_pass: bool
    prediction_pass: bool
    kl_pass: bool
    authenticated: bool
    matching_predictions: int
    mismatching_predictions: int
    validation_status: str
    decision_rule: str = "all_metrics (baseline)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────
# 3. Step 1: Fingerprint Validation
# ──────────────────────────────────────────────

def validate_fingerprint_schema(fp: Dict[str, Any], name: str = "Fingerprint") -> None:
    """Validate internal structure and data types of a single fingerprint."""
    if not isinstance(fp, dict):
        raise FingerprintValidationError(f"{name} must be a dictionary, got {type(fp).__name__}")

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in fp:
            raise FingerprintValidationError(f"{name} missing required field: '{field}'")

    num_challenges = fp["num_challenges"]
    if not isinstance(num_challenges, int) or num_challenges <= 0:
        raise FingerprintValidationError(
            f"{name} 'num_challenges' must be a positive integer, got {num_challenges}"
        )

    # Length consistency checks
    for field in ["sample_ids", "transform_types", "predicted_classes", "confidence_scores", "probability_vectors"]:
        val = fp[field]
        if not hasattr(val, "__len__"):
            raise FingerprintValidationError(f"{name} '{field}' must be a list/sequence")
        if len(val) != num_challenges:
            raise FingerprintValidationError(
                f"{name} '{field}' length ({len(val)}) does not match num_challenges ({num_challenges})"
            )

    class_names = fp["class_names"]
    if not hasattr(class_names, "__len__") or len(class_names) == 0:
        raise FingerprintValidationError(f"{name} 'class_names' must be a non-empty sequence")
    expected_prob_dim = len(class_names)

    # Check probability vector dimensions
    for i, vec in enumerate(fp["probability_vectors"]):
        if not hasattr(vec, "__len__") or len(vec) != expected_prob_dim:
            raise FingerprintValidationError(
                f"{name} probability vector at challenge {i} has dimension "
                f"{len(vec) if hasattr(vec, '__len__') else type(vec)}, expected {expected_prob_dim}"
            )


def validate_fingerprint_alignment(
    ref_fp: Dict[str, Any],
    submitted_fp: Dict[str, Any],
) -> None:
    """Validate that two fingerprints share identical challenge alignment and compatible schemas.

    Fails explicitly if challenge IDs, challenge counts, or transform sequences do not match.
    """
    validate_fingerprint_schema(ref_fp, "Reference fingerprint")
    validate_fingerprint_schema(submitted_fp, "Submitted fingerprint")

    if ref_fp["num_challenges"] != submitted_fp["num_challenges"]:
        raise FingerprintValidationError(
            f"Challenge count mismatch: reference has {ref_fp['num_challenges']}, "
            f"submitted has {submitted_fp['num_challenges']}"
        )

    if ref_fp["class_names"] != submitted_fp["class_names"]:
        raise FingerprintValidationError(
            f"Class names mismatch: reference has {ref_fp['class_names']}, "
            f"submitted has {submitted_fp['class_names']}"
        )

    if list(ref_fp["sample_ids"]) != list(submitted_fp["sample_ids"]):
        raise FingerprintValidationError(
            "Challenge sample_ids do not match in identity or ordering between reference and submitted fingerprints"
        )

    if list(ref_fp["transform_types"]) != list(submitted_fp["transform_types"]):
        raise FingerprintValidationError(
            "Challenge transform_types do not match in sequence between reference and submitted fingerprints"
        )


# ──────────────────────────────────────────────
# 4. Step 2: Prediction Agreement
# ──────────────────────────────────────────────

def calculate_prediction_agreement(
    ref_preds: Union[List[int], np.ndarray],
    sub_preds: Union[List[int], np.ndarray],
) -> Tuple[float, int, int]:
    """Calculate challenge-by-challenge prediction agreement.

    Returns:
        (prediction_agreement, matching_count, mismatching_count)
        where prediction_agreement is in [0.0, 1.0].
    """
    ref_arr = np.asarray(ref_preds, dtype=int)
    sub_arr = np.asarray(sub_preds, dtype=int)

    if len(ref_arr) != len(sub_arr):
        raise FingerprintValidationError("Prediction lists must have identical lengths")
    if len(ref_arr) == 0:
        raise FingerprintValidationError("Cannot compute prediction agreement on empty predictions")

    matches = int(np.sum(ref_arr == sub_arr))
    total = len(ref_arr)
    mismatches = total - matches
    agreement = float(matches / total)

    return agreement, matches, mismatches


# ──────────────────────────────────────────────
# 5. Step 3: Cosine Similarity
# ──────────────────────────────────────────────

def calculate_cosine_similarity(
    ref_probs: Union[List[List[float]], np.ndarray],
    sub_probs: Union[List[List[float]], np.ndarray],
) -> Tuple[float, np.ndarray]:
    """Calculate challenge-level and mean cosine similarity between probability vectors.

    For each challenge vector u (reference) and v (submitted):
        cos(u, v) = (u . v) / (||u||_2 * ||v||_2)

    Returns:
        (mean_cosine_similarity, challenge_level_cosines)
        Higher score indicates higher behavioral similarity (1.0 = identical).
    """
    P = np.asarray(ref_probs, dtype=np.float64)
    Q = np.asarray(sub_probs, dtype=np.float64)

    if P.shape != Q.shape:
        raise FingerprintValidationError(
            f"Probability matrix shape mismatch: {P.shape} vs {Q.shape}"
        )

    # Dot products along class dimension
    dot_products = np.sum(P * Q, axis=1)

    # L2 norms with epsilon clamp for zero-vector safety
    norm_P = np.linalg.norm(P, axis=1)
    norm_Q = np.linalg.norm(Q, axis=1)
    denominators = np.maximum(norm_P * norm_Q, EPSILON)

    challenge_cosines = dot_products / denominators
    # Clamp to [-1.0, 1.0] to prevent floating point overshoot
    challenge_cosines = np.clip(challenge_cosines, -1.0, 1.0)

    mean_cosine = float(np.mean(challenge_cosines))
    return mean_cosine, challenge_cosines


# ──────────────────────────────────────────────
# 6. Step 4: KL Divergence
# ──────────────────────────────────────────────

def calculate_kl_divergence(
    ref_probs: Union[List[List[float]], np.ndarray],
    sub_probs: Union[List[List[float]], np.ndarray],
    epsilon: float = EPSILON,
) -> Tuple[float, np.ndarray]:
    """Calculate challenge-level and mean Kullback-Leibler (KL) divergence.

    Formulation & Direction:
        P = Reference distribution (ground-truth reference behavior)
        Q = Submitted distribution (candidate model behavior)
        D_KL(P || Q) = sum_k [ P(k) * log(P(k) / Q(k)) ]

    Numerical Stability:
        To handle zero probabilities safely without biased infinities or crashes:
        1. Vectors are clipped to a minimum epsilon (1e-12).
        2. Clipped vectors are re-normalized so sum_k P(k) = 1 and sum_k Q(k) = 1.
        3. D_KL is computed using natural logarithm (nats).
        When P == Q, D_KL is identically 0.0.

    Returns:
        (mean_kl_divergence, challenge_level_kl)
        Lower score indicates higher behavioral similarity (0.0 = identical).
    """
    P_raw = np.asarray(ref_probs, dtype=np.float64)
    Q_raw = np.asarray(sub_probs, dtype=np.float64)

    if P_raw.shape != Q_raw.shape:
        raise FingerprintValidationError(
            f"Probability matrix shape mismatch: {P_raw.shape} vs {Q_raw.shape}"
        )

    # Clip for safety and re-normalize across classes
    P_clipped = np.clip(P_raw, epsilon, 1.0)
    Q_clipped = np.clip(Q_raw, epsilon, 1.0)

    P = P_clipped / np.sum(P_clipped, axis=1, keepdims=True)
    Q = Q_clipped / np.sum(Q_clipped, axis=1, keepdims=True)

    # D_KL(P || Q) = sum(P * log(P / Q))
    challenge_kl = np.sum(P * np.log(P / Q), axis=1)
    # Clip negative values resulting from minuscule precision artifacts
    challenge_kl = np.maximum(challenge_kl, 0.0)

    mean_kl = float(np.mean(challenge_kl))
    return mean_kl, challenge_kl


# ──────────────────────────────────────────────
# 7. Step 5 & 6 & 7: Authenticator Class
# ──────────────────────────────────────────────

class ModelAuthenticator:
    """Authentication Engine for comparing deep learning model behavioral fingerprints."""

    def __init__(self, thresholds: Optional[AuthenticationThresholds] = None):
        self.thresholds = thresholds or AuthenticationThresholds()

    def compare_metrics(
        self,
        reference_fingerprint: Dict[str, Any],
        submitted_fingerprint: Dict[str, Any],
    ) -> AuthenticationMetrics:
        """Validate fingerprints and calculate the three independent behavioral metrics.

        Metrics are returned separately without combining into an arbitrary weighted score.
        """
        # Step 1: Validate alignment
        validate_fingerprint_alignment(reference_fingerprint, submitted_fingerprint)

        num_challenges = reference_fingerprint["num_challenges"]

        # Step 2: Prediction Agreement
        agreement, matches, mismatches = calculate_prediction_agreement(
            reference_fingerprint["predicted_classes"],
            submitted_fingerprint["predicted_classes"],
        )

        # Step 3: Cosine Similarity
        mean_cosine, _ = calculate_cosine_similarity(
            reference_fingerprint["probability_vectors"],
            submitted_fingerprint["probability_vectors"],
        )

        # Step 4: KL Divergence
        mean_kl, _ = calculate_kl_divergence(
            reference_fingerprint["probability_vectors"],
            submitted_fingerprint["probability_vectors"],
        )

        return AuthenticationMetrics(
            num_challenges=num_challenges,
            matching_predictions=matches,
            mismatching_predictions=mismatches,
            prediction_agreement=agreement,
            cosine_similarity=mean_cosine,
            kl_divergence=mean_kl,
            validation_status="VALID",
        )

    def evaluate_thresholds(
        self,
        metrics: AuthenticationMetrics,
        thresholds: Optional[AuthenticationThresholds] = None,
    ) -> AuthenticationReport:
        """Apply configurable threshold evaluation to computed metrics.

        Baseline decision rule: 'all_metrics'
            cosine_similarity >= cosine_threshold
            AND prediction_agreement >= prediction_threshold
            AND kl_divergence <= kl_threshold
            -> AUTHENTIC
            Otherwise -> REJECTED
        """
        t = thresholds or self.thresholds

        cosine_pass = bool(metrics.cosine_similarity >= t.cosine_threshold)
        prediction_pass = bool(metrics.prediction_agreement >= t.prediction_threshold)
        kl_pass = bool(metrics.kl_divergence <= t.kl_threshold)

        authenticated = bool(cosine_pass and prediction_pass and kl_pass)

        return AuthenticationReport(
            num_challenges=metrics.num_challenges,
            prediction_agreement=metrics.prediction_agreement,
            cosine_similarity=metrics.cosine_similarity,
            kl_divergence=metrics.kl_divergence,
            cosine_threshold=t.cosine_threshold,
            prediction_threshold=t.prediction_threshold,
            kl_threshold=t.kl_threshold,
            cosine_pass=cosine_pass,
            prediction_pass=prediction_pass,
            kl_pass=kl_pass,
            authenticated=authenticated,
            matching_predictions=metrics.matching_predictions,
            mismatching_predictions=metrics.mismatching_predictions,
            validation_status=metrics.validation_status,
            decision_rule="all_metrics (baseline)",
        )

    def authenticate(
        self,
        reference_fingerprint: Union[Dict[str, Any], str, Path],
        submitted_fingerprint: Union[Dict[str, Any], str, Path],
        thresholds: Optional[AuthenticationThresholds] = None,
    ) -> AuthenticationReport:
        """End-to-end authentication: load/validate fingerprints, compute metrics, and evaluate thresholds."""
        ref_fp = self._resolve_fingerprint(reference_fingerprint)
        sub_fp = self._resolve_fingerprint(submitted_fingerprint)

        metrics = self.compare_metrics(ref_fp, sub_fp)
        return self.evaluate_thresholds(metrics, thresholds)

    @staticmethod
    def _resolve_fingerprint(fp: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
        """Helper to accept either an in-memory dict or a filesystem path to a JSON fingerprint."""
        if isinstance(fp, dict):
            return fp
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"Fingerprint file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


# ──────────────────────────────────────────────
# Convenience Module-Level Functions
# ──────────────────────────────────────────────

def authenticate_fingerprints(
    reference: Union[Dict[str, Any], str, Path],
    submitted: Union[Dict[str, Any], str, Path],
    cosine_threshold: float = 0.90,
    prediction_threshold: float = 0.85,
    kl_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Functional convenience entry point returning a JSON-serializable report dict."""
    thresholds = AuthenticationThresholds(
        cosine_threshold=cosine_threshold,
        prediction_threshold=prediction_threshold,
        kl_threshold=kl_threshold,
    )
    authenticator = ModelAuthenticator(thresholds)
    report = authenticator.authenticate(reference, submitted)
    return report.to_dict()
