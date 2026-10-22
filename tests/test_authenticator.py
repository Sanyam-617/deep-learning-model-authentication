"""
Unit tests for ModelAuthenticator
=================================
Tests validation, metric calculations (prediction agreement, cosine similarity,
KL divergence), numerical stability, and threshold-based decisions.

Run:
    python -m pytest tests/test_authenticator.py -v
"""

import copy
from pathlib import Path
import pytest
import numpy as np

from src.authentication.authenticator import (
    AuthenticationReport,
    AuthenticationThresholds,
    FingerprintValidationError,
    ModelAuthenticator,
    calculate_cosine_similarity,
    calculate_kl_divergence,
    calculate_prediction_agreement,
    validate_fingerprint_alignment,
    validate_fingerprint_schema,
)


# ──────────────────────────────────────────────
# Fixture: Synthetic Minimal Fingerprints
# ──────────────────────────────────────────────

def create_synthetic_fingerprint(
    num_challenges: int = 5,
    seed: int = 42,
    num_classes: int = 10,
    force_zeros: bool = False,
):
    """Generate a valid synthetic fingerprint with reproducible values."""
    rng = np.random.RandomState(seed)

    class_names = [f"class_{i}" for i in range(num_classes)]
    sample_ids = [100 + i for i in range(num_challenges)]
    transform_types = [i % 5 for i in range(num_challenges)]

    # Generate random logits then softmax
    logits = rng.randn(num_challenges, num_classes)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if force_zeros:
        # Zero out certain elements to test sparsity/numerical stability
        probs[:, :3] = 0.0
        probs = probs / np.sum(probs, axis=1, keepdims=True)

    preds = np.argmax(probs, axis=1).tolist()
    confs = np.max(probs, axis=1).tolist()
    prob_vectors = probs.tolist()

    return {
        "num_challenges": num_challenges,
        "class_names": class_names,
        "sample_ids": sample_ids,
        "transform_types": transform_types,
        "predicted_classes": preds,
        "confidence_scores": confs,
        "probability_vectors": prob_vectors,
    }


# ──────────────────────────────────────────────
# 1. Identical Fingerprints
# ──────────────────────────────────────────────

def test_identical_fingerprints():
    """Identical fingerprints must yield 100% agreement, cosine ~ 1.0, and KL ~ 0.0."""
    fp1 = create_synthetic_fingerprint(num_challenges=8, seed=123)
    fp2 = copy.deepcopy(fp1)

    authenticator = ModelAuthenticator()
    report = authenticator.authenticate(fp1, fp2)

    assert report.validation_status == "VALID"
    assert report.num_challenges == 8
    assert report.matching_predictions == 8
    assert report.mismatching_predictions == 0
    assert report.prediction_agreement == 1.0
    assert pytest.approx(report.cosine_similarity, rel=1e-5) == 1.0
    assert pytest.approx(report.kl_divergence, abs=1e-7) == 0.0
    assert report.authenticated is True


# ──────────────────────────────────────────────
# 2. Different Fingerprints
# ──────────────────────────────────────────────

def test_different_fingerprints():
    """Different models must show lower agreement, lower cosine, and higher KL."""
    fp_ref = create_synthetic_fingerprint(num_challenges=10, seed=42)
    fp_sub = create_synthetic_fingerprint(num_challenges=10, seed=999)

    authenticator = ModelAuthenticator()
    report = authenticator.authenticate(fp_ref, fp_sub)

    assert report.prediction_agreement < 1.0
    assert report.cosine_similarity < 1.0
    assert report.kl_divergence > 0.01


# ──────────────────────────────────────────────
# 3. Mismatched sample_ids Validation Failure
# ──────────────────────────────────────────────

def test_mismatched_sample_ids_fails():
    """Mismatched or reordered sample_ids must fail validation."""
    fp1 = create_synthetic_fingerprint(num_challenges=5, seed=1)
    fp2 = copy.deepcopy(fp1)
    fp2["sample_ids"] = [999, 101, 102, 103, 104]  # Alter first sample ID

    authenticator = ModelAuthenticator()
    with pytest.raises(FingerprintValidationError, match="sample_ids do not match"):
        authenticator.authenticate(fp1, fp2)


# ──────────────────────────────────────────────
# 4. Mismatched transform_types Validation Failure
# ──────────────────────────────────────────────

def test_mismatched_transform_types_fails():
    """Mismatched transform sequence must fail validation."""
    fp1 = create_synthetic_fingerprint(num_challenges=5, seed=1)
    fp2 = copy.deepcopy(fp1)
    fp2["transform_types"][0] = 4 if fp1["transform_types"][0] != 4 else 0

    authenticator = ModelAuthenticator()
    with pytest.raises(FingerprintValidationError, match="transform_types do not match"):
        authenticator.authenticate(fp1, fp2)


# ──────────────────────────────────────────────
# 5. Invalid Probability-Vector Dimensions Validation Failure
# ──────────────────────────────────────────────

def test_invalid_probability_vector_dimensions_fails():
    """Probability vectors with incorrect dimension (e.g. 9 instead of 10) must fail validation."""
    fp = create_synthetic_fingerprint(num_challenges=5, seed=1, num_classes=10)
    fp["probability_vectors"][2] = [0.1] * 9  # 9 elements instead of 10

    with pytest.raises(FingerprintValidationError, match="expected 10"):
        validate_fingerprint_schema(fp)


# ──────────────────────────────────────────────
# 6. Missing Required Field Validation Failure
# ──────────────────────────────────────────────

@pytest.mark.parametrize("missing_key", [
    "num_challenges",
    "class_names",
    "sample_ids",
    "transform_types",
    "predicted_classes",
    "confidence_scores",
    "probability_vectors",
])
def test_missing_required_field_fails(missing_key):
    """Removing any required field must raise FingerprintValidationError."""
    fp = create_synthetic_fingerprint(num_challenges=4, seed=1)
    del fp[missing_key]

    with pytest.raises(FingerprintValidationError, match=f"missing required field: '{missing_key}'"):
        validate_fingerprint_schema(fp)


# ──────────────────────────────────────────────
# 7. Distributions Containing Zeros: Numerical Stability
# ──────────────────────────────────────────────

def test_numerical_stability_with_exact_zeros():
    """Distributions containing exact zeros must not divide-by-zero or produce NaN/inf."""
    fp1 = create_synthetic_fingerprint(num_challenges=6, seed=10, force_zeros=True)
    fp2 = copy.deepcopy(fp1)

    mean_cosine, cosines = calculate_cosine_similarity(
        fp1["probability_vectors"],
        fp2["probability_vectors"],
    )
    mean_kl, kls = calculate_kl_divergence(
        fp1["probability_vectors"],
        fp2["probability_vectors"],
    )

    assert not np.isnan(mean_cosine)
    assert not np.isinf(mean_cosine)
    assert not np.isnan(mean_kl)
    assert not np.isinf(mean_kl)

    # Identical with zeros should still yield 1.0 cosine and 0.0 KL
    assert pytest.approx(mean_cosine, rel=1e-5) == 1.0
    assert pytest.approx(mean_kl, abs=1e-7) == 0.0


# ──────────────────────────────────────────────
# 8. Configurable Threshold Pass/Fail Behavior
# ──────────────────────────────────────────────

def test_threshold_pass_fail_behavior():
    """Verify that configurable thresholds correctly determine pass/fail flags and decision."""
    fp_ref = create_synthetic_fingerprint(num_challenges=10, seed=42)
    fp_sub = create_synthetic_fingerprint(num_challenges=10, seed=99)

    # Case A: Strict thresholds where sub fails
    strict_thresholds = AuthenticationThresholds(
        cosine_threshold=0.999,
        prediction_threshold=0.99,
        kl_threshold=0.001,
    )
    auth_strict = ModelAuthenticator(strict_thresholds)
    report_strict = auth_strict.authenticate(fp_ref, fp_sub)

    assert report_strict.authenticated is False
    assert (
        report_strict.cosine_pass is False
        or report_strict.prediction_pass is False
        or report_strict.kl_pass is False
    )

    # Case B: Permissive thresholds where sub passes
    permissive_thresholds = AuthenticationThresholds(
        cosine_threshold=0.10,
        prediction_threshold=0.10,
        kl_threshold=10.0,
    )
    auth_permissive = ModelAuthenticator(permissive_thresholds)
    report_permissive = auth_permissive.authenticate(fp_ref, fp_sub)

    assert report_permissive.cosine_pass is True
    assert report_permissive.prediction_pass is True
    assert report_permissive.kl_pass is True
    assert report_permissive.authenticated is True


# ──────────────────────────────────────────────
# 9. Main runner for CLI execution
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
