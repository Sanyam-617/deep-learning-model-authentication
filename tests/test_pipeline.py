"""
Integration Tests for run_authentication.py Pipeline
===================================================
Tests end-to-end authentication, input validation, multi-model evaluation,
and result export integrity.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import CIFAR10CNN
from run_authentication import (
    authenticate_model,
    get_or_create_reference_fingerprint,
    run_pipeline,
    validate_and_load_candidate_model,
    validate_challenge_set,
)
from src.authentication.authenticator import (
    AuthenticationThresholds,
    ModelAuthenticator,
)


CHALLENGE_PATH = PROJECT_ROOT / "data" / "challenges" / "challenge_set.pt"
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "models" / "original_model.pth"
SUBSTITUTE_MODEL_PATH = PROJECT_ROOT / "models" / "substitute_model.pth"


def test_validate_challenge_set_success():
    """Verify that existing challenge set passes validation."""
    data = validate_challenge_set(CHALLENGE_PATH)
    assert "images" in data
    assert "labels" in data
    assert "sample_ids" in data
    assert "transform_types" in data
    assert data["images"].ndim == 4
    assert data["images"].shape[1] == 3


def test_validate_challenge_set_missing_file():
    """Missing challenge set should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Challenge set not found"):
        validate_challenge_set(Path("non_existent_challenge_set.pt"))


def test_validate_and_load_candidate_model_success():
    """Original model checkpoint should load cleanly in eval mode."""
    device = torch.device("cpu")
    model = validate_and_load_candidate_model(ORIGINAL_MODEL_PATH, device)
    assert isinstance(model, CIFAR10CNN)
    assert not model.training  # Must be in eval mode


def test_validate_and_load_candidate_model_mismatched_architecture():
    """A checkpoint with incorrect layer keys should fail validation."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        dummy_state = {"incompatible_layer.weight": torch.randn(10, 10)}
        torch.save(dummy_state, tmp_path)
        with pytest.raises(ValueError, match="Model architecture mismatch"):
            validate_and_load_candidate_model(tmp_path, torch.device("cpu"))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_end_to_end_pipeline_multi_model():
    """Run full pipeline on both original (authentic) and substitute (rejected) models."""
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ref_fp_path = tmp_path / "reference_fingerprint.pt"
        json_out = tmp_path / "results.json"
        csv_out = tmp_path / "results.csv"

        results = run_pipeline(
            submitted_model_paths=[ORIGINAL_MODEL_PATH, SUBSTITUTE_MODEL_PATH],
            original_model_path=ORIGINAL_MODEL_PATH,
            challenge_path=CHALLENGE_PATH,
            reference_fp_path=ref_fp_path,
            output_dir=tmp_path,
            output_json=json_out,
            output_csv=csv_out,
            batch_size=32,
            device_name="cpu",
        )

        assert len(results) == 2
        res_original, res_substitute = results[0], results[1]

        # 1. Original model must authenticate with 100% agreement, cosine ~ 1.0, KL ~ 0.0
        assert res_original["authenticated"] is True
        assert res_original["decision"] == "AUTHENTIC"
        assert res_original["prediction_agreement"] == 1.0
        assert pytest.approx(res_original["cosine_similarity"], rel=1e-5) == 1.0
        assert pytest.approx(res_original["kl_divergence"], abs=1e-6) == 0.0
        assert res_original["prediction_pass"] is True
        assert res_original["cosine_pass"] is True
        assert res_original["kl_pass"] is True

        # 2. Substitute model with random weights must be REJECTED
        assert res_substitute["authenticated"] is False
        assert res_substitute["decision"] == "REJECTED"
        assert res_substitute["prediction_agreement"] < 0.50
        assert res_substitute["cosine_similarity"] < 0.80
        assert res_substitute["kl_divergence"] > 0.50

        # 3. Output files must exist and be readable
        assert json_out.exists()
        assert csv_out.exists()

        with open(json_out, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert len(saved_json) == 2
        assert saved_json[0]["decision"] == "AUTHENTIC"
        assert saved_json[1]["decision"] == "REJECTED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

