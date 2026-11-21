"""
Unit tests for Module 2: Behavioral Fingerprint Extraction
==========================================================
Tests model loading, eval mode, fingerprint extraction shapes,
probability distribution validity, and reproducibility verification.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import CIFAR10CNN
from src.fingerprint.fingerprint import (
    DEFAULT_CHALLENGE_PATH,
    DEFAULT_MODEL_PATH,
    extract_fingerprint,
    load_model,
    save_fingerprint,
    verify_reproducibility,
)


def test_load_model_eval_mode():
    """Verify that load_model loads weights and puts CNN in eval mode."""
    device = torch.device("cpu")
    model = load_model(DEFAULT_MODEL_PATH, device)
    assert isinstance(model, CIFAR10CNN)
    assert not model.training


def test_extract_fingerprint_shapes_and_values():
    """Verify extracted fingerprint keys, tensor shapes, and probability axioms."""
    device = torch.device("cpu")
    model = load_model(DEFAULT_MODEL_PATH, device)
    fp = extract_fingerprint(
        model=model,
        challenge_path=str(DEFAULT_CHALLENGE_PATH),
        device=device,
        batch_size=32,
    )

    num_challenges = fp["num_challenges"]
    assert num_challenges > 0
    assert fp["predicted_classes"].shape == (num_challenges,)
    assert fp["confidence_scores"].shape == (num_challenges,)
    assert fp["probability_vectors"].shape == (num_challenges, 10)
    assert fp["sample_ids"].shape == (num_challenges,)
    assert fp["transform_types"].shape == (num_challenges,)

    # Probabilities must be in [0, 1] and sum to 1.0 along dim=1
    probs = fp["probability_vectors"]
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    row_sums = torch.sum(probs, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Max probability should equal confidence score
    max_probs = torch.max(probs, dim=1).values
    assert torch.allclose(max_probs, fp["confidence_scores"], atol=1e-5)


def test_verify_reproducibility():
    """Verify that two identical extractions pass reproducibility check."""
    device = torch.device("cpu")
    model = load_model(DEFAULT_MODEL_PATH, device)
    fp1 = extract_fingerprint(model, str(DEFAULT_CHALLENGE_PATH), device, batch_size=32)
    fp2 = extract_fingerprint(model, str(DEFAULT_CHALLENGE_PATH), device, batch_size=32)
    assert verify_reproducibility(fp1, fp2) is True


def test_save_and_load_fingerprint():
    """Verify that save_fingerprint persists to disk and can be reloaded."""
    device = torch.device("cpu")
    model = load_model(DEFAULT_MODEL_PATH, device)
    fp = extract_fingerprint(model, str(DEFAULT_CHALLENGE_PATH), device, batch_size=32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test_ref_fp.pt"
        save_fingerprint(fp, out_file)
        assert out_file.exists()

        loaded = torch.load(out_file, map_location="cpu", weights_only=False)
        assert verify_reproducibility(fp, loaded) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

