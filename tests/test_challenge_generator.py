"""
Tests for challenge_generator.py
=================================
Basic checks for tensor dimensions, reproducibility, label consistency,
and output file existence.

Run from the project root:
    python -m pytest tests/test_challenge_generator.py -v

These tests do NOT run model training.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

# ──────────────────────────────────────────────
# Ensure the project root is importable
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.challenges.challenge_generator import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGE_SIZE,
    NUM_CLASSES,
    SPLIT_SEED,
    TRANSFORM_NAMES,
    VALIDATION_SPLIT,
    generate_challenges,
    get_validation_indices,
    normalize,
    add_gaussian_noise,
    apply_rotation,
    adjust_brightness,
    random_crop_resize,
)


# ──────────────────────────────────────────────
# 1.  Validation split consistency
# ──────────────────────────────────────────────

def test_validation_indices_are_deterministic():
    """Calling get_validation_indices twice must yield identical lists."""
    idx_a = get_validation_indices()
    idx_b = get_validation_indices()
    assert idx_a == idx_b, "Validation indices changed between calls"


def test_validation_split_size():
    """Validation set should be 10 % of 50 000 = 5 000 images."""
    idx = get_validation_indices()
    expected = int(50_000 * VALIDATION_SPLIT)
    assert len(idx) == expected, (
        f"Expected {expected} validation indices, got {len(idx)}"
    )


def test_validation_indices_within_range():
    """All indices must be in [0, 49999]."""
    idx = get_validation_indices()
    assert all(0 <= i < 50_000 for i in idx)


# ──────────────────────────────────────────────
# 2.  Individual transform checks
# ──────────────────────────────────────────────

def _dummy_image() -> torch.Tensor:
    """Return a deterministic (3, 32, 32) tensor in [0, 1]."""
    gen = torch.Generator().manual_seed(0)
    return torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=gen)


def test_gaussian_noise_shape_and_range():
    img = _dummy_image()
    noisy = add_gaussian_noise(img, std=0.05, seed=123)
    assert noisy.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0


def test_gaussian_noise_reproducibility():
    img = _dummy_image()
    a = add_gaussian_noise(img, std=0.05, seed=99)
    b = add_gaussian_noise(img, std=0.05, seed=99)
    assert torch.equal(a, b), "Gaussian noise not reproducible with same seed"


def test_rotation_shape():
    img = _dummy_image()
    rotated = apply_rotation(img, angle=15.0)
    assert rotated.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_brightness_shape_and_range():
    img = _dummy_image()
    bright = adjust_brightness(img, factor=1.3)
    assert bright.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert bright.min() >= 0.0 and bright.max() <= 1.0


def test_crop_resize_shape():
    img = _dummy_image()
    cropped = random_crop_resize(img, crop_size=24, output_size=IMAGE_SIZE, seed=42)
    assert cropped.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_crop_resize_reproducibility():
    img = _dummy_image()
    a = random_crop_resize(img, crop_size=24, seed=42)
    b = random_crop_resize(img, crop_size=24, seed=42)
    assert torch.equal(a, b), "Crop-resize not reproducible with same seed"


def test_normalize_shape():
    img = _dummy_image()
    normed = normalize(img)
    assert normed.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


# ──────────────────────────────────────────────
# 3.  End-to-end generation
# ──────────────────────────────────────────────

def _run_generation(output_dir: str, num_per_class: int = 2, seed: int = 42):
    """Run the generator with minimal settings and return the result dict."""
    return generate_challenges(
        data_dir=os.path.join(".", "data"),
        output_dir=output_dir,
        num_per_class=num_per_class,
        seed=seed,
    )


def test_output_tensor_shapes():
    """Generated tensors should have correct shapes and dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _run_generation(tmpdir, num_per_class=2)

        num_originals = 2 * NUM_CLASSES
        num_transforms = len(TRANSFORM_NAMES)
        expected_total = num_originals * num_transforms

        assert data["images"].shape == (expected_total, 3, IMAGE_SIZE, IMAGE_SIZE), \
            f"Images shape mismatch: {data['images'].shape}"
        assert data["labels"].shape == (expected_total,)
        assert data["sample_ids"].shape == (expected_total,)
        assert data["transform_types"].shape == (expected_total,)
        assert data["images"].dtype == torch.float32
        assert data["labels"].dtype == torch.long


def test_label_balance():
    """Each class should have equal representation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _run_generation(tmpdir, num_per_class=2)
        labels = data["labels"]
        num_transforms = len(TRANSFORM_NAMES)
        for cls in range(NUM_CLASSES):
            count = (labels == cls).sum().item()
            # Each original image produces num_transforms variants
            assert count == 2 * num_transforms, (
                f"Class {cls}: expected {2 * num_transforms} samples, got {count}"
            )


def test_output_files_exist():
    """challenge_set.pt and metadata.json must be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_generation(tmpdir, num_per_class=2)
        assert (Path(tmpdir) / "challenge_set.pt").exists()
        assert (Path(tmpdir) / "metadata.json").exists()


def test_metadata_contents():
    """Metadata JSON should contain expected keys and values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_generation(tmpdir, num_per_class=2, seed=42)
        with open(Path(tmpdir) / "metadata.json") as f:
            meta = json.load(f)

        assert meta["dataset"]["name"] == "CIFAR-10"
        assert meta["dataset"]["split_seed"] == SPLIT_SEED
        assert meta["normalization"]["mean"] == list(CIFAR10_MEAN)
        assert meta["normalization"]["std"] == list(CIFAR10_STD)
        assert meta["selection"]["num_per_class"] == 2
        assert meta["master_seed"] == 42
        assert meta["image_shape"] == [3, IMAGE_SIZE, IMAGE_SIZE]


def test_reproducibility_across_runs():
    """Two runs with the same seed must produce bit-identical tensors."""
    with tempfile.TemporaryDirectory() as tmpdir_a, \
         tempfile.TemporaryDirectory() as tmpdir_b:
        data_a = _run_generation(tmpdir_a, num_per_class=2, seed=42)
        data_b = _run_generation(tmpdir_b, num_per_class=2, seed=42)

        assert torch.equal(data_a["images"], data_b["images"]), \
            "Images differ between runs with same seed"
        assert torch.equal(data_a["labels"], data_b["labels"]), \
            "Labels differ between runs with same seed"
        assert torch.equal(data_a["sample_ids"], data_b["sample_ids"]), \
            "Sample IDs differ between runs with same seed"
        assert torch.equal(data_a["transform_types"], data_b["transform_types"]), \
            "Transform types differ between runs with same seed"


def test_transform_types_coverage():
    """Every original should produce one variant per transform type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _run_generation(tmpdir, num_per_class=2)
        t_types = data["transform_types"]
        for t_id in TRANSFORM_NAMES:
            count = (t_types == t_id).sum().item()
            # 2 per class × 10 classes = 20 originals → 20 of each type
            assert count == 2 * NUM_CLASSES, (
                f"Transform '{TRANSFORM_NAMES[t_id]}': "
                f"expected {2 * NUM_CLASSES}, got {count}"
            )


# ──────────────────────────────────────────────
# 4.  Run with pytest or directly
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
