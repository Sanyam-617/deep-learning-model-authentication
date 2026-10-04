"""
Challenge Set Generator
=======================
Deep Learning Model Authentication Using Behavioral Challenge–Response

Generates a reproducible, balanced challenge set from the CIFAR-10
validation split.  Each challenge consists of the original image plus
controlled transformations (Gaussian noise, rotation, brightness,
random crop + resize).  Transformations are applied *before*
normalization so the fingerprint extractor receives data in the same
normalized space used during training.

Output
------
data/challenges/challenge_set.pt   – dict with tensors (images, labels,
                                      sample_ids, transform_types)
data/challenges/metadata.json       – human-readable provenance info

Usage
-----
    python -m src.challenges.challenge_generator           # defaults
    python -m src.challenges.challenge_generator --num_per_class 5 --seed 42
    python -m src.challenges.challenge_generator --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import random_split


# ──────────────────────────────────────────────
# 1.  Constants — must match train.py exactly
# ──────────────────────────────────────────────

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

VALIDATION_SPLIT: float = 0.1      # same as train.py CONFIG["validation_split"]
SPLIT_SEED: int = 42               # same as train.py CONFIG["random_seed"]
TOTAL_TRAIN_SAMPLES: int = 50_000  # CIFAR-10 training set size

NUM_CLASSES: int = 10
IMAGE_SIZE: int = 32
CLASS_NAMES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Default output directory
DEFAULT_OUTPUT_DIR: str = os.path.join("data", "challenges")
DEFAULT_DATA_DIR: str = os.path.join(".", "data")

# Transformation type labels (stored alongside each image)
TRANSFORM_ORIGINAL: int = 0
TRANSFORM_GAUSSIAN_NOISE: int = 1
TRANSFORM_ROTATION: int = 2
TRANSFORM_BRIGHTNESS: int = 3
TRANSFORM_CROP_RESIZE: int = 4

TRANSFORM_NAMES: Dict[int, str] = {
    TRANSFORM_ORIGINAL: "original",
    TRANSFORM_GAUSSIAN_NOISE: "gaussian_noise",
    TRANSFORM_ROTATION: "rotation",
    TRANSFORM_BRIGHTNESS: "brightness",
    TRANSFORM_CROP_RESIZE: "crop_resize",
}


# ──────────────────────────────────────────────
# 2.  Validation split reproduction
# ──────────────────────────────────────────────

def get_validation_indices(
    total: int = TOTAL_TRAIN_SAMPLES,
    val_fraction: float = VALIDATION_SPLIT,
    seed: int = SPLIT_SEED,
) -> List[int]:
    """Reproduce the validation indices from ``train.py``'s ``random_split``.

    Uses the same ``torch.Generator`` seed and split sizes so that the
    challenge set draws from exactly the same validation pool.

    Parameters
    ----------
    total : int
        Total number of training samples (50 000 for CIFAR-10).
    val_fraction : float
        Fraction reserved for validation (0.1 → 5 000 images).
    seed : int
        Random seed for the generator.

    Returns
    -------
    list[int]
        Sorted list of dataset indices belonging to the validation set.
    """
    val_size = int(total * val_fraction)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    # random_split internally calls randperm; we replicate that to get indices
    full_dataset_placeholder = list(range(total))
    train_subset, val_subset = random_split(
        full_dataset_placeholder,
        [train_size, val_size],
        generator=generator,
    )
    val_indices = sorted(val_subset.indices)
    return val_indices


def select_balanced_samples(
    dataset: torchvision.datasets.CIFAR10,
    val_indices: List[int],
    num_per_class: int,
    seed: int,
) -> List[int]:
    """Select a balanced subset of validation images (equal per class).

    Parameters
    ----------
    dataset : torchvision.datasets.CIFAR10
        The full CIFAR-10 training dataset (not split).
    val_indices : list[int]
        Indices that belong to the validation split.
    num_per_class : int
        Number of images to select from each of the 10 classes.
    seed : int
        RNG seed for reproducible selection.

    Returns
    -------
    list[int]
        Selected indices into the *full* training dataset, sorted.

    Raises
    ------
    ValueError
        If any class has fewer validation samples than ``num_per_class``.
    """
    rng = np.random.RandomState(seed)

    # Group validation indices by class label
    class_buckets: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
    for idx in val_indices:
        _, label = dataset[idx]
        class_buckets[label].append(idx)

    selected: List[int] = []
    for cls in range(NUM_CLASSES):
        available = class_buckets[cls]
        if len(available) < num_per_class:
            raise ValueError(
                f"Class '{CLASS_NAMES[cls]}' has only {len(available)} "
                f"validation samples, but {num_per_class} were requested."
            )
        chosen = rng.choice(available, size=num_per_class, replace=False)
        selected.extend(chosen.tolist())

    return sorted(selected)


# ──────────────────────────────────────────────
# 3.  Transformation functions
# ──────────────────────────────────────────────
# All functions operate on a [C, H, W] float tensor in [0, 1] range
# (i.e. after ToTensor but BEFORE Normalize).

def add_gaussian_noise(
    image: torch.Tensor,
    std: float = 0.05,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Add Gaussian noise to an image tensor.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W) in [0, 1].
    std : float
        Standard deviation of the Gaussian noise.
    seed : int, optional
        If given, seeds the generator for reproducibility.

    Returns
    -------
    torch.Tensor
        Noisy image clamped to [0, 1].
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    noise = torch.randn(image.shape, generator=gen) * std
    return torch.clamp(image + noise, 0.0, 1.0)


def apply_rotation(
    image: torch.Tensor,
    angle: float = 15.0,
) -> torch.Tensor:
    """Rotate an image by a fixed angle (degrees).

    Parameters
    ----------
    image : torch.Tensor
        Image tensor (C, H, W).
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    torch.Tensor
        Rotated image.
    """
    return TF.rotate(image, angle=angle)


def adjust_brightness(
    image: torch.Tensor,
    factor: float = 1.3,
) -> torch.Tensor:
    """Adjust image brightness by a multiplicative factor.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor (C, H, W) in [0, 1].
    factor : float
        Brightness factor (>1 brightens, <1 darkens).

    Returns
    -------
    torch.Tensor
        Brightness-adjusted image clamped to [0, 1].
    """
    return torch.clamp(TF.adjust_brightness(image, factor), 0.0, 1.0)


def random_crop_resize(
    image: torch.Tensor,
    crop_size: int = 24,
    output_size: int = IMAGE_SIZE,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Randomly crop and resize back to original dimensions.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor (C, H, W).
    crop_size : int
        Size of the square crop (< original size).
    output_size : int
        Target output size after resizing.
    seed : int, optional
        Seeds the crop position for reproducibility.

    Returns
    -------
    torch.Tensor
        Cropped-and-resized image.
    """
    _, h, w = image.shape
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    top = rng.randint(0, h - crop_size + 1)
    left = rng.randint(0, w - crop_size + 1)
    cropped = TF.crop(image, top, left, crop_size, crop_size)
    resized = TF.resize(cropped, [output_size, output_size], antialias=True)
    return resized


# ──────────────────────────────────────────────
# 4.  Normalization helper
# ──────────────────────────────────────────────

def normalize(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """Apply channel-wise normalization.

    Parameters
    ----------
    image : torch.Tensor
        Image (C, H, W) in [0, 1].
    mean, std : tuple[float, float, float]
        Per-channel mean and std.

    Returns
    -------
    torch.Tensor
        Normalized image.
    """
    return TF.normalize(image, mean=list(mean), std=list(std))


# ──────────────────────────────────────────────
# 5.  Core generator
# ──────────────────────────────────────────────

def generate_challenges(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_per_class: int = 10,
    seed: int = 42,
    noise_std: float = 0.05,
    rotation_angle: float = 15.0,
    brightness_factor: float = 1.3,
    crop_size: int = 24,
) -> Dict:
    """Generate a reproducible challenge set from CIFAR-10 validation data.

    The pipeline:
      1. Load the raw CIFAR-10 training set (no augmentation).
      2. Reproduce the validation split from ``train.py``.
      3. Select a balanced subset (``num_per_class`` per class).
      4. For each selected image, produce 5 variants:
         original, noisy, rotated, brightness-adjusted, crop+resize.
      5. Normalize all images with the same stats used in training.
      6. Save tensors and metadata to ``output_dir``.

    Parameters
    ----------
    data_dir : str
        Root directory containing (or to download) CIFAR-10.
    output_dir : str
        Directory where outputs will be written.
    num_per_class : int
        Number of validation images per class.
    seed : int
        Master seed for reproducibility.
    noise_std : float
        Std for Gaussian noise transform.
    rotation_angle : float
        Rotation angle (degrees).
    brightness_factor : float
        Brightness multiplier.
    crop_size : int
        Crop side length (pixels) before resizing to 32×32.

    Returns
    -------
    dict
        The saved challenge-set dictionary (images, labels, etc.).
    """
    print("=" * 56)
    print("  Challenge Set Generator")
    print("  Deep Learning Model Authentication (Challenge-Response)")
    print("=" * 56)

    # ── Step 1: load raw dataset (ToTensor only, NO augmentation) ──
    print("\n> Loading CIFAR-10 (raw) ...")
    raw_transform = T.ToTensor()  # -> float32, [0, 1]
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=raw_transform,
    )
    print(f"  Total training samples: {len(full_train):,}")

    # ── Step 2: reproduce validation indices ──
    print("\n> Reproducing validation split (seed={}, frac={}) ...".format(
        SPLIT_SEED, VALIDATION_SPLIT
    ))
    val_indices = get_validation_indices()
    print(f"  Validation set size: {len(val_indices):,}")

    # ── Step 3: balanced selection ──
    print(f"\n> Selecting {num_per_class} images per class (seed={seed}) ...")
    selected_indices = select_balanced_samples(
        full_train, val_indices, num_per_class, seed=seed,
    )
    total_originals = len(selected_indices)
    print(f"  Selected {total_originals} images across {NUM_CLASSES} classes")

    # ── Step 4: generate transformed variants ──
    num_transforms = len(TRANSFORM_NAMES)  # 5 (including original)
    total_challenges = total_originals * num_transforms

    print(f"\n> Generating {total_challenges} challenge images "
          f"({total_originals} x {num_transforms} transforms) ...")

    all_images: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_sample_ids: List[int] = []
    all_transform_types: List[int] = []

    for i, dataset_idx in enumerate(selected_indices):
        raw_img, label = full_train[dataset_idx]
        # raw_img: (3, 32, 32) float32 in [0, 1]

        # Per-image sub-seed ensures reproducibility per sample
        img_seed = seed * 100_000 + dataset_idx

        variants = [
            (TRANSFORM_ORIGINAL, raw_img),
            (TRANSFORM_GAUSSIAN_NOISE,
             add_gaussian_noise(raw_img, std=noise_std, seed=img_seed + 1)),
            (TRANSFORM_ROTATION,
             apply_rotation(raw_img, angle=rotation_angle)),
            (TRANSFORM_BRIGHTNESS,
             adjust_brightness(raw_img, factor=brightness_factor)),
            (TRANSFORM_CROP_RESIZE,
             random_crop_resize(raw_img, crop_size=crop_size, seed=img_seed + 4)),
        ]

        for t_type, t_img in variants:
            normalised = normalize(t_img)
            all_images.append(normalised)
            all_labels.append(label)
            all_sample_ids.append(dataset_idx)
            all_transform_types.append(t_type)

    # Stack into tensors
    images_tensor = torch.stack(all_images)          # (N, 3, 32, 32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)         # (N,)
    sample_ids_tensor = torch.tensor(all_sample_ids, dtype=torch.long)  # (N,)
    transform_types_tensor = torch.tensor(all_transform_types, dtype=torch.long)  # (N,)

    print(f"  images          : {tuple(images_tensor.shape)}")
    print(f"  labels          : {tuple(labels_tensor.shape)}")
    print(f"  sample_ids      : {tuple(sample_ids_tensor.shape)}")
    print(f"  transform_types : {tuple(transform_types_tensor.shape)}")

    # ── Step 5: save ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    challenge_data = {
        "images": images_tensor,
        "labels": labels_tensor,
        "sample_ids": sample_ids_tensor,
        "transform_types": transform_types_tensor,
    }

    tensor_file = output_path / "challenge_set.pt"
    torch.save(challenge_data, tensor_file)
    print(f"\n> Saved tensor data  -> {tensor_file}")

    # Metadata
    metadata = {
        "description": (
            "Challenge set for deep-learning model authentication. "
            "Contains original and transformed CIFAR-10 validation images."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "CIFAR-10",
            "split": "validation (from training set)",
            "split_seed": SPLIT_SEED,
            "validation_fraction": VALIDATION_SPLIT,
            "total_train_samples": TOTAL_TRAIN_SAMPLES,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
        },
        "selection": {
            "num_per_class": num_per_class,
            "total_originals": total_originals,
            "selection_seed": seed,
        },
        "transforms": {
            "types": TRANSFORM_NAMES,
            "num_per_original": num_transforms,
            "total_challenges": total_challenges,
            "parameters": {
                "gaussian_noise_std": noise_std,
                "rotation_angle_deg": rotation_angle,
                "brightness_factor": brightness_factor,
                "crop_size": crop_size,
                "output_size": IMAGE_SIZE,
            },
        },
        "normalization": {
            "mean": list(CIFAR10_MEAN),
            "std": list(CIFAR10_STD),
        },
        "image_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "output_files": {
            "tensor_file": str(tensor_file),
            "metadata_file": str(output_path / "metadata.json"),
        },
        "master_seed": seed,
    }

    meta_file = output_path / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata     -> {meta_file}")

    print(f"\n[OK] Challenge set generation complete "
          f"({total_challenges} images).\n")

    return challenge_data


# ──────────────────────────────────────────────
# 6.  CLI entry point
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the challenge generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reproducible challenge set from CIFAR-10 "
            "validation images for deep-learning model authentication."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help="Root directory for CIFAR-10 data (default: %(default)s).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for challenge files (default: %(default)s).",
    )
    parser.add_argument(
        "--num_per_class", type=int, default=10,
        help="Number of validation images per class (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed for reproducible selection and transforms (default: %(default)s).",
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.05,
        help="Std dev for Gaussian noise (default: %(default)s).",
    )
    parser.add_argument(
        "--rotation_angle", type=float, default=15.0,
        help="Rotation angle in degrees (default: %(default)s).",
    )
    parser.add_argument(
        "--brightness_factor", type=float, default=1.3,
        help="Brightness multiplier (default: %(default)s).",
    )
    parser.add_argument(
        "--crop_size", type=int, default=24,
        help="Square crop size before resizing to 32x32 (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    generate_challenges(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_per_class=args.num_per_class,
        seed=args.seed,
        noise_std=args.noise_std,
        rotation_angle=args.rotation_angle,
        brightness_factor=args.brightness_factor,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
