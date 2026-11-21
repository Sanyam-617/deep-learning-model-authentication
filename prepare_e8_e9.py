"""
Prepare E8 (Independently Trained Model) and E9 (Randomly Initialized Model)
===========================================================================
Generates:
  - models/independent_model.pth
  - models/random_model.pth

Safeguards:
  - Never touches models/original_model.pth or reference_fingerprint.pt.
  - Instantiates independent models from scratch.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from train import CIFAR10CNN, get_transforms

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_MODEL_PATH = MODELS_DIR / "original_model.pth"
REF_FP_PATH = DATA_DIR / "fingerprints" / "reference_fingerprint.pt"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def train_independent_model(
    device: torch.device,
    seed: int = 808,
    num_batches: int = 150,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
):
    print(f"\n[+] Training independent model (Seed={seed}, Batches={num_batches})...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_transform, eval_transform = get_transforms()

    full_train = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=False, transform=train_transform
    )

    # Replicate train/val split (seed 42, 10% validation)
    total = len(full_train)
    val_size = int(total * 0.1)
    train_size = total - val_size
    gen = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    # Use evaluation transform for validation set
    val_set_eval = copy.copy(val_set)
    val_set_eval.dataset = copy.copy(full_train)
    val_set_eval.dataset.transform = eval_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set_eval, batch_size=batch_size, shuffle=False)

    model.train()
    running_loss = 0.0
    total_samples = 0
    t0 = time.time()

    for b_idx, (inputs, targets) in enumerate(train_loader):
        if b_idx >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    train_loss = running_loss / total_samples
    elapsed = time.time() - t0

    # Evaluate validation accuracy
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            val_correct += predicted.eq(targets).sum().item()
            val_total += inputs.size(0)

    val_acc = 100.0 * val_correct / val_total
    val_avg_loss = val_loss / val_total

    print(f"    Trained {num_batches} batches ({total_samples} samples) in {elapsed:.2f}s")
    print(f"    Train Loss: {train_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    training_meta = {
        "model_type": "Independently trained CIFAR10CNN",
        "training_seed": seed,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_batches_trained": num_batches,
        "samples_trained": total_samples,
        "training_time_sec": round(elapsed, 3),
        "train_loss": round(train_loss, 4),
        "validation_loss": round(val_avg_loss, 4),
        "validation_accuracy_pct": round(val_acc, 2),
    }

    return model, training_meta


def create_random_model(seed: int = 909):
    print(f"\n[+] Instantiating randomly initialized model (Seed={seed})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CIFAR10CNN()
    model.eval()

    meta = {
        "model_type": "Randomly initialized CIFAR10CNN (untrained)",
        "seed": seed,
        "weights": "Default PyTorch initialization (Kaiming uniform / uniform)",
    }
    return model, meta


def main():
    print("=" * 70)
    print("  PREPARING E8 (Independent) & E9 (Random) MODELS")
    print("=" * 70)

    orig_hash_before = compute_sha256(ORIGINAL_MODEL_PATH)
    ref_fp_hash_before = compute_sha256(REF_FP_PATH)
    print(f"Original Model SHA256 Before:      {orig_hash_before}")
    print(f"Reference Fingerprint SHA256 Before: {ref_fp_hash_before}")

    device = torch.device("cpu")

    # 1. E8: Independent model
    e8_model, e8_meta = train_independent_model(device, seed=808, num_batches=150)
    e8_path = MODELS_DIR / "independent_model.pth"
    torch.save(e8_model.state_dict(), e8_path)
    e8_meta["checkpoint_path"] = str(e8_path)
    e8_meta["sha256"] = compute_sha256(e8_path)
    e8_meta["file_size_bytes"] = e8_path.stat().st_size
    print(f"Saved E8 to: {e8_path.name} (SHA256: {e8_meta['sha256']})")

    # Save E8 training metadata
    e8_meta_path = MODELS_DIR / "independent_model_meta.json"
    with open(e8_meta_path, "w", encoding="utf-8") as f:
        json.dump(e8_meta, f, indent=2)
    print(f"Saved E8 metadata to: {e8_meta_path.name}")

    # 2. E9: Random model
    e9_model, e9_meta = create_random_model(seed=909)
    e9_path = MODELS_DIR / "random_model.pth"
    torch.save(e9_model.state_dict(), e9_path)
    e9_meta["checkpoint_path"] = str(e9_path)
    e9_meta["sha256"] = compute_sha256(e9_path)
    e9_meta["file_size_bytes"] = e9_path.stat().st_size
    print(f"Saved E9 to: {e9_path.name} (SHA256: {e9_meta['sha256']})")

    # Check hashes unchanged
    orig_hash_after = compute_sha256(ORIGINAL_MODEL_PATH)
    ref_fp_hash_after = compute_sha256(REF_FP_PATH)
    assert orig_hash_after == orig_hash_before, "CRITICAL: original_model.pth was altered!"
    assert ref_fp_hash_after == ref_fp_hash_before, "CRITICAL: reference_fingerprint.pt was altered!"
    print("\n[+] Verification: Baseline hashes strictly unchanged!")
    print("=" * 70)


if __name__ == "__main__":
    main()

