"""
Phase 1 – Train a CNN on CIFAR-10
=================================
Deep Learning Model Authentication Using Behavioral Challenge-Response

This script:
  • Downloads / loads CIFAR-10
  • Applies data augmentation + normalization
  • Trains a lightweight custom CNN for 10 epochs
  • Tracks training loss, validation loss, and validation accuracy
  • Saves the best model weights to  models/original_model.pth
  • Saves model configuration to     models/model_config.json

Designed to run on a regular laptop CPU or a Colab GPU.
"""

import os
import json
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend (safe for servers / Colab)
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 1.  Constants & Configuration
# ──────────────────────────────────────────────

# CIFAR-10 class names (official order)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Hyper-parameters (easy to override in future phases)
CONFIG = {
    "dataset": "CIFAR-10",
    "num_classes": 10,
    "image_size": 32,
    "in_channels": 3,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 10,
    "validation_split": 0.1,          # 10 % of training data for validation
    "random_seed": 42,
    "model_save_dir": "models",
    "model_filename": "original_model.pth",
    "config_filename": "model_config.json",
    "data_dir": "./data",
}


# ──────────────────────────────────────────────
# 2.  Data Loading & Transforms
# ──────────────────────────────────────────────

def get_transforms():
    """Return training and evaluation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])
    return train_transform, eval_transform


def load_datasets(config: dict):
    """Download CIFAR-10 and split into train / val / test sets."""
    train_transform, eval_transform = get_transforms()

    full_train_set = torchvision.datasets.CIFAR10(
        root=config["data_dir"], train=True,
        download=True, transform=train_transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=config["data_dir"], train=False,
        download=True, transform=eval_transform,
    )

    # Split training data into train + validation
    total = len(full_train_set)
    val_size = int(total * config["validation_split"])
    train_size = total - val_size

    generator = torch.Generator().manual_seed(config["random_seed"])
    train_set, val_set = random_split(full_train_set, [train_size, val_size], generator=generator)

    # Validation set should use the evaluation (no-augmentation) transform.
    # We wrap it with a simple helper.
    val_set.dataset = copy.copy(full_train_set)
    val_set.dataset.transform = eval_transform

    print(f"  Training samples   : {train_size:,}")
    print(f"  Validation samples : {val_size:,}")
    print(f"  Test samples       : {len(test_set):,}")

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 3.  Model Architecture
# ──────────────────────────────────────────────

class CIFAR10CNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10.

    Architecture
    ────────────
    Conv Block 1 :  Conv2d(3→32, 3×3) → ReLU → Conv2d(32→32, 3×3) → ReLU → MaxPool(2) → Dropout(0.25)
    Conv Block 2 :  Conv2d(32→64, 3×3) → ReLU → Conv2d(64→64, 3×3) → ReLU → MaxPool(2) → Dropout(0.25)
    Classifier   :  Flatten → FC(64*5*5 → 256) → ReLU → Dropout(0.5) → FC(256 → 10)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(config: dict, device: torch.device) -> CIFAR10CNN:
    """Instantiate the CNN and move it to *device*."""
    model = CIFAR10CNN(
        num_classes=config["num_classes"],
        in_channels=config["in_channels"],
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable:,}")
    return model


# ──────────────────────────────────────────────
# 4.  Training & Validation Loops
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset. Returns (avg_loss, accuracy %)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, config, device):
    """
    Full training loop.

    Returns
    -------
    history : dict   –  lists of train_loss, val_loss, val_acc per epoch
    best_state : dict – state_dict of the best model (by val accuracy)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_state = None

    epochs = config["epochs"]
    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}")
    print("─" * 52)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            marker = " ★"

        print(f"  {epoch:>3}/{epochs}   {train_loss:>10.4f}   {val_loss:>8.4f}   {val_acc:>6.2f}%  {elapsed:>5.1f}s{marker}")

    print(f"\n  Best validation accuracy: {best_acc:.2f}%")
    return history, best_state


# ──────────────────────────────────────────────
# 5.  Plotting
# ──────────────────────────────────────────────

def plot_history(history: dict, save_path: str = "training_curves.png"):
    """Plot training/validation loss and validation accuracy."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", label="Training Loss", color="#4e79a7")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Validation Loss", color="#e15759")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["val_acc"], "^-", label="Validation Accuracy", color="#59a14f")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to: {save_path}")


# ──────────────────────────────────────────────
# 6.  Save Utilities
# ──────────────────────────────────────────────

def save_model(state_dict: dict, config: dict):
    """Save model weights and a JSON config for later reconstruction."""
    save_dir = Path(config["model_save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    weights_path = save_dir / config["model_filename"]
    torch.save(state_dict, weights_path)
    print(f"  Model weights saved to : {weights_path}")

    # Config (architecture info + class names)
    model_meta = {
        "architecture": "CIFAR10CNN",
        "num_classes": config["num_classes"],
        "in_channels": config["in_channels"],
        "class_names": CLASS_NAMES,
        "dataset": config["dataset"],
        "image_size": config["image_size"],
        "epochs_trained": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
    }
    config_path = save_dir / config["config_filename"]
    with open(config_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"  Model config saved to  : {config_path}")


# ──────────────────────────────────────────────
# 7.  Main Entry Point
# ──────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  Phase 1 – CIFAR-10 CNN Training")
    print("  Deep Learning Model Authentication (Challenge-Response)")
    print("=" * 56)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▸ Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Data
    print("\n▸ Loading CIFAR-10 dataset …")
    try:
        train_loader, val_loader, test_loader = load_datasets(CONFIG)
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return

    # Model
    print("\n▸ Building model …")
    model = build_model(CONFIG, device)

    # Train
    print("\n▸ Training …")
    try:
        history, best_state = train_model(model, train_loader, val_loader, CONFIG, device)
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return

    # Plot
    print("\n▸ Saving training curves …")
    plot_history(history)

    # Save
    print("\n▸ Saving model …")
    if best_state is not None:
        save_model(best_state, CONFIG)
    else:
        print("  ✗ No best state was recorded — skipping save.")

    print("\n✓ Phase 1 complete.\n")


if __name__ == "__main__":
    main()
