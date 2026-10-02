"""
Verify Model - Deterministic Inference Check
=============================================
Deep Learning Model Authentication Using Behavioral Challenge-Response

This script:
  - Loads the CNN architecture & config from models/model_config.json
  - Loads the trained weights from models/original_model.pth
  - Runs inference TWICE on the same fixed batch of CIFAR-10 images
  - Verifies that predictions and output probabilities are bit-identical
  - Prints per-image predicted labels with confidence scores

Does NOT modify the saved checkpoint.
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# -- Import the model class from the training script ----------
from train import CIFAR10CNN


# ----------------------------------------------
# 1.  Paths & Defaults
# ----------------------------------------------

MODEL_DIR       = Path("models")
CONFIG_PATH     = MODEL_DIR / "model_config.json"
WEIGHTS_PATH    = MODEL_DIR / "original_model.pth"
DATA_DIR        = "./data"
BATCH_SIZE      = 16        # small batch for readable output
NUM_DISPLAY     = 16        # how many individual predictions to print


# ----------------------------------------------
# 2.  Load Config
# ----------------------------------------------

def load_config(config_path: Path) -> dict:
    """Read the JSON model config saved during training."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at '{config_path}'.\n"
            "  -> Run train.py first to generate it."
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"  Config loaded from : {config_path}")
    print(f"  Architecture       : {config['architecture']}")
    print(f"  Classes            : {config['num_classes']}")
    return config


# ----------------------------------------------
# 3.  Build & Load Model
# ----------------------------------------------

def load_model(config: dict, weights_path: Path, device: torch.device) -> CIFAR10CNN:
    """
    Reconstruct the CNN from config and load the saved weights.

    The checkpoint is loaded with `weights_only=True` for safety and is
    never written back -- the original file is not modified.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at '{weights_path}'.\n"
            "  -> Run train.py first to generate the checkpoint."
        )

    model = CIFAR10CNN(
        num_classes=config["num_classes"],
        in_channels=config["in_channels"],
    )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()          # critical: disables dropout, batchnorm running stats

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Weights loaded from: {weights_path}")
    print(f"  Parameters         : {total_params:,}")
    print(f"  Mode               : eval (inference)")
    return model


# ----------------------------------------------
# 4.  Data Loading (evaluation transforms only)
# ----------------------------------------------

def get_eval_transform():
    """Same normalization used during training -- no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])


def load_verification_batch(batch_size: int = BATCH_SIZE):
    """
    Return a fixed batch of CIFAR-10 *test* images and their ground-truth
    labels.  Using shuffle=False guarantees the same images every run.
    """
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False,
        download=True, transform=get_eval_transform(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))       # first batch
    return images, labels


# ----------------------------------------------
# 5.  Inference Helper
# ----------------------------------------------

@torch.no_grad()
def run_inference(model: CIFAR10CNN, images: torch.Tensor, device: torch.device):
    """
    Run a single forward pass.

    Returns
    -------
    logits       :  raw model outputs         (B x C)
    probs        :  softmax probabilities     (B x C)
    predictions  :  predicted class indices   (B,)
    confidences  :  confidence for predicted class (B,)
    """
    images = images.to(device)
    logits = model(images)
    probs  = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    return logits, probs, predictions, confidences


# ----------------------------------------------
# 6.  Display Predictions
# ----------------------------------------------

def display_predictions(predictions, confidences, labels, class_names, n=NUM_DISPLAY):
    """Print a formatted table of predictions vs ground truth."""
    n = min(n, len(predictions))
    print(f"\n  {'#':>3}  {'Predicted':>12}  {'Confidence':>11}  {'Ground Truth':>13}  {'Match':>6}")
    print("  " + "-" * 52)

    correct = 0
    for i in range(n):
        pred_idx   = predictions[i].item()
        true_idx   = labels[i].item()
        conf       = confidences[i].item() * 100
        pred_name  = class_names[pred_idx]
        true_name  = class_names[true_idx]
        match      = "YES" if pred_idx == true_idx else "NO"
        if pred_idx == true_idx:
            correct += 1
        print(f"  {i+1:>3}  {pred_name:>12}  {conf:>10.2f}%  {true_name:>13}  {match:>5}")

    acc = 100.0 * correct / n
    print(f"\n  Batch accuracy: {correct}/{n} ({acc:.1f}%)")


# ----------------------------------------------
# 7.  Consistency Check (double inference)
# ----------------------------------------------

def check_determinism(model, images, device):
    """
    Run inference twice on the *same* input and verify bit-identical outputs.

    Because the model is in eval mode (dropout disabled, batchnorm frozen),
    two forward passes with the same input must produce exactly the same
    logits, probabilities, and predicted classes.
    """
    logits_1, probs_1, preds_1, confs_1 = run_inference(model, images, device)
    logits_2, probs_2, preds_2, confs_2 = run_inference(model, images, device)

    # Bit-exact comparison
    logits_match = torch.equal(logits_1, logits_2)
    probs_match  = torch.equal(probs_1, probs_2)
    preds_match  = torch.equal(preds_1, preds_2)
    confs_match  = torch.equal(confs_1, confs_2)

    all_match = logits_match and probs_match and preds_match and confs_match

    print("\n  Determinism check (Run 1 vs Run 2):")
    print(f"    Logits identical        : {'PASS' if logits_match else 'FAIL'}")
    print(f"    Probabilities identical : {'PASS' if probs_match  else 'FAIL'}")
    print(f"    Predictions identical   : {'PASS' if preds_match  else 'FAIL'}")
    print(f"    Confidences identical   : {'PASS' if confs_match  else 'FAIL'}")

    return all_match, (logits_1, probs_1, preds_1, confs_1)


# ----------------------------------------------
# 8.  Main
# ----------------------------------------------

def main():
    print("=" * 56)
    print("  Model Verification - Deterministic Inference Check")
    print("  Deep Learning Model Authentication (Challenge-Response)")
    print("=" * 56)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n> Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Load config
    print("\n> Loading model configuration ...")
    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    class_names = config.get("class_names", [f"class_{i}" for i in range(config["num_classes"])])

    # Load model
    print("\n> Loading model weights ...")
    try:
        model = load_model(config, WEIGHTS_PATH, device)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # Load a fixed batch of images
    print("\n> Loading verification batch ...")
    try:
        images, labels = load_verification_batch()
        print(f"  Batch size : {images.size(0)}")
        print(f"  Image shape: {tuple(images.shape[1:])}")
    except Exception as e:
        print(f"  [ERROR] Failed to load data: {e}")
        sys.exit(1)

    # Double inference + consistency check
    print("\n> Running inference (x2) ...")
    all_match, (logits, probs, preds, confs) = check_determinism(model, images, device)

    # Display per-image predictions
    print("\n> Predictions (Run 1):")
    display_predictions(preds, confs, labels, class_names)

    # Final verdict
    print("\n" + "=" * 56)
    if all_match:
        print("  [PASS] Model outputs are deterministic.")
        print("    Predictions and probabilities matched exactly")
        print("    across two independent forward passes.")
    else:
        print("  [FAIL] Model outputs differed between runs!")
        print("    This may indicate that model.eval() was not set,")
        print("    or that a non-deterministic operation is present.")
    print("=" * 56 + "\n")

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
