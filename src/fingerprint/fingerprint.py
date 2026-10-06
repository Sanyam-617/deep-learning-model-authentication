"""
Behavioral Fingerprint Extraction

Loads the original CIFAR-10 CNN, runs inference on the generated
challenge set, and extracts:
    - predicted classes
    - confidence scores
    - probability vectors

The resulting reference fingerprint is saved for the authentication engine.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from train import CIFAR10CNN


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "original_model.pth"
DEFAULT_CHALLENGE_PATH = (
    PROJECT_ROOT / "data" / "challenges" / "challenge_set.pt"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "fingerprints" / "reference_fingerprint.pt"
)


# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model(model_path, device):
    """Load the original trained CNN."""

    model = CIFAR10CNN()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.to(device)

    # Important: inference mode
    model.eval()

    return model


# ---------------------------------------------------------
# Extract fingerprint
# ---------------------------------------------------------

def extract_fingerprint(
    model,
    challenge_path,
    device,
    batch_size=64
):
    """
    Run the original model on all challenge images and
    extract its behavioral fingerprint.
    """

    # Load challenge set
    challenge_data = torch.load(
        challenge_path,
        map_location="cpu"
    )

    images = challenge_data["images"]
    labels = challenge_data["labels"]
    sample_ids = challenge_data["sample_ids"]
    transform_types = challenge_data["transform_types"]

    print(f"Loaded {len(images)} challenge images.")

    # Challenge images are already normalized by the
    # challenge generator, so DO NOT normalize again.
    dataset = TensorDataset(images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    predicted_classes = []
    confidence_scores = []
    probability_vectors = []

    # No gradients are needed during fingerprint extraction
    with torch.no_grad():

        for (batch_images,) in loader:

            batch_images = batch_images.to(device)

            # Model output
            logits = model(batch_images)

            # Convert logits into probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Predicted class
            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            # Highest probability = confidence
            confidence = torch.max(
                probabilities,
                dim=1
            ).values

            predicted_classes.append(
                predictions.cpu()
            )

            confidence_scores.append(
                confidence.cpu()
            )

            probability_vectors.append(
                probabilities.cpu()
            )

    # Combine all batches
    predicted_classes = torch.cat(
        predicted_classes
    )

    confidence_scores = torch.cat(
        confidence_scores
    )

    probability_vectors = torch.cat(
        probability_vectors
    )

    # Create fingerprint
    fingerprint = {
        "predicted_classes": predicted_classes,
        "confidence_scores": confidence_scores,
        "probability_vectors": probability_vectors,

        # Metadata needed to match outputs
        # with the corresponding challenges
        "sample_ids": sample_ids,
        "transform_types": transform_types,
        "labels": labels,

        "class_names": CLASS_NAMES,
        "num_challenges": len(images),
    }

    return fingerprint


# ---------------------------------------------------------
# Save fingerprint
# ---------------------------------------------------------

def save_fingerprint(fingerprint, output_path):
    """Save the reference fingerprint to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        fingerprint,
        output_path
    )

    print(f"\nFingerprint saved to:")
    print(output_path)


# ---------------------------------------------------------
# Reproducibility verification
# ---------------------------------------------------------

def verify_reproducibility(fingerprint1, fingerprint2):
    """
    Check whether two fingerprint extractions are identical.
    """

    checks = [
        torch.equal(
            fingerprint1["predicted_classes"],
            fingerprint2["predicted_classes"]
        ),

        torch.equal(
            fingerprint1["confidence_scores"],
            fingerprint2["confidence_scores"]
        ),

        torch.equal(
            fingerprint1["probability_vectors"],
            fingerprint2["probability_vectors"]
        ),
    ]

    return all(checks)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Extract behavioral fingerprint from original CNN"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to original_model.pth"
    )

    parser.add_argument(
        "--challenges",
        default=DEFAULT_CHALLENGE_PATH,
        help="Path to challenge_set.pt"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save reference fingerprint"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    args = parser.parse_args()

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load original model
    print("\nLoading original CNN...")
    model = load_model(
        args.model,
        device
    )

    print("Original CNN loaded successfully.")

    # Extract fingerprint
    print("\nExtracting behavioral fingerprint...")

    fingerprint = extract_fingerprint(
        model=model,
        challenge_path=args.challenges,
        device=device,
        batch_size=args.batch_size
    )

    # Print basic information
    print("\nFingerprint information:")
    print(
        f"Number of challenges: "
        f"{fingerprint['num_challenges']}"
    )

    print(
        f"Probability vector shape: "
        f"{fingerprint['probability_vectors'].shape}"
    )

    print(
        f"Predictions shape: "
        f"{fingerprint['predicted_classes'].shape}"
    )

    print(
        f"Confidence shape: "
        f"{fingerprint['confidence_scores'].shape}"
    )

    # Save
    save_fingerprint(
        fingerprint,
        args.output
    )


if __name__ == "__main__":
    main()