"""
Deep Learning Model Authentication Pipeline
===========================================
Central entry point for end-to-end behavioral challenge-response model authentication.

Pipeline stages:
  1. Validates inputs, challenge set, and original CNN checkpoint.
  2. Ensures registered reference fingerprint exists (loads or extracts deterministically).
  3. Evaluates one or more candidate/submitted PyTorch models.
  4. Computes prediction agreement, cosine similarity, and KL divergence.
  5. Applies configurable threshold decision logic.
  6. Outputs formatted terminal reports and saves results in JSON and CSV formats.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import CIFAR10CNN
from src.fingerprint.fingerprint import (
    DEFAULT_CHALLENGE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_REF_FP_PATH,
    extract_fingerprint,
    load_model,
    save_fingerprint,
)
from src.authentication.authenticator import (
    AuthenticationMetrics,
    AuthenticationReport,
    AuthenticationThresholds,
    FingerprintValidationError,
    ModelAuthenticator,
    validate_fingerprint_alignment,
    validate_fingerprint_schema,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


# ──────────────────────────────────────────────
# 1. Validation Helpers
# ──────────────────────────────────────────────

def validate_challenge_set(challenge_path: Path) -> Dict[str, torch.Tensor]:
    """Validate that the challenge set file exists and conforms to the expected schema."""
    if not challenge_path.exists():
        raise FileNotFoundError(
            f"Challenge set not found at: '{challenge_path}'\n"
            "  -> Run 'python -m src.challenges.challenge_generator' to generate it."
        )

    try:
        data = torch.load(challenge_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to read challenge set at '{challenge_path}': {e}") from e

    required_keys = ["images", "labels", "sample_ids", "transform_types"]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"Challenge set missing required key: '{k}'")

    images = data["images"]
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Expected images tensor of shape (N, 3, H, W), got {getattr(images, 'shape', type(images))}"
        )

    n_samples = images.shape[0]
    for k in ["labels", "sample_ids", "transform_types"]:
        tensor = data[k]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != n_samples:
            raise ValueError(f"Length mismatch for key '{k}': expected {n_samples}, got {getattr(tensor, 'shape', None)}")

    return data


def validate_and_load_candidate_model(
    model_path: Path,
    device: torch.device,
) -> CIFAR10CNN:
    """Validate checkpoint integrity and architecture compatibility, then return model in eval mode."""
    if not model_path.exists():
        raise FileNotFoundError(f"Submitted model checkpoint not found: '{model_path}'")

    model = CIFAR10CNN()

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # Fall back to weights_only=False for legacy / pickled checkpoints
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as err:
            raise ValueError(f"Cannot load PyTorch checkpoint '{model_path}': {err}") from err

    # Extract state_dict if wrapped in a checkpoint dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Checkpoint '{model_path}' must be a dictionary or state_dict, got {type(checkpoint)}")

    # Check key compatibility
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        raise ValueError(
            f"Model architecture mismatch in '{model_path}'. Missing required parameters: {sorted(missing)}"
        )
    if unexpected:
        raise ValueError(
            f"Model architecture mismatch in '{model_path}'. Unexpected parameters: {sorted(unexpected)}"
        )

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Failed to load state_dict into CIFAR10CNN architecture for '{model_path}': {e}") from e

    model.to(device)
    model.eval()  # Mandatory eval mode
    return model


# ──────────────────────────────────────────────
# 2. Reference Fingerprint Loader
# ──────────────────────────────────────────────

def get_or_create_reference_fingerprint(
    original_model_path: Path,
    challenge_path: Path,
    reference_fp_path: Path,
    device: torch.device,
    batch_size: int = 64,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Load the registered reference fingerprint, or extract and save it if not yet present.

    Guarantees that original_model.pth is never modified, and existing reference
    fingerprint files are not overwritten unless force_regenerate is True.
    """
    if reference_fp_path.exists() and not force_regenerate:
        print(f"[+] Loading existing reference fingerprint: {reference_fp_path}")
        ref_fp = torch.load(reference_fp_path, map_location="cpu", weights_only=False)
        validate_fingerprint_schema(ref_fp, "Reference fingerprint")
        return ref_fp

    if not original_model_path.exists():
        raise FileNotFoundError(
            f"Registered original model not found at: '{original_model_path}'\n"
            "  -> Run 'python train.py' first to produce original_model.pth."
        )

    print(f"[*] Extracting reference fingerprint from original model: {original_model_path}")
    original_model = validate_and_load_candidate_model(original_model_path, device)

    ref_fp = extract_fingerprint(
        model=original_model,
        challenge_path=str(challenge_path),
        device=device,
        batch_size=batch_size,
    )
    validate_fingerprint_schema(ref_fp, "Newly extracted reference fingerprint")

    # Save without overwriting unless forced
    if not reference_fp_path.exists() or force_regenerate:
        save_fingerprint(ref_fp, str(reference_fp_path))

    return ref_fp


# ──────────────────────────────────────────────
# 3. Pipeline Core Runner
# ──────────────────────────────────────────────

def authenticate_model(
    model_path: Path,
    reference_fingerprint: Dict[str, Any],
    challenge_path: Path,
    authenticator: ModelAuthenticator,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Authenticate a single candidate model against the reference fingerprint."""
    t0 = time.time()

    # Load candidate model in eval mode
    model = validate_and_load_candidate_model(model_path, device)

    # Extract candidate fingerprint under torch.no_grad()
    candidate_fp = extract_fingerprint(
        model=model,
        challenge_path=str(challenge_path),
        device=device,
        batch_size=batch_size,
    )

    # Validate alignment against reference
    validate_fingerprint_alignment(reference_fingerprint, candidate_fp)

    # Compute metrics and evaluate thresholds
    metrics = authenticator.compare_metrics(reference_fingerprint, candidate_fp)
    report = authenticator.evaluate_thresholds(metrics)

    elapsed = time.time() - t0

    result = report.to_dict()
    result["model_path"] = str(model_path)
    result["model_filename"] = model_path.name
    result["evaluation_time_sec"] = round(elapsed, 3)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["decision"] = "AUTHENTIC" if report.authenticated else "REJECTED"

    return result


def run_pipeline(
    submitted_model_paths: List[Path],
    original_model_path: Path = DEFAULT_MODEL_PATH,
    challenge_path: Path = DEFAULT_CHALLENGE_PATH,
    reference_fp_path: Path = DEFAULT_REF_FP_PATH,
    thresholds: Optional[AuthenticationThresholds] = None,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    batch_size: int = 64,
    device_name: Optional[str] = None,
    force_regenerate_ref: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the complete end-to-end authentication pipeline across one or more models."""
    print("=" * 80)
    print("  DEEP LEARNING MODEL AUTHENTICATION PIPELINE")
    print("  Behavioral Challenge–Response Verification Engine")
    print("=" * 80)

    # Select compute device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Compute device: {device}")

    # Validate challenge set
    print(f"\n[1/4] Validating challenge set at: {challenge_path}")
    validate_challenge_set(challenge_path)
    print("      Challenge set verified successfully.")

    # Load or generate reference fingerprint
    print(f"\n[2/4] Resolving registered reference fingerprint...")
    ref_fp = get_or_create_reference_fingerprint(
        original_model_path=original_model_path,
        challenge_path=challenge_path,
        reference_fp_path=reference_fp_path,
        device=device,
        batch_size=batch_size,
        force_regenerate=force_regenerate_ref,
    )
    print(f"      Reference fingerprint ready ({ref_fp['num_challenges']} challenges).")

    # Prepare authenticator
    auth_thresholds = thresholds or AuthenticationThresholds()
    authenticator = ModelAuthenticator(auth_thresholds)

    print(f"\n[3/4] Threshold configuration:")
    print(f"      • Prediction Agreement threshold : >= {auth_thresholds.prediction_threshold:.2%}")
    print(f"      • Cosine Similarity threshold    : >= {auth_thresholds.cosine_threshold:.4f}")
    print(f"      • KL Divergence threshold        : <= {auth_thresholds.kl_threshold:.4f}")

    # Authenticate submitted models
    print(f"\n[4/4] Authenticating {len(submitted_model_paths)} submitted model(s)...")
    results: List[Dict[str, Any]] = []

    for idx, model_path in enumerate(submitted_model_paths, start=1):
        print(f"\n  ---> [{idx}/{len(submitted_model_paths)}] Evaluating: {model_path.name}")
        try:
            result = authenticate_model(
                model_path=model_path,
                reference_fingerprint=ref_fp,
                challenge_path=challenge_path,
                authenticator=authenticator,
                device=device,
                batch_size=batch_size,
            )
            results.append(result)

            tag = "[AUTHENTIC]" if result["authenticated"] else "[REJECTED]"
            print(f"       Decision: {tag}")
            print(
                f"       Agreement: {result['prediction_agreement']:.2%} "
                f"({'PASS' if result['prediction_pass'] else 'FAIL'}) | "
                f"Cosine: {result['cosine_similarity']:.4f} "
                f"({'PASS' if result['cosine_pass'] else 'FAIL'}) | "
                f"KL Div: {result['kl_divergence']:.4f} "
                f"({'PASS' if result['kl_pass'] else 'FAIL'})"
            )
        except Exception as e:
            print(f"       [ERROR] Authentication failed for '{model_path.name}': {e}")
            err_result = {
                "model_path": str(model_path),
                "model_filename": model_path.name,
                "decision": "ERROR",
                "authenticated": False,
                "error_message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            results.append(err_result)

    # Display summary table
    print_summary_table(results, auth_thresholds)

    # Save results
    save_results(results, output_dir=output_dir, output_json=output_json, output_csv=output_csv)

    return results


# ──────────────────────────────────────────────
# 4. Reporting & Export Utilities
# ──────────────────────────────────────────────

def print_summary_table(results: List[Dict[str, Any]], thresholds: AuthenticationThresholds) -> None:
    """Print an aligned ASCII summary table of authentication results."""
    print("\n" + "=" * 98)
    print("AUTHENTICATION PIPELINE SUMMARY")
    print("=" * 98)
    header = (
        f"{'Model File':<28} | {'Decision':<10} | {'Agreement':<10} | {'Cosine Sim':<10} | "
        f"{'KL Div':<9} | {'Checks (Agr/Cos/KL)':<19}"
    )
    print(header)
    print("-" * 98)

    for r in results:
        fname = r.get("model_filename", "unknown")[:28]
        dec = r.get("decision", "ERROR")

        if dec == "ERROR":
            print(f"{fname:<28} | {dec:<10} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<9} | {r.get('error_message', '')[:19]}")
            continue

        agr = f"{r['prediction_agreement']:.2%}"
        cos = f"{r['cosine_similarity']:.4f}"
        kl = f"{r['kl_divergence']:.4f}"

        c_pass = "P" if r.get("cosine_pass") else "F"
        p_pass = "P" if r.get("prediction_pass") else "F"
        k_pass = "P" if r.get("kl_pass") else "F"
        checks = f"{p_pass} / {c_pass} / {k_pass}"

        print(f"{fname:<28} | {dec:<10} | {agr:<10} | {cos:<10} | {kl:<9} | {checks:<19}")

    print("=" * 98)
    print(
        f"Thresholds: Agreement >= {thresholds.prediction_threshold:.2%} | "
        f"Cosine >= {thresholds.cosine_threshold:.4f} | "
        f"KL Div <= {thresholds.kl_threshold:.4f}"
    )
    print("=" * 98)


def save_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> None:
    """Persist pipeline results to JSON and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_json or (output_dir / "authentication_results.json")
    csv_path = output_csv or (output_dir / "authentication_results.csv")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved detailed JSON report to : {json_path}")

    # Save CSV
    fieldnames = [
        "model_filename",
        "decision",
        "authenticated",
        "prediction_agreement",
        "cosine_similarity",
        "kl_divergence",
        "matching_predictions",
        "mismatching_predictions",
        "num_challenges",
        "prediction_pass",
        "cosine_pass",
        "kl_pass",
        "evaluation_time_sec",
        "timestamp",
        "model_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[+] Saved summary CSV table to  : {csv_path}\n")


# ──────────────────────────────────────────────
# 5. CLI Interface
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Learning Model Authentication Pipeline (Behavioral Challenge-Response)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(DEFAULT_MODEL_PATH)],
        help="One or more submitted candidate model checkpoint paths (.pth).",
    )
    parser.add_argument(
        "--original_model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the registered original CNN weights.",
    )
    parser.add_argument(
        "--challenges",
        default=str(DEFAULT_CHALLENGE_PATH),
        help="Path to the generated challenge set (.pt).",
    )
    parser.add_argument(
        "--reference_fingerprint",
        default=str(DEFAULT_REF_FP_PATH),
        help="Path to the registered reference fingerprint (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save JSON and CSV reports.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional custom path for the output JSON file.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional custom path for the output CSV file.",
    )
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.90,
        help="Threshold for mean cosine similarity (higher is better).",
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.85,
        help="Threshold for prediction agreement fraction (higher is better).",
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=0.15,
        help="Threshold for mean KL divergence (lower is better).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--force_regenerate_ref",
        action="store_true",
        help="Force regeneration of reference fingerprint even if one exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    submitted_paths = [Path(p) for p in args.models]
    thresholds = AuthenticationThresholds(
        cosine_threshold=args.cosine_threshold,
        prediction_threshold=args.prediction_threshold,
        kl_threshold=args.kl_threshold,
    )

    out_json = Path(args.output_json) if args.output_json else None
    out_csv = Path(args.output_csv) if args.output_csv else None

    run_pipeline(
        submitted_model_paths=submitted_paths,
        original_model_path=Path(args.original_model),
        challenge_path=Path(args.challenges),
        reference_fp_path=Path(args.reference_fingerprint),
        thresholds=thresholds,
        output_dir=Path(args.output_dir),
        output_json=out_json,
        output_csv=out_csv,
        batch_size=args.batch_size,
        device_name=args.device,
        force_regenerate_ref=args.force_regenerate_ref,
    )


if __name__ == "__main__":
    main()

