"""
Controlled Model Modification Generator (E3–E7)
================================================
Generates controlled parameter-perturbed model variants from `models/original_model.pth`.

Variants:
  - modified_model_1pct.pth   (1% parameters perturbed)
  - modified_model_5pct.pth   (5% parameters perturbed)
  - modified_model_10pct.pth  (10% parameters perturbed)
  - modified_model_25pct.pth  (25% parameters perturbed)
  - modified_model_50pct.pth  (50% parameters perturbed)

Guarantees:
  1. Never modifies original_model.pth or reference_fingerprint.pt.
  2. Creates each variant independently from original_model.pth.
  3. Uses a deterministic seed and zero-mean Gaussian noise scaled by layer std.
  4. Fully audits and verifies changed parameter counts element-by-element.
  5. Saves a comprehensive JSON manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from train import CIFAR10CNN

PROJECT_ROOT = Path(__file__).resolve().parent
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "models" / "original_model.pth"
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "modified_models_manifest.json"

TARGET_CONFIGS = [
    {"name": "modified_model_1pct.pth", "target_fraction": 0.01, "seed": 52},
    {"name": "modified_model_5pct.pth", "target_fraction": 0.05, "seed": 92},
    {"name": "modified_model_10pct.pth", "target_fraction": 0.10, "seed": 142},
    {"name": "modified_model_25pct.pth", "target_fraction": 0.25, "seed": 292},
    {"name": "modified_model_50pct.pth", "target_fraction": 0.50, "seed": 542},
]


def compute_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def perturb_model(
    base_model: CIFAR10CNN,
    fraction: float,
    seed: int,
) -> Tuple[CIFAR10CNN, Dict[str, Any]]:
    """
    Independently perturb a target fraction of eligible weights from base_model.
    
    Procedure:
      - Set random seed.
      - For each parameter tensor, generate a boolean mask: rand < fraction.
      - Add zero-mean Gaussian noise scaled by the parameter tensor's standard deviation.
      - Returns modified model and audit record.
    """
    torch.manual_seed(seed)
    modified = copy.deepcopy(base_model)
    modified.eval()

    audit_layers: List[Dict[str, Any]] = []
    total_elements = 0
    total_modified = 0

    with torch.no_grad():
        for name, param in modified.named_parameters():
            numel = param.numel()
            total_elements += numel

            mask = torch.rand_like(param.data) < fraction
            num_changed = int(mask.sum().item())
            total_modified += num_changed

            param_std = float(param.data.std().item())
            if param_std == 0.0 or np.isnan(param_std):
                param_std = 0.05

            if num_changed > 0:
                noise = torch.randn_like(param.data[mask]) * param_std
                param.data[mask] += noise

            audit_layers.append({
                "layer_name": name,
                "shape": list(param.shape),
                "num_elements": numel,
                "modified_elements": num_changed,
                "layer_fraction_modified": round(num_changed / numel, 6),
                "layer_std": round(param_std, 6),
            })

    audit_summary = {
        "target_fraction": fraction,
        "target_percentage": f"{fraction * 100:.1f}%",
        "random_seed": seed,
        "noise_distribution": "Gaussian N(0, (layer_std)^2)",
        "total_model_parameters": total_elements,
        "total_modified_parameters": total_modified,
        "actual_fraction_modified": round(total_modified / total_elements, 6),
        "actual_percentage_modified": f"{(total_modified / total_elements) * 100:.4f}%",
        "layer_audit": audit_layers,
    }

    return modified, audit_summary


def verify_checkpoint(
    checkpoint_path: Path,
    original_state: Dict[str, torch.Tensor],
    expected_modified: int,
) -> Dict[str, Any]:
    """Verify saved checkpoint loads cleanly and exactly matches expected modification count."""
    reloaded_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    test_model = CIFAR10CNN()
    test_model.load_state_dict(reloaded_state)
    test_model.eval()

    total_diff = 0
    layer_diffs = {}
    for name, orig_p in original_state.items():
        new_p = reloaded_state[name]
        diff_mask = (orig_p != new_p)
        diff_count = int(diff_mask.sum().item())
        total_diff += diff_count
        layer_diffs[name] = diff_count

    matches_expected = (total_diff == expected_modified)
    return {
        "load_successful": True,
        "architecture_compatible": True,
        "total_differing_elements": total_diff,
        "matches_expected_modified_count": matches_expected,
        "layer_diff_counts": layer_diffs,
    }


def main():
    print("=" * 78)
    print("  CONTROLLED MODEL MODIFICATION GENERATOR (E3–E7)")
    print("=" * 78)

    orig_hash_before = compute_sha256(ORIGINAL_MODEL_PATH)
    print(f"\n[+] Verified original model: {ORIGINAL_MODEL_PATH.name}")
    print(f"    SHA256: {orig_hash_before}")

    # Load original model
    original_model = CIFAR10CNN()
    original_state = torch.load(ORIGINAL_MODEL_PATH, map_location="cpu", weights_only=True)
    original_model.load_state_dict(original_state)
    original_model.eval()

    total_params = sum(p.numel() for p in original_model.parameters())
    print(f"[+] Total trainable parameters: {total_params:,}")

    manifest_records = []

    for cfg in TARGET_CONFIGS:
        fname = cfg["name"]
        target_f = cfg["target_fraction"]
        seed = cfg["seed"]
        out_path = MODELS_DIR / fname

        print(f"\n---> Generating: {fname} (Target: {target_f*100:.1f}%, Seed: {seed})")

        # Independent perturbation from clean original
        pert_model, audit = perturb_model(original_model, fraction=target_f, seed=seed)

        # Save checkpoint
        torch.save(pert_model.state_dict(), out_path)
        ckpt_hash = compute_sha256(out_path)
        file_size = out_path.stat().st_size

        # Verification of saved file
        verification = verify_checkpoint(out_path, original_state, audit["total_modified_parameters"])

        print(f"     Saved: {out_path.name} ({file_size:,} bytes)")
        print(f"     SHA256: {ckpt_hash}")
        print(
            f"     Modified elements: {audit['total_modified_parameters']:,} / {total_params:,} "
            f"({audit['actual_percentage_modified']})"
        )
        print(f"     Verification check: {'PASS' if verification['matches_expected_modified_count'] else 'FAIL'}")

        record = {
            "checkpoint_filename": fname,
            "checkpoint_path": str(out_path),
            "target_fraction": target_f,
            "target_percentage": f"{target_f * 100:.1f}%",
            "random_seed": seed,
            "modification_procedure": {
                "mask_generation": f"Bernoulli trial per weight with p={target_f} using torch.manual_seed({seed})",
                "noise_distribution": "Additive zero-mean Gaussian noise N(0, sigma^2)",
                "noise_scaling": "sigma = layer_std (standard deviation of the layer's weights)",
                "source_model": "models/original_model.pth (independent copy)",
            },
            "parameters_summary": {
                "total_model_parameters": total_params,
                "total_modified_elements": audit["total_modified_parameters"],
                "actual_fraction_modified": audit["actual_fraction_modified"],
                "actual_percentage_modified": audit["actual_percentage_modified"],
            },
            "file_info": {
                "file_size_bytes": file_size,
                "sha256": ckpt_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "verification": verification,
            "layer_breakdown": audit["layer_audit"],
        }
        manifest_records.append(record)

    # Save manifest
    manifest_data = {
        "title": "Manifest of Controlled Modified Model Checkpoints (E3–E7)",
        "source_model": {
            "path": str(ORIGINAL_MODEL_PATH),
            "sha256": orig_hash_before,
            "total_parameters": total_params,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "modified_models": manifest_records,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n[+] Saved complete modification manifest to: {MANIFEST_PATH}")

    # Final original model hash check
    orig_hash_after = compute_sha256(ORIGINAL_MODEL_PATH)
    print(f"\n[+] Final check: Original model hash unchanged: {orig_hash_after == orig_hash_before}")
    print("=" * 78)


if __name__ == "__main__":
    main()

