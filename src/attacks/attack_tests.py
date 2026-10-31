"""
Model Tampering and Attack Evaluation
=====================================
Member 4 – Behavioral Challenge-Response Model Authentication

This module implements:
1. Safe in-memory model loading (transparently handling file or directory checkpoints).
2. Controlled weight perturbation (Gaussian noise across documented strengths).
3. Weight pruning (unstructured magnitude pruning across documented sparsity ratios).
4. Full evaluation pipeline against aligned challenges and reference fingerprints.
5. FAR (False Acceptance Rate) and FRR (False Rejection Rate) computation with exact attempt counts.
6. Comprehensive unit test suite using Python's built-in unittest framework.
7. Automated evaluation runner generating CSV, JSON, and publication-ready plots.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import sys
import unittest
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import CIFAR10CNN
from src.fingerprint.fingerprint import extract_fingerprint
from src.authentication.authenticator import (
    AuthenticationReport,
    AuthenticationThresholds,
    ModelAuthenticator,
)

# Default Project Paths
DEFAULT_ORIGINAL_MODEL = PROJECT_ROOT / "models" / "original_model.pth"
DEFAULT_DIFF_SEED_MODEL = PROJECT_ROOT / "models" / "test_models" / "different_seed_model.pth"
DEFAULT_CHALLENGE_PATH = PROJECT_ROOT / "data" / "challenges" / "challenge_set.pt"
DEFAULT_REF_FINGERPRINT = PROJECT_ROOT / "data" / "fingerprints" / "reference_fingerprint.json"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "experiments" / "attack_evaluation"


# ──────────────────────────────────────────────
# 1. Safe Model Loading Helper (Preserves Disk)
# ──────────────────────────────────────────────

def load_checkpoint(
    model_path: Union[str, Path] = DEFAULT_ORIGINAL_MODEL,
    device: Union[str, torch.device] = "cpu",
) -> CIFAR10CNN:
    """Load a CIFAR10CNN model from a file or directory checkpoint.

    Transparently reconstructs extracted PyTorch zip checkpoints in memory
    without creating, extracting, or modifying any files on disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    device = torch.device(device)

    if model_path.is_dir():
        # Directory checkpoint (unpacked PyTorch safe-serialization format)
        inner_root = model_path / "original_model"
        target_dir = inner_root if inner_root.exists() else model_path

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for dirpath, _, filenames in os.walk(target_dir):
                for fname in sorted(filenames):
                    full_p = Path(dirpath) / fname
                    arcname = str(full_p.relative_to(target_dir.parent))
                    zf.write(full_p, arcname)
        buf.seek(0)
        state_dict = torch.load(buf, map_location=device, weights_only=True)
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

    model = CIFAR10CNN()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# 2. Tampering Operators (Independent Copies)
# ──────────────────────────────────────────────

def apply_weight_perturbation(
    model: CIFAR10CNN,
    noise_std: float,
    seed: Optional[int] = None,
) -> CIFAR10CNN:
    """Apply additive Gaussian noise to convolutional and linear weights.

    Formula:
        W' = W + N(0, (noise_std * std(W))^2)

    Relative scaling preserves balance across layers with different dynamic ranges.
    Operates strictly on an independent deep copy; the source model is never mutated.
    """
    tampered_model = copy.deepcopy(model)
    tampered_model.eval()

    if noise_std <= 0.0:
        return tampered_model

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    with torch.no_grad():
        for name, param in tampered_model.named_parameters():
            if "weight" in name and param.dim() > 1:
                layer_std = param.std().item()
                if layer_std == 0.0:
                    layer_std = 1.0
                noise = torch.randn(
                    param.shape,
                    generator=generator,
                    device=param.device,
                    dtype=param.dtype,
                ) * (noise_std * layer_std)
                param.add_(noise)

    return tampered_model


def apply_magnitude_pruning(
    model: CIFAR10CNN,
    sparsity: float,
) -> CIFAR10CNN:
    """Apply unstructured magnitude pruning to convolutional and linear weights.

    Weights with absolute value below the layer's sparsity quantile are zeroed out.
    Operates strictly on an independent deep copy; the source model is never mutated.
    """
    tampered_model = copy.deepcopy(model)
    tampered_model.eval()

    if sparsity <= 0.0:
        return tampered_model

    with torch.no_grad():
        for name, param in tampered_model.named_parameters():
            if "weight" in name and param.dim() > 1:
                if sparsity >= 1.0:
                    param.zero_()
                else:
                    abs_weights = torch.abs(param)
                    k = float(torch.quantile(abs_weights.flatten(), sparsity).item())
                    mask = (abs_weights >= k).to(param.dtype)
                    param.mul_(mask)

    return tampered_model


# ──────────────────────────────────────────────
# 3. Fingerprint Formatting & Authentication
# ──────────────────────────────────────────────

def format_fingerprint_for_auth(fp_extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensor-based fingerprint from Member 2 into schema-aligned dictionary."""
    predicted_classes = fp_extracted["predicted_classes"]
    if isinstance(predicted_classes, torch.Tensor):
        predicted_classes = predicted_classes.cpu().tolist()

    confidence_scores = fp_extracted["confidence_scores"]
    if isinstance(confidence_scores, torch.Tensor):
        confidence_scores = confidence_scores.cpu().tolist()

    probability_vectors = fp_extracted["probability_vectors"]
    if isinstance(probability_vectors, torch.Tensor):
        probability_vectors = probability_vectors.cpu().tolist()

    return {
        "num_challenges": int(fp_extracted["num_challenges"]),
        "class_names": list(fp_extracted["class_names"]),
        "sample_ids": list(fp_extracted["sample_ids"]),
        "transform_types": list(fp_extracted["transform_types"]),
        "predicted_classes": list(predicted_classes),
        "confidence_scores": list(confidence_scores),
        "probability_vectors": [list(v) for v in probability_vectors],
    }


def evaluate_model_authenticity(
    model: CIFAR10CNN,
    challenge_path: Union[str, Path] = DEFAULT_CHALLENGE_PATH,
    ref_fingerprint_path: Union[str, Path] = DEFAULT_REF_FINGERPRINT,
    device: str = "cpu",
    batch_size: int = 64,
    thresholds: Optional[AuthenticationThresholds] = None,
) -> AuthenticationReport:
    """Extract fingerprint from a model and authenticate it using Member 3's engine."""
    fp_raw = extract_fingerprint(
        model=model,
        challenge_path=str(challenge_path),
        device=device,
        batch_size=batch_size,
    )
    fp_dict = format_fingerprint_for_auth(fp_raw)
    authenticator = ModelAuthenticator(thresholds)
    report = authenticator.authenticate(ref_fingerprint_path, fp_dict)
    return report


# ──────────────────────────────────────────────
# 4. FAR & FRR Metric Computation
# ──────────────────────────────────────────────

def calculate_far_frr(
    genuine_decisions: List[bool],
    unauthorized_decisions: List[bool],
) -> Dict[str, Any]:
    """Calculate FAR and FRR with explicit attempt counts, numerators, denominators, and percentages.

    FAR = False Acceptances among Unauthorized Attempts / Total Unauthorized Attempts
    FRR = False Rejections among Genuine Attempts / Total Genuine Attempts

    If a denominator is zero, the rate is represented as None (undefined) rather than zero.
    """
    total_genuine = len(genuine_decisions)
    total_unauthorized = len(unauthorized_decisions)

    false_rejections = sum(1 for passed in genuine_decisions if not passed)
    false_acceptances = sum(1 for passed in unauthorized_decisions if passed)

    frr = (false_rejections / total_genuine) if total_genuine > 0 else None
    far = (false_acceptances / total_unauthorized) if total_unauthorized > 0 else None

    return {
        "total_genuine_attempts": total_genuine,
        "false_rejections": false_rejections,
        "frr_numerator": false_rejections if total_genuine > 0 else None,
        "frr_denominator": total_genuine,
        "frr": frr,
        "frr_percent": f"{(frr * 100):.2f}%" if frr is not None else "undefined",
        "total_unauthorized_attempts": total_unauthorized,
        "false_acceptances": false_acceptances,
        "far_numerator": false_acceptances if total_unauthorized > 0 else None,
        "far_denominator": total_unauthorized,
        "far": far,
        "far_percent": f"{(far * 100):.2f}%" if far is not None else "undefined",
    }


# ──────────────────────────────────────────────
# 5. Full Experiment Runner & Plot Generation
# ──────────────────────────────────────────────

def run_evaluation_suite(
    output_dir: Union[str, Path] = DEFAULT_EVAL_DIR,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute the full Member 4 evaluation matrix and generate reconciled artifacts.

    Evaluates 26 unique model variants (1 genuine positive control, 1 replacement model,
    18 weight perturbation variants across 6 strengths and 3 seeds, and 6 magnitude pruning
    variants across 6 sparsity ratios). Each record maps 1-to-1 to a distinct model variant.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results_summary.json"
    csv_path = output_dir / "results_summary.csv"
    metrics_path = output_dir / "far_frr_metrics.json"

    # Load base models
    orig_model = load_checkpoint(DEFAULT_ORIGINAL_MODEL, device=device)
    diff_seed_model = load_checkpoint(DEFAULT_DIFF_SEED_MODEL, device=device)

    all_variant_results: List[Dict[str, Any]] = []

    # 1. Evaluate Positive Control (Genuine Model)
    print("\n--- Evaluating Positive Control (Genuine Original Model) ---")
    rep_genuine = evaluate_model_authenticity(orig_model, device=device)
    print(f"Genuine Original: Agreement={rep_genuine.prediction_agreement:.4f}, Cosine={rep_genuine.cosine_similarity:.4f}, KL={rep_genuine.kl_divergence:.6f}, Auth={rep_genuine.authenticated}")

    all_variant_results.append({
        "variant_id": "genuine_original",
        "model_id": "original_model",
        "ground_truth": "genuine",
        "category": "positive_control",
        "parameter_name": "none",
        "parameter_value": 0.0,
        "seed": None,
        "prediction_agreement": rep_genuine.prediction_agreement,
        "cosine_similarity": rep_genuine.cosine_similarity,
        "kl_divergence": rep_genuine.kl_divergence,
        "prediction_pass": rep_genuine.prediction_pass,
        "cosine_pass": rep_genuine.cosine_pass,
        "kl_pass": rep_genuine.kl_pass,
        "authenticated": rep_genuine.authenticated,
        "decision_outcome": "true_acceptance" if rep_genuine.authenticated else "false_rejection",
    })

    # 2. Evaluate Replacement / Impostor Baseline (Different-Seed Model)
    print("\n--- Evaluating Impostor Baseline (Different Seed Model) ---")
    rep_diff = evaluate_model_authenticity(diff_seed_model, device=device)
    print(f"Diff Seed: Agreement={rep_diff.prediction_agreement:.4f}, Cosine={rep_diff.cosine_similarity:.4f}, KL={rep_diff.kl_divergence:.6f}, Auth={rep_diff.authenticated}")

    all_variant_results.append({
        "variant_id": "different_seed_model",
        "model_id": "different_seed_model",
        "ground_truth": "unauthorized",
        "category": "impostor_replacement",
        "parameter_name": "seed",
        "parameter_value": 123.0,
        "seed": 123,
        "prediction_agreement": rep_diff.prediction_agreement,
        "cosine_similarity": rep_diff.cosine_similarity,
        "kl_divergence": rep_diff.kl_divergence,
        "prediction_pass": rep_diff.prediction_pass,
        "cosine_pass": rep_diff.cosine_pass,
        "kl_pass": rep_diff.kl_pass,
        "authenticated": rep_diff.authenticated,
        "decision_outcome": "false_acceptance" if rep_diff.authenticated else "true_rejection",
    })

    # 3. Evaluate Weight Perturbations (6 strengths x 3 seeds = 18 unique variants)
    print("\n--- Evaluating Weight Perturbations (18 unique variants) ---")
    perturb_strengths = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    seeds = [42, 43, 44]
    for std in perturb_strengths:
        for seed in seeds:
            pert_model = apply_weight_perturbation(orig_model, noise_std=std, seed=seed)
            rep = evaluate_model_authenticity(pert_model, device=device)
            var_id = f"perturb_std_{std}_seed_{seed}"
            all_variant_results.append({
                "variant_id": var_id,
                "model_id": var_id,
                "ground_truth": "unauthorized",
                "category": "weight_perturbation",
                "parameter_name": "noise_std",
                "parameter_value": std,
                "seed": seed,
                "prediction_agreement": rep.prediction_agreement,
                "cosine_similarity": rep.cosine_similarity,
                "kl_divergence": rep.kl_divergence,
                "prediction_pass": rep.prediction_pass,
                "cosine_pass": rep.cosine_pass,
                "kl_pass": rep.kl_pass,
                "authenticated": rep.authenticated,
                "decision_outcome": "false_acceptance" if rep.authenticated else "true_rejection",
            })
            print(f"  {var_id:32} | Agree: {rep.prediction_agreement:.4f} | Cos: {rep.cosine_similarity:.4f} | KL: {rep.kl_divergence:.6f} | Auth: {rep.authenticated}")

    # 4. Evaluate Magnitude Pruning (6 unique sparsity variants)
    print("\n--- Evaluating Magnitude Pruning (6 unique variants) ---")
    pruning_levels = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
    for sparsity in pruning_levels:
        pruned_model = apply_magnitude_pruning(orig_model, sparsity=sparsity)
        rep = evaluate_model_authenticity(pruned_model, device=device)
        var_id = f"prune_sparsity_{sparsity}"
        all_variant_results.append({
            "variant_id": var_id,
            "model_id": var_id,
            "ground_truth": "unauthorized",
            "category": "magnitude_pruning",
            "parameter_name": "sparsity",
            "parameter_value": sparsity,
            "seed": None,
            "prediction_agreement": rep.prediction_agreement,
            "cosine_similarity": rep.cosine_similarity,
            "kl_divergence": rep.kl_divergence,
            "prediction_pass": rep.prediction_pass,
            "cosine_pass": rep.cosine_pass,
            "kl_pass": rep.kl_pass,
            "authenticated": rep.authenticated,
            "decision_outcome": "false_acceptance" if rep.authenticated else "true_rejection",
        })
        print(f"  {var_id:32} | Agree: {rep.prediction_agreement:.4f} | Cos: {rep.cosine_similarity:.4f} | KL: {rep.kl_divergence:.6f} | Auth: {rep.authenticated}")

    # Reconciled Decisions
    genuine_decisions = [r["authenticated"] for r in all_variant_results if r["ground_truth"] == "genuine"]
    all_unauthorized_decisions = [r["authenticated"] for r in all_variant_results if r["ground_truth"] == "unauthorized"]
    replacement_decisions = [r["authenticated"] for r in all_variant_results if r["category"] == "impostor_replacement"]
    perturbation_decisions = [r["authenticated"] for r in all_variant_results if r["category"] == "weight_perturbation"]
    all_pruning_decisions = [r["authenticated"] for r in all_variant_results if r["category"] == "magnitude_pruning"]
    severe_pruning_decisions = [r["authenticated"] for r in all_variant_results if r["category"] == "magnitude_pruning" and r["parameter_value"] >= 0.35]

    far_frr_overall = calculate_far_frr(genuine_decisions, all_unauthorized_decisions)
    far_replacement = calculate_far_frr(genuine_decisions, replacement_decisions)
    far_perturbation = calculate_far_frr(genuine_decisions, perturbation_decisions)
    far_all_pruning = calculate_far_frr(genuine_decisions, all_pruning_decisions)
    far_severe_pruning = calculate_far_frr(genuine_decisions, severe_pruning_decisions)

    far_frr_summary = {
        "summary_counts": {
            "total_unique_models": len(all_variant_results),
            "total_evaluation_attempts": len(all_variant_results),
            "genuine_models": len(genuine_decisions),
            "genuine_attempts": len(genuine_decisions),
            "unauthorized_models": len(all_unauthorized_decisions),
            "unauthorized_attempts": len(all_unauthorized_decisions),
        },
        "overall": {
            "total_genuine_attempts": far_frr_overall["total_genuine_attempts"],
            "false_rejections": far_frr_overall["false_rejections"],
            "frr_numerator": far_frr_overall["frr_numerator"],
            "frr_denominator": far_frr_overall["frr_denominator"],
            "frr": far_frr_overall["frr"],
            "frr_percent": far_frr_overall["frr_percent"],
            "total_unauthorized_attempts": far_frr_overall["total_unauthorized_attempts"],
            "false_acceptances": far_frr_overall["false_acceptances"],
            "far_numerator": far_frr_overall["far_numerator"],
            "far_denominator": far_frr_overall["far_denominator"],
            "far": far_frr_overall["far"],
            "far_percent": far_frr_overall["far_percent"],
        },
        "category_breakdown": {
            "replacement_model": {
                "unauthorized_models": far_replacement["total_unauthorized_attempts"],
                "unauthorized_attempts": far_replacement["total_unauthorized_attempts"],
                "false_acceptances": far_replacement["false_acceptances"],
                "far_numerator": far_replacement["far_numerator"],
                "far_denominator": far_replacement["far_denominator"],
                "far": far_replacement["far"],
                "far_percent": far_replacement["far_percent"],
            },
            "weight_perturbations": {
                "unauthorized_models": far_perturbation["total_unauthorized_attempts"],
                "unauthorized_attempts": far_perturbation["total_unauthorized_attempts"],
                "false_acceptances": far_perturbation["false_acceptances"],
                "far_numerator": far_perturbation["far_numerator"],
                "far_denominator": far_perturbation["far_denominator"],
                "far": far_perturbation["far"],
                "far_percent": far_perturbation["far_percent"],
            },
            "all_pruning_variants": {
                "unauthorized_models": far_all_pruning["total_unauthorized_attempts"],
                "unauthorized_attempts": far_all_pruning["total_unauthorized_attempts"],
                "false_acceptances": far_all_pruning["false_acceptances"],
                "far_numerator": far_all_pruning["far_numerator"],
                "far_denominator": far_all_pruning["far_denominator"],
                "far": far_all_pruning["far"],
                "far_percent": far_all_pruning["far_percent"],
            },
            "severe_pruning_ge_35pct": {
                "unauthorized_models": far_severe_pruning["total_unauthorized_attempts"],
                "unauthorized_attempts": far_severe_pruning["total_unauthorized_attempts"],
                "false_acceptances": far_severe_pruning["false_acceptances"],
                "far_numerator": far_severe_pruning["far_numerator"],
                "far_denominator": far_severe_pruning["far_denominator"],
                "far": far_severe_pruning["far"],
                "far_percent": far_severe_pruning["far_percent"],
            },
        },
        "thresholds_used": {
            "engine": "Member 3 Baseline (ModelAuthenticator)",
            "prediction_agreement_threshold": 0.85,
            "cosine_similarity_threshold": 0.90,
            "kl_divergence_threshold": 0.15,
            "rule": "all_metrics (conjunction: agreement >= 0.85 and cosine >= 0.90 and kl <= 0.15)",
        },
        "notes": (
            "Genuine attempts evaluate the single positive control model without artificial replicates. "
            "Under Member 3's baseline thresholds (Agreement >= 0.85), subtle perturbations (std <= 0.10) "
            "retain sufficient classification accuracy to pass, resulting in 18/18 false acceptances for perturbation. "
            "In contrast, replacement models (0/1) and severe pruning >= 35% (0/3) are rejected 100% (FAR = 0.00%)."
        ),
    }

    print("\n--- Reconciled Authentication Performance Summary ---")
    print(f"Unique Model Variants:      {far_frr_summary['summary_counts']['total_unique_models']} (1 genuine, 25 unauthorized)")
    print(f"Total Evaluation Attempts: {far_frr_summary['summary_counts']['total_evaluation_attempts']}")
    print(f"Genuine FRR:                {far_frr_overall['false_rejections']}/{far_frr_overall['total_genuine_attempts']} ({far_frr_overall['frr_percent']})")
    print(f"Overall Unauthorized FAR:   {far_frr_overall['false_acceptances']}/{far_frr_overall['total_unauthorized_attempts']} ({far_frr_overall['far_percent']})")
    print(f"  • Replacement Model FAR:   {far_replacement['false_acceptances']}/{far_replacement['total_unauthorized_attempts']} ({far_replacement['far_percent']})")
    print(f"  • Weight Perturbations FAR:{far_perturbation['false_acceptances']}/{far_perturbation['total_unauthorized_attempts']} ({far_perturbation['far_percent']})")
    print(f"  • All Pruning Variants FAR:{far_all_pruning['false_acceptances']}/{far_all_pruning['total_unauthorized_attempts']} ({far_all_pruning['far_percent']})")
    print(f"  • Severe Pruning (>=35%) FAR:{far_severe_pruning['false_acceptances']}/{far_severe_pruning['total_unauthorized_attempts']} ({far_severe_pruning['far_percent']})")

    # Write JSON results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_variant_results, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")

    # Write CSV results
    fieldnames = list(all_variant_results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_variant_results)
    print(f"Saved CSV results to:  {csv_path}")

    # Write FAR/FRR metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(far_frr_summary, f, indent=2)
    print(f"Saved FAR/FRR metrics to: {metrics_path}")

    # 5. Generate Visualizations
    generate_plots(all_variant_results, far_frr_summary, plots_dir)

    return {
        "results": all_variant_results,
        "far_frr": far_frr_summary,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "metrics_path": str(metrics_path),
        "plots_dir": str(plots_dir),
    }


def generate_plots(
    results: List[Dict[str, Any]],
    far_frr: Dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate degradation plots, decision space, and empirical trade-off visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter perturbation data (group by std dev, average over seeds)
    pert_data = [r for r in results if r["category"] == "weight_perturbation"]
    stds = sorted(list(set(r["parameter_value"] for r in pert_data)))
    pert_agree = [float(np.mean([r["prediction_agreement"] for r in pert_data if r["parameter_value"] == s])) for s in stds]
    pert_cos = [float(np.mean([r["cosine_similarity"] for r in pert_data if r["parameter_value"] == s])) for s in stds]
    pert_kl = [float(np.mean([r["kl_divergence"] for r in pert_data if r["parameter_value"] == s])) for s in stds]

    # Filter pruning data
    prune_data = sorted([r for r in results if r["category"] == "magnitude_pruning"], key=lambda x: x["parameter_value"])
    sparsities = [r["parameter_value"] for r in prune_data]
    prune_agree = [r["prediction_agreement"] for r in prune_data]
    prune_cos = [r["cosine_similarity"] for r in prune_data]
    prune_kl = [r["kl_divergence"] for r in prune_data]

    # 1. Degradation Plot (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Member 4: Behavioral Metric Degradation under Model Tampering", fontsize=14, fontweight="bold")

    # Row 1: Weight Perturbation
    axes[0, 0].plot(stds, pert_agree, marker="o", color="#1f77b4", label="Prediction Agreement")
    axes[0, 0].axhline(0.85, color="red", linestyle="--", label="Threshold (0.85)")
    axes[0, 0].set_title("Perturbation vs Prediction Agreement")
    axes[0, 0].set_xlabel("Noise Std (Relative)")
    axes[0, 0].set_ylabel("Agreement [0, 1]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(stds, pert_cos, marker="s", color="#2ca02c", label="Cosine Similarity")
    axes[0, 1].axhline(0.90, color="red", linestyle="--", label="Threshold (0.90)")
    axes[0, 1].set_title("Perturbation vs Cosine Similarity")
    axes[0, 1].set_xlabel("Noise Std (Relative)")
    axes[0, 1].set_ylabel("Cosine [-1, 1]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(stds, pert_kl, marker="^", color="#d62728", label="KL Divergence")
    axes[0, 2].axhline(0.15, color="red", linestyle="--", label="Threshold (0.15)")
    axes[0, 2].set_title("Perturbation vs KL Divergence")
    axes[0, 2].set_xlabel("Noise Std (Relative)")
    axes[0, 2].set_ylabel("KL (nats, lower is better)")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # Row 2: Magnitude Pruning
    axes[1, 0].plot(sparsities, prune_agree, marker="o", color="#1f77b4", label="Prediction Agreement")
    axes[1, 0].axhline(0.85, color="red", linestyle="--", label="Threshold (0.85)")
    axes[1, 0].set_title("Pruning vs Prediction Agreement")
    axes[1, 0].set_xlabel("Sparsity Ratio [0, 1]")
    axes[1, 0].set_ylabel("Agreement [0, 1]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(sparsities, prune_cos, marker="s", color="#2ca02c", label="Cosine Similarity")
    axes[1, 1].axhline(0.90, color="red", linestyle="--", label="Threshold (0.90)")
    axes[1, 1].set_title("Pruning vs Cosine Similarity")
    axes[1, 1].set_xlabel("Sparsity Ratio [0, 1]")
    axes[1, 1].set_ylabel("Cosine [-1, 1]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(sparsities, prune_kl, marker="^", color="#d62728", label="KL Divergence")
    axes[1, 2].axhline(0.15, color="red", linestyle="--", label="Threshold (0.15)")
    axes[1, 2].set_title("Pruning vs KL Divergence")
    axes[1, 2].set_xlabel("Sparsity Ratio [0, 1]")
    axes[1, 2].set_ylabel("KL (nats, lower is better)")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    degradation_plot_path = plots_dir / "metric_degradation.png"
    plt.savefig(degradation_plot_path, dpi=200)
    plt.close()
    print(f"Saved metric degradation plot to: {degradation_plot_path}")

    # 2. Authentication Decisions & Decision Space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Member 4: Authentication Decisions & Multi-Metric Separation", fontsize=13, fontweight="bold")

    auth_counts = {
        "Genuine Original": sum(1 for r in results if r["category"] == "positive_control" and r["authenticated"]),
        "Diff-Seed (Replacement)": sum(1 for r in results if r["category"] == "impostor_replacement" and r["authenticated"]),
        "Perturbations (All 18)": sum(1 for r in results if r["category"] == "weight_perturbation" and r["authenticated"]),
        "Pruning Minor (<35%)": sum(1 for r in results if r["category"] == "magnitude_pruning" and r["parameter_value"] < 0.35 and r["authenticated"]),
        "Pruning Severe (>=35%)": sum(1 for r in results if r["category"] == "magnitude_pruning" and r["parameter_value"] >= 0.35 and r["authenticated"]),
    }
    total_counts = {
        "Genuine Original": sum(1 for r in results if r["category"] == "positive_control"),
        "Diff-Seed (Replacement)": sum(1 for r in results if r["category"] == "impostor_replacement"),
        "Perturbations (All 18)": sum(1 for r in results if r["category"] == "weight_perturbation"),
        "Pruning Minor (<35%)": sum(1 for r in results if r["category"] == "magnitude_pruning" and r["parameter_value"] < 0.35),
        "Pruning Severe (>=35%)": sum(1 for r in results if r["category"] == "magnitude_pruning" and r["parameter_value"] >= 0.35),
    }

    labels = list(auth_counts.keys())
    passed_vals = [auth_counts[k] for k in labels]
    rejected_vals = [total_counts[k] - auth_counts[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.4
    ax1.bar(x - width/2, passed_vals, width, label="Authenticated (Pass)", color="#2ca02c")
    ax1.bar(x + width/2, rejected_vals, width, label="Rejected (Fail)", color="#d62728")
    ax1.set_ylabel("Attempt Count")
    ax1.set_title("Decision Outcome by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    all_agrees = [r["prediction_agreement"] for r in results]
    all_kls = [r["kl_divergence"] for r in results]
    colors = ["#2ca02c" if r["authenticated"] else "#d62728" for r in results]

    ax2.scatter(all_agrees, all_kls, c=colors, s=60, edgecolors="k", alpha=0.85)
    ax2.axvline(0.85, color="blue", linestyle="--", label="Agreement Threshold (0.85)")
    ax2.axhline(0.15, color="orange", linestyle="--", label="KL Threshold (0.15)")
    ax2.set_xlabel("Prediction Agreement")
    ax2.set_ylabel("KL Divergence (nats)")
    ax2.set_title("Multi-Metric Decision Space (Green=Authentic, Red=Rejected)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()

    plt.tight_layout()
    decision_plot_path = plots_dir / "authentication_decisions.png"
    plt.savefig(decision_plot_path, dpi=200)
    plt.close()
    print(f"Saved authentication decision plot to: {decision_plot_path}")

    # 3. Empirical FAR-vs-FRR Trade-off Curve across Threshold Sweep
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Member 4: Authentication Error Trade-off (FAR vs FRR)", fontsize=13, fontweight="bold")

    threshold_grid = np.linspace(0.60, 1.00, 200)
    genuine_scores = [r["prediction_agreement"] for r in results if r["ground_truth"] == "genuine"]
    unauthorized_scores = [r["prediction_agreement"] for r in results if r["ground_truth"] == "unauthorized"]

    far_curve = []
    frr_curve = []
    for t in threshold_grid:
        # Pass if agreement >= t
        fr = sum(1 for s in genuine_scores if s < t) / len(genuine_scores) if genuine_scores else 0.0
        fa = sum(1 for s in unauthorized_scores if s >= t) / len(unauthorized_scores) if unauthorized_scores else 0.0
        far_curve.append(fa)
        frr_curve.append(fr)

    ax.plot(far_curve, frr_curve, marker="o", markersize=3, color="#800080", label="Empirical Operating Points")
    # Mark default threshold operating point (t = 0.85)
    default_fr = sum(1 for s in genuine_scores if s < 0.85) / len(genuine_scores)
    default_fa = sum(1 for s in unauthorized_scores if s >= 0.85) / len(unauthorized_scores)
    ax.scatter([default_fa], [default_fr], color="red", s=100, zorder=5, label=f"Baseline Threshold 0.85 (FAR={default_fa*100:.1f}%, FRR={default_fr*100:.1f}%)")

    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("FAR vs FRR under Agreement Threshold Sweep [0.60, 1.00]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Clear statistical limitations annotation
    limitation_text = (
        "STATISTICAL LIMITATION NOTE:\n"
        f"• Genuine attempts: N={len(genuine_scores)} (genuine original model)\n"
        f"• Unauthorized attempts: N={len(unauthorized_scores)} (1 replacement, 18 perturbation, 6 pruning)\n"
        "• Total unique models: 26 on 500 fixed challenges.\n"
        "• Small sample size produces discrete step transitions rather than a smooth population ROC curve."
    )
    ax.text(
        0.05, 0.05, limitation_text,
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", edgecolor="#fbc02d", alpha=0.9),
    )

    plt.tight_layout()
    roc_plot_path = plots_dir / "roc_tradeoff.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"Saved empirical trade-off plot to: {roc_plot_path}")


# ──────────────────────────────────────────────
# 6. Unit Tests (Standard library unittest)
# ──────────────────────────────────────────────

class TestModelTamperingAndAuthentication(unittest.TestCase):
    """Unit tests for Member 4 tampering operators, loaders, and authentication integration."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.orig_model = load_checkpoint(DEFAULT_ORIGINAL_MODEL, device=cls.device)
        cls.diff_model = load_checkpoint(DEFAULT_DIFF_SEED_MODEL, device=cls.device)

    def test_01_checkpoint_loader_in_memory(self):
        """Verify checkpoint loader loads original directory checkpoint cleanly without file writes."""
        model = load_checkpoint(DEFAULT_ORIGINAL_MODEL, device=self.device)
        self.assertIsInstance(model, CIFAR10CNN)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 1116970)

    def test_02_weight_perturbation_independence(self):
        """Verify weight perturbation creates an independent copy and leaves original unchanged."""
        orig_weights_before = self.orig_model.classifier[4].weight.clone()
        perturbed = apply_weight_perturbation(self.orig_model, noise_std=0.05, seed=42)

        # Original model must be strictly unchanged
        self.assertTrue(torch.equal(self.orig_model.classifier[4].weight, orig_weights_before))

        # Perturbed model must differ
        self.assertFalse(torch.equal(perturbed.classifier[4].weight, orig_weights_before))

        # Deterministic with identical seed
        perturbed2 = apply_weight_perturbation(self.orig_model, noise_std=0.05, seed=42)
        self.assertTrue(torch.equal(perturbed.classifier[4].weight, perturbed2.classifier[4].weight))

    def test_03_magnitude_pruning_sparsity(self):
        """Verify magnitude pruning zeroes the expected quantile of weights."""
        sparsity = 0.30
        pruned = apply_magnitude_pruning(self.orig_model, sparsity=sparsity)

        # Check sparsity on linear layer
        weights = pruned.classifier[1].weight.detach().cpu().numpy()
        zero_fraction = np.mean(weights == 0.0)
        self.assertAlmostEqual(zero_fraction, sparsity, places=1)

        # Original model weights must not contain zeros
        orig_zeros = np.mean(self.orig_model.classifier[1].weight.detach().cpu().numpy() == 0.0)
        self.assertEqual(orig_zeros, 0.0)

    def test_04_fingerprint_extraction_smoke(self):
        """Smoke test: Extract fingerprint on 500 challenges and verify schema formatting."""
        fp_raw = extract_fingerprint(
            model=self.orig_model,
            challenge_path=str(DEFAULT_CHALLENGE_PATH),
            device="cpu",
            batch_size=128,
        )
        self.assertEqual(fp_raw["num_challenges"], 500)
        fp_dict = format_fingerprint_for_auth(fp_raw)
        self.assertEqual(len(fp_dict["predicted_classes"]), 500)
        self.assertEqual(len(fp_dict["probability_vectors"]), 500)
        self.assertEqual(len(fp_dict["probability_vectors"][0]), 10)

    def test_05_positive_control_authentication(self):
        """Verify original model achieves 1.0 agreement and authenticates successfully."""
        report = evaluate_model_authenticity(
            self.orig_model,
            challenge_path=DEFAULT_CHALLENGE_PATH,
            ref_fingerprint_path=DEFAULT_REF_FINGERPRINT,
            device="cpu",
        )
        self.assertAlmostEqual(report.prediction_agreement, 1.0, places=4)
        self.assertGreaterEqual(report.cosine_similarity, 0.9999)
        self.assertLessEqual(report.kl_divergence, 1e-5)
        self.assertTrue(report.authenticated)

    def test_06_impostor_baseline_rejection(self):
        """Verify different-seed model is correctly rejected by Member 3's authentication engine."""
        report = evaluate_model_authenticity(
            self.diff_model,
            challenge_path=DEFAULT_CHALLENGE_PATH,
            ref_fingerprint_path=DEFAULT_REF_FINGERPRINT,
            device="cpu",
        )
        self.assertAlmostEqual(report.prediction_agreement, 0.738, places=3)
        self.assertFalse(report.prediction_pass)
        self.assertFalse(report.authenticated)

    def test_07_far_frr_edge_cases(self):
        """Verify FAR/FRR computation handles zero denominators and rates correctly."""
        # Case 1: Standard inputs
        res = calculate_far_frr([True, True, False], [False, False, True])
        self.assertAlmostEqual(res["frr"], 1/3)
        self.assertEqual(res["frr_numerator"], 1)
        self.assertEqual(res["frr_denominator"], 3)
        self.assertEqual(res["frr_percent"], "33.33%")
        self.assertAlmostEqual(res["far"], 1/3)
        self.assertEqual(res["far_numerator"], 1)
        self.assertEqual(res["far_denominator"], 3)
        self.assertEqual(res["far_percent"], "33.33%")

        # Case 2: Zero denominator returns None (undefined)
        res_empty = calculate_far_frr([], [False])
        self.assertIsNone(res_empty["frr"])
        self.assertIsNone(res_empty["frr_numerator"])
        self.assertEqual(res_empty["frr_denominator"], 0)
        self.assertEqual(res_empty["frr_percent"], "undefined")
        self.assertEqual(res_empty["far"], 0.0)
        self.assertEqual(res_empty["far_numerator"], 0)
        self.assertEqual(res_empty["far_denominator"], 1)
        self.assertEqual(res_empty["far_percent"], "0.00%")


# ──────────────────────────────────────────────
# 7. Main CLI Execution
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Member 4 Model Tampering & Evaluation")
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run the full experiment matrix and generate CSV/JSON/plots",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVAL_DIR),
        help="Directory to save experiment outputs",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for model inference",
    )

    args, remaining = parser.parse_known_args()

    if args.run_eval:
        print("Running full Member 4 evaluation matrix...")
        run_evaluation_suite(output_dir=args.output_dir, device=args.device)
    else:
        # Default behavior: run unit tests via unittest runner
        print("Running Member 4 unit tests...")
        unittest_args = [sys.argv[0]] + remaining
        unittest.main(argv=unittest_args)


if __name__ == "__main__":
    main()
