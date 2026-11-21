"""
Systematic Authentication Experiments Runner
============================================
Evaluates 10 distinct model conditions against the registered original CNN
using behavioral challenge–response authentication.

Test Cases:
  E1 : Exact original model
  E2 : Original model loaded and re-saved (serialization invariance)
  E3 : Independently trained model (same architecture, distinct training trajectory)
  E4 : 1% weights modified (small perturbation tampering)
  E5 : 5% weights modified (moderate perturbation tampering)
  E6 : 10% weights modified (substantial perturbation tampering)
  E7 : 25% weights modified (heavy perturbation tampering)
  E8 : 50% weights modified (severe perturbation tampering)
  E9 : Randomly initialized model (same architecture, untrained)
  E10: Selected layers modified (head replacement / transfer tampering)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Ensure project root is importable
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
)
from src.authentication.authenticator import (
    AuthenticationThresholds,
    ModelAuthenticator,
    validate_fingerprint_alignment,
)

MODELS_EXP_DIR = PROJECT_ROOT / "models" / "experiments"
RESULTS_EXP_DIR = PROJECT_ROOT / "results" / "experiments"


# ──────────────────────────────────────────────
# 1. Model Generation & Perturbation Utilities
# ──────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model state dict to disk safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_independent_model(
    device: torch.device,
    data_dir: str = "./data",
    seed: int = 101,
    num_batches: int = 100,
) -> CIFAR10CNN:
    """Train an independent CIFAR10CNN on CIFAR-10 data with a different seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def perturb_weights(
    base_model: CIFAR10CNN,
    fraction: float,
    seed: int,
) -> CIFAR10CNN:
    """Create a perturbed copy of base_model modifying a given fraction of weights."""
    torch.manual_seed(seed)
    perturbed_model = copy.deepcopy(base_model)

    with torch.no_grad():
        for param in perturbed_model.parameters():
            mask = torch.rand_like(param.data) < fraction
            if mask.any():
                # Add zero-mean Gaussian noise scaled to the layer's standard deviation
                param_std = param.data.std().item()
                if param_std == 0.0 or np.isnan(param_std):
                    param_std = 0.05
                noise = torch.randn_like(param.data[mask]) * param_std
                param.data[mask] += noise

    perturbed_model.eval()
    return perturbed_model


def modify_selected_layers(
    base_model: CIFAR10CNN,
    seed: int = 1010,
) -> CIFAR10CNN:
    """Reinitialize the final output linear classifier (head replacement attack)."""
    torch.manual_seed(seed)
    modified_model = copy.deepcopy(base_model)

    # Re-initialize classifier[4] (Linear 256 -> 10)
    new_head = nn.Linear(256, 10)
    with torch.no_grad():
        modified_model.classifier[4].weight.copy_(new_head.weight)
        modified_model.classifier[4].bias.copy_(new_head.bias)

    modified_model.eval()
    return modified_model


# ──────────────────────────────────────────────
# 2. Experiment Setup & Execution
# ──────────────────────────────────────────────

EXPERIMENT_SPECS = [
    {
        "id": "E1",
        "name": "Exact Original Model",
        "category": "Legitimate",
        "expected_auth": True,
        "description": "Unmodified original model loaded directly from checkpoint",
    },
    {
        "id": "E2",
        "name": "Loaded & Re-saved Model",
        "category": "Legitimate",
        "expected_auth": True,
        "description": "Original model deserialized and re-serialized (testing serialization invariance)",
    },
    {
        "id": "E3",
        "name": "Independently Trained Model",
        "category": "Attack / Substitute",
        "expected_auth": False,
        "description": "Same architecture independently trained on CIFAR-10 from scratch (seed=101)",
    },
    {
        "id": "E4",
        "name": "1% Weights Modified",
        "category": "Attack / Tampering",
        "expected_auth": False,
        "description": "Original weights with 1% random parameter elements perturbed with Gaussian noise",
    },
    {
        "id": "E5",
        "name": "5% Weights Modified",
        "category": "Attack / Tampering",
        "expected_auth": False,
        "description": "Original weights with 5% random parameter elements perturbed with Gaussian noise",
    },
    {
        "id": "E6",
        "name": "10% Weights Modified",
        "category": "Attack / Tampering",
        "expected_auth": False,
        "description": "Original weights with 10% random parameter elements perturbed with Gaussian noise",
    },
    {
        "id": "E7",
        "name": "25% Weights Modified",
        "category": "Attack / Tampering",
        "expected_auth": False,
        "description": "Original weights with 25% random parameter elements perturbed with Gaussian noise",
    },
    {
        "id": "E8",
        "name": "50% Weights Modified",
        "category": "Attack / Tampering",
        "expected_auth": False,
        "description": "Original weights with 50% random parameter elements perturbed with Gaussian noise",
    },
    {
        "id": "E9",
        "name": "Randomly Initialized Model",
        "category": "Attack / Impostor",
        "expected_auth": False,
        "description": "Same architecture with random untrained weights (sanity negative control)",
    },
    {
        "id": "E10",
        "name": "Selected Layers Modified",
        "category": "Attack / Head Replacement",
        "expected_auth": False,
        "description": "Original model with classification head (Linear 256->10) reinitialized",
    },
]


def prepare_models(
    original_model: CIFAR10CNN,
    device: torch.device,
    exp_dir: Path = MODELS_EXP_DIR,
) -> Dict[str, Path]:
    """Generate and save all 10 experiment models."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    print("\n" + "=" * 70)
    print("  STEP 1: PREPARING EXPERIMENTAL CANDIDATE MODELS")
    print("=" * 70)

    # E1: Exact original copy
    p1 = exp_dir / "e1_exact_original.pth"
    save_checkpoint(original_model, p1)
    paths["E1"] = p1
    print(f"[+] E1  -> Saved copy of original model to: {p1.name}")

    # E2: Load and re-save
    p2 = exp_dir / "e2_loaded_saved.pth"
    loaded_orig = CIFAR10CNN()
    loaded_orig.load_state_dict(torch.load(p1, map_location=device, weights_only=True))
    save_checkpoint(loaded_orig, p2)
    paths["E2"] = p2
    print(f"[+] E2  -> Saved re-serialized model to: {p2.name}")

    # E3: Independently trained model
    p3 = exp_dir / "e3_independently_trained.pth"
    print(f"[*] E3  -> Training independent model on CIFAR-10 data (seed=101)...")
    t0 = time.time()
    indep_model = train_independent_model(device=device, seed=101, num_batches=100)
    save_checkpoint(indep_model, p3)
    paths["E3"] = p3
    print(f"[+] E3  -> Trained and saved independent model in {time.time()-t0:.2f}s to: {p3.name}")

    # E4-E8: Weight perturbations (1%, 5%, 10%, 25%, 50%)
    fractions = [("E4", 0.01), ("E5", 0.05), ("E6", 0.10), ("E7", 0.25), ("E8", 0.50)]
    for eid, frac in fractions:
        p = exp_dir / f"{eid.lower()}_weights_{int(frac*100)}pct_modified.pth"
        m_pert = perturb_weights(original_model, fraction=frac, seed=42 + int(frac * 1000))
        save_checkpoint(m_pert, p)
        paths[eid] = p
        print(f"[+] {eid}  -> Perturbed {frac*100:4.1f}% of weights; saved to: {p.name}")

    # E9: Randomly initialized model
    p9 = exp_dir / "e9_random_init.pth"
    torch.manual_seed(777)
    m_rand = CIFAR10CNN()
    save_checkpoint(m_rand, p9)
    paths[eid_rand := "E9"] = p9
    print(f"[+] E9  -> Saved randomly initialized model to: {p9.name}")

    # E10: Selected layers modified (head replacement)
    p10 = exp_dir / "e10_selected_layers_modified.pth"
    m_head = modify_selected_layers(original_model, seed=1010)
    save_checkpoint(m_head, p10)
    paths["E10"] = p10
    print(f"[+] E10 -> Reinitialized classification head; saved to: {p10.name}")

    return paths


def run_experiments(
    original_model_path: Path = DEFAULT_MODEL_PATH,
    challenge_path: Path = DEFAULT_CHALLENGE_PATH,
    reference_fp_path: Path = DEFAULT_REF_FP_PATH,
    output_dir: Path = RESULTS_EXP_DIR,
    device_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Execute all 10 experiments systematically."""
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SYSTEMATIC AUTHENTICATION EXPERIMENTS (E1 - E10)")
    print("  Deep Learning Model Authentication Using Behavioral Challenge-Response")
    print("=" * 80)
    print(f"[+] Compute device: {device}")

    # Load original model safely
    print(f"[+] Loading original legitimate model from: {original_model_path}")
    orig_model = load_model(str(original_model_path), device)

    # Load reference fingerprint
    print(f"[+] Loading registered reference fingerprint from: {reference_fp_path}")
    ref_fp = torch.load(reference_fp_path, map_location="cpu", weights_only=False)
    num_challenges = ref_fp["num_challenges"]
    print(f"[+] Reference fingerprint loaded ({num_challenges} challenges).")

    # Generate experiment candidate models
    model_paths = prepare_models(orig_model, device=device)

    # Initialize authenticator with standard baseline thresholds
    thresholds = AuthenticationThresholds(
        cosine_threshold=0.90,
        prediction_threshold=0.85,
        kl_threshold=0.15,
    )
    authenticator = ModelAuthenticator(thresholds)

    print("\n" + "=" * 70)
    print("  STEP 2: RUNNING AUTHENTICATION PROTOCOL")
    print("=" * 70)

    results: List[Dict[str, Any]] = []

    for spec in EXPERIMENT_SPECS:
        eid = spec["id"]
        model_path = model_paths[eid]
        error_msg = None
        t0 = time.time()

        try:
            # Load candidate model in eval mode
            candidate_model = CIFAR10CNN()
            candidate_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            candidate_model.to(device)
            candidate_model.eval()

            # Extract candidate fingerprint
            candidate_fp = extract_fingerprint(
                model=candidate_model,
                challenge_path=str(challenge_path),
                device=device,
                batch_size=64,
            )

            # Validate alignment
            validate_fingerprint_alignment(ref_fp, candidate_fp)

            # Compute metrics
            metrics = authenticator.compare_metrics(ref_fp, candidate_fp)
            report = authenticator.evaluate_thresholds(metrics)

            elapsed = time.time() - t0

            # Determine security classification outcome
            is_legitimate = spec["category"] == "Legitimate"
            is_auth = report.authenticated

            if is_legitimate and is_auth:
                security_outcome = "True Acceptance (Legitimate Authenticated)"
            elif is_legitimate and not is_auth:
                security_outcome = "False Rejection (Legitimate Rejected)"
            elif not is_legitimate and not is_auth:
                security_outcome = "True Rejection (Attack Defeated)"
            else:
                security_outcome = "False Acceptance (Attack Succeeded / Evasion)"

            rec = {
                "experiment_id": eid,
                "experiment_name": spec["name"],
                "category": spec["category"],
                "model_filename": model_path.name,
                "model_path": str(model_path),
                "description": spec["description"],
                "prediction_agreement": report.prediction_agreement,
                "cosine_similarity": report.cosine_similarity,
                "kl_divergence": report.kl_divergence,
                "matching_predictions": report.matching_predictions,
                "mismatching_predictions": report.mismatching_predictions,
                "num_challenges": report.num_challenges,
                "prediction_pass": report.prediction_pass,
                "cosine_pass": report.cosine_pass,
                "kl_pass": report.kl_pass,
                "decision": "AUTHENTIC" if report.authenticated else "REJECTED",
                "authenticated": report.authenticated,
                "expected_auth": spec["expected_auth"],
                "security_outcome": security_outcome,
                "execution_error": None,
                "evaluation_time_sec": round(elapsed, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            elapsed = time.time() - t0
            rec = {
                "experiment_id": eid,
                "experiment_name": spec["name"],
                "category": spec["category"],
                "model_filename": model_path.name,
                "model_path": str(model_path),
                "description": spec["description"],
                "prediction_agreement": None,
                "cosine_similarity": None,
                "kl_divergence": None,
                "matching_predictions": None,
                "mismatching_predictions": None,
                "num_challenges": num_challenges,
                "prediction_pass": False,
                "cosine_pass": False,
                "kl_pass": False,
                "decision": "ERROR",
                "authenticated": False,
                "expected_auth": spec["expected_auth"],
                "security_outcome": "Execution Failure",
                "execution_error": str(e),
                "evaluation_time_sec": round(elapsed, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        results.append(rec)
        dec_tag = f"[{rec['decision']}]"
        agr_str = f"{rec['prediction_agreement']*100:5.1f}%" if rec['prediction_agreement'] is not None else " N/A "
        cos_str = f"{rec['cosine_similarity']:.4f}" if rec['cosine_similarity'] is not None else " N/A "
        kl_str = f"{rec['kl_divergence']:.4f}" if rec['kl_divergence'] is not None else " N/A "
        print(f"  {eid:<3} | {spec['name']:<30} | {dec_tag:<11} | Agr: {agr_str} | Cos: {cos_str} | KL: {kl_str} | {rec['security_outcome']}")

    # Save results to JSON and CSV
    save_experiment_reports(results, output_dir)

    # Generate charts
    generate_experiment_charts(results, thresholds, output_dir)

    # Print ASCII summary table
    print_comparison_table(results, thresholds)

    return results


# ──────────────────────────────────────────────
# 3. Persistence & Reporting
# ──────────────────────────────────────────────

def save_experiment_reports(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save full experiment records in JSON and CSV formats."""
    json_path = output_dir / "experiment_results.json"
    csv_path = output_dir / "experiment_results.csv"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved detailed JSON report to : {json_path}")

    # Save CSV
    fieldnames = [
        "experiment_id",
        "experiment_name",
        "category",
        "decision",
        "authenticated",
        "expected_auth",
        "security_outcome",
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
        "model_filename",
        "execution_error",
        "description",
        "timestamp",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[+] Saved summary CSV report to    : {csv_path}")


def print_comparison_table(results: List[Dict[str, Any]], thresholds: AuthenticationThresholds) -> None:
    """Print an aligned ASCII comparison table for all 10 experiments."""
    print("\n" + "=" * 115)
    print("SYSTEMATIC EXPERIMENT EVALUATION TABLE")
    print("=" * 115)
    header = (
        f"{'ID':<4} | {'Experiment Name':<28} | {'Category':<18} | {'Decision':<10} | "
        f"{'Agreement':<10} | {'Cosine Sim':<10} | {'KL Div':<8} | {'Security Outcome':<28}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        eid = r["experiment_id"]
        name = r["experiment_name"][:28]
        cat = r["category"][:18]
        dec = r["decision"]
        agr = f"{r['prediction_agreement']*100:5.1f}%" if r["prediction_agreement"] is not None else "N/A"
        cos = f"{r['cosine_similarity']:.4f}" if r["cosine_similarity"] is not None else "N/A"
        kl = f"{r['kl_divergence']:.4f}" if r["kl_divergence"] is not None else "N/A"
        outcome = r["security_outcome"][:28]

        print(f"{eid:<4} | {name:<28} | {cat:<18} | {dec:<10} | {agr:<10} | {cos:<10} | {kl:<8} | {outcome:<28}")

    print("=" * 115)
    print(
        f"Baseline Thresholds: Prediction Agreement >= {thresholds.prediction_threshold:.0%} | "
        f"Cosine Similarity >= {thresholds.cosine_threshold:.2f} | "
        f"KL Divergence <= {thresholds.kl_threshold:.2f}"
    )
    print("=" * 115)


# ──────────────────────────────────────────────
# 4. Publication-Quality Visualization
# ──────────────────────────────────────────────

def generate_experiment_charts(
    results: List[Dict[str, Any]],
    thresholds: AuthenticationThresholds,
    output_dir: Path,
) -> None:
    """Generate high-resolution comparison graphs across all experiments."""
    ids = [r["experiment_id"] for r in results]
    agreements = [r["prediction_agreement"] * 100 for r in results]
    cosines = [r["cosine_similarity"] for r in results]
    kl_divs = [r["kl_divergence"] for r in results]
    decisions = [r["decision"] for r in results]

    # Color palette based on decision
    bar_colors = ["#2ca02c" if d == "AUTHENTIC" else "#d62728" for d in decisions]

    # ── Figure 1: 3-Panel Metric Comparison Dashboard ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1. Prediction Agreement
    ax1 = axes[0]
    bars1 = ax1.bar(ids, agreements, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
    ax1.axhline(
        thresholds.prediction_threshold * 100,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({thresholds.prediction_threshold*100:.0f}%)",
    )
    ax1.set_ylabel("Prediction Agreement (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.set_title("Behavioral Authentication Metrics Across Systematic Experiments (E1 - E10)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 2. Cosine Similarity
    ax2 = axes[1]
    bars2 = ax2.bar(ids, cosines, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
    ax2.axhline(
        thresholds.cosine_threshold,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({thresholds.cosine_threshold:.2f})",
    )
    ax2.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.08)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 3. KL Divergence
    ax3 = axes[2]
    bars3 = ax3.bar(ids, kl_divs, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
    ax3.axhline(
        thresholds.kl_threshold,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({thresholds.kl_threshold:.2f})",
    )
    ax3.set_ylabel("KL Divergence (nats)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Experiment ID", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.03, f"{yval:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    chart1_path = output_dir / "experiment_metric_comparison.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close(fig)
    print(f"[+] Saved metric comparison chart to: {chart1_path}")

    # ── Figure 2: Weight Perturbation Sensitivity Curve (E1, E4, E5, E6, E7, E8) ──
    pert_cases = [
        (0.0, next(r for r in results if r["experiment_id"] == "E1")),
        (1.0, next(r for r in results if r["experiment_id"] == "E4")),
        (5.0, next(r for r in results if r["experiment_id"] == "E5")),
        (10.0, next(r for r in results if r["experiment_id"] == "E6")),
        (25.0, next(r for r in results if r["experiment_id"] == "E7")),
        (50.0, next(r for r in results if r["experiment_id"] == "E8")),
    ]

    p_pcts = [p[0] for p in pert_cases]
    p_agrs = [p[1]["prediction_agreement"] * 100 for p in pert_cases]
    p_coss = [p[1]["cosine_similarity"] for p in pert_cases]
    p_kls = [p[1]["kl_divergence"] for p in pert_cases]

    fig2, (p_ax1, p_ax2, p_ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Agreement Curve
    p_ax1.plot(p_pcts, p_agrs, "o-", color="#1f77b4", linewidth=2.5, markersize=7)
    p_ax1.axhline(thresholds.prediction_threshold * 100, color="red", linestyle="--", label="Threshold (85%)")
    p_ax1.set_xlabel("Weights Modified (%)", fontweight="bold")
    p_ax1.set_ylabel("Prediction Agreement (%)", fontweight="bold")
    p_ax1.set_title("Agreement vs Perturbation", fontweight="bold")
    p_ax1.set_ylim(0, 105)
    p_ax1.grid(True, linestyle=":", alpha=0.6)
    p_ax1.legend()

    # Cosine Curve
    p_ax2.plot(p_pcts, p_coss, "s-", color="#2ca02c", linewidth=2.5, markersize=7)
    p_ax2.axhline(thresholds.cosine_threshold, color="red", linestyle="--", label="Threshold (0.90)")
    p_ax2.set_xlabel("Weights Modified (%)", fontweight="bold")
    p_ax2.set_ylabel("Cosine Similarity", fontweight="bold")
    p_ax2.set_title("Cosine Similarity vs Perturbation", fontweight="bold")
    p_ax2.set_ylim(0.4, 1.05)
    p_ax2.grid(True, linestyle=":", alpha=0.6)
    p_ax2.legend()

    # KL Curve
    p_ax3.plot(p_pcts, p_kls, "^-", color="#ff7f0e", linewidth=2.5, markersize=7)
    p_ax3.axhline(thresholds.kl_threshold, color="red", linestyle="--", label="Threshold (0.15)")
    p_ax3.set_xlabel("Weights Modified (%)", fontweight="bold")
    p_ax3.set_ylabel("KL Divergence (nats)", fontweight="bold")
    p_ax3.set_title("KL Divergence vs Perturbation", fontweight="bold")
    p_ax3.grid(True, linestyle=":", alpha=0.6)
    p_ax3.legend()

    plt.tight_layout()
    chart2_path = output_dir / "perturbation_sensitivity_curves.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close(fig2)
    print(f"[+] Saved perturbation sensitivity curves to: {chart2_path}")


# ──────────────────────────────────────────────
# 5. CLI Entry Point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic Model Authentication Experiments (E1-E10)")
    parser.add_argument("--original_model", default=str(DEFAULT_MODEL_PATH), help="Path to original legitimate model")
    parser.add_argument("--challenges", default=str(DEFAULT_CHALLENGE_PATH), help="Path to challenge set")
    parser.add_argument("--reference_fp", default=str(DEFAULT_REF_FP_PATH), help="Path to reference fingerprint")
    parser.add_argument("--output_dir", default=str(RESULTS_EXP_DIR), help="Output directory for reports & plots")
    parser.add_argument("--device", default=None, help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    run_experiments(
        original_model_path=Path(args.original_model),
        challenge_path=Path(args.challenges),
        reference_fp_path=Path(args.reference_fp),
        output_dir=Path(args.output_dir),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

