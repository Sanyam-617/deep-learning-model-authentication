"""
Experimental Improved Detection Runner
======================================
Member 4 – Behavioral Challenge-Response Model Authentication
Phase 9: Stricter Authentication Threshold Evaluation (Proposal Rank 1)

This module implements an isolated, non-destructive experimental comparison
of Member 3's baseline authentication rule against six candidate stricter
threshold configurations using the existing ModelAuthenticator interface.

Preservation guarantees:
- Operates strictly via Member 3's existing AuthenticationThresholds dataclass.
- Does not modify any source code in src/authentication/ or src/fingerprint/.
- Does not overwrite validated baseline artifacts in experiments/attack_evaluation/.
- Saves all new comparison artifacts in experiments/attack_evaluation/improved_detection/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import CIFAR10CNN
from src.fingerprint.fingerprint import extract_fingerprint
from src.authentication.authenticator import (
    AuthenticationMetrics,
    AuthenticationReport,
    AuthenticationThresholds,
    ModelAuthenticator,
)
from src.attacks.attack_tests import (
    DEFAULT_CHALLENGE_PATH,
    DEFAULT_DIFF_SEED_MODEL,
    DEFAULT_ORIGINAL_MODEL,
    DEFAULT_REF_FINGERPRINT,
    apply_magnitude_pruning,
    apply_weight_perturbation,
    calculate_far_frr,
    format_fingerprint_for_auth,
    load_checkpoint,
)

# Output Directory for Phase 9
DEFAULT_IMPROVED_DIR = (
    PROJECT_ROOT / "experiments" / "attack_evaluation" / "improved_detection"
)

# Candidate Threshold Configurations (Phase 9 Specification)
CANDIDATE_CONFIGS: Dict[str, Dict[str, Any]] = OrderedDict([
    (
        "baseline",
        {
            "name": "Baseline (Member 3 Default)",
            "short_name": "Baseline",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.85,
                cosine_threshold=0.90,
                kl_threshold=0.15,
            ),
            "description": "Baseline development defaults: Agr >= 0.85, Cos >= 0.90, KL <= 0.15",
        },
    ),
    (
        "moderate",
        {
            "name": "Moderate",
            "short_name": "Moderate",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.95,
                cosine_threshold=0.95,
                kl_threshold=0.05,
            ),
            "description": "Moderate: Agr >= 0.95, Cos >= 0.95, KL <= 0.05",
        },
    ),
    (
        "balanced",
        {
            "name": "Balanced",
            "short_name": "Balanced",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.98,
                cosine_threshold=0.99,
                kl_threshold=0.01,
            ),
            "description": "Balanced: Agr >= 0.98, Cos >= 0.99, KL <= 0.01",
        },
    ),
    (
        "strict",
        {
            "name": "Strict",
            "short_name": "Strict",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.99,
                cosine_threshold=0.99,
                kl_threshold=0.001,
            ),
            "description": "Strict: Agr >= 0.99, Cos >= 0.99, KL <= 0.001",
        },
    ),
    (
        "high_sensitivity",
        {
            "name": "High-Sensitivity",
            "short_name": "High-Sens",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.99,
                cosine_threshold=0.999,
                kl_threshold=0.0001,
            ),
            "description": "High-Sensitivity: Agr >= 0.99, Cos >= 0.999, KL <= 0.0001",
        },
    ),
    (
        "ultra_sensitive",
        {
            "name": "Ultra-Sensitive",
            "short_name": "Ultra-Sens",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.99,
                cosine_threshold=0.9999,
                kl_threshold=0.00001,
            ),
            "description": "Ultra-Sensitive: Agr >= 0.99, Cos >= 0.9999, KL <= 0.00001",
        },
    ),
    (
        "zero_tolerance",
        {
            "name": "Zero-Tolerance Candidate",
            "short_name": "Zero-Tol",
            "thresholds": AuthenticationThresholds(
                prediction_threshold=0.999,
                cosine_threshold=0.9999,
                kl_threshold=0.000001,
            ),
            "description": "Zero-Tolerance Candidate: Agr >= 0.999, Cos >= 0.9999, KL <= 0.000001",
        },
    ),
])


def run_improved_detection_evaluation(
    output_dir: Union[str, Path] = DEFAULT_IMPROVED_DIR,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Execute live evaluation of 26 model variants against all candidate threshold configurations."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load reference fingerprint
    with open(DEFAULT_REF_FINGERPRINT, "r") as f:
        ref_fp = json.load(f)

    # Base models
    orig_model = load_checkpoint(DEFAULT_ORIGINAL_MODEL, device=device)
    diff_seed_model = load_checkpoint(DEFAULT_DIFF_SEED_MODEL, device=device)

    # Authenticator instance for metrics comparison
    authenticator = ModelAuthenticator()

    print("=" * 80)
    print("PHASE 9: EXPERIMENTAL IMPROVED DETECTION EVALUATION (PROPOSAL RANK 1)")
    print(f"Candidate Configurations: {len(CANDIDATE_CONFIGS)}")
    print(f"Fixed Challenge Set     : {DEFAULT_CHALLENGE_PATH}")
    print(f"Reference Fingerprint   : {DEFAULT_REF_FINGERPRINT}")
    print(f"Output Directory        : {output_dir}")
    print("=" * 80)

    # Build model variants list
    # Each entry: (variant_id, model, category, ground_truth, param_name, param_val, seed)
    variants_to_eval = []

    # 1. Genuine positive control
    variants_to_eval.append((
        "genuine_original",
        orig_model,
        "positive_control",
        "genuine",
        "none",
        0.0,
        None,
    ))

    # 2. Replacement impostor
    variants_to_eval.append((
        "different_seed_model",
        diff_seed_model,
        "impostor_replacement",
        "unauthorized",
        "seed",
        123.0,
        123,
    ))

    # 3. Weight perturbations (6 strengths x 3 seeds = 18 models)
    perturb_strengths = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    seeds = [42, 43, 44]
    for std in perturb_strengths:
        for seed in seeds:
            pert_model = apply_weight_perturbation(orig_model, noise_std=std, seed=seed)
            variants_to_eval.append((
                f"perturb_std_{std}_seed_{seed}",
                pert_model,
                "weight_perturbation",
                "unauthorized",
                "noise_std",
                std,
                seed,
            ))

    # 4. Magnitude pruning (6 sparsity levels = 6 models)
    pruning_levels = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
    for sparsity in pruning_levels:
        pruned_model = apply_magnitude_pruning(orig_model, sparsity=sparsity)
        variants_to_eval.append((
            f"prune_sparsity_{sparsity}",
            pruned_model,
            "magnitude_pruning",
            "unauthorized",
            "sparsity",
            sparsity,
            None,
        ))

    print(f"\nTotal model variants to evaluate: {len(variants_to_eval)} (1 genuine, 25 unauthorized)")

    # Step 1: Extract fingerprints and compute raw AuthenticationMetrics
    print("\n--- Extracting Fingerprints & Computing Behavioral Metrics ---")
    model_records: List[Dict[str, Any]] = []

    for idx, (var_id, model, cat, gt, p_name, p_val, seed) in enumerate(variants_to_eval, 1):
        fp_raw = extract_fingerprint(
            model=model,
            challenge_path=str(DEFAULT_CHALLENGE_PATH),
            device=device,
            batch_size=batch_size,
        )
        fp_formatted = format_fingerprint_for_auth(fp_raw)
        metrics: AuthenticationMetrics = authenticator.compare_metrics(ref_fp, fp_formatted)

        record: Dict[str, Any] = {
            "variant_id": var_id,
            "category": cat,
            "ground_truth": gt,
            "parameter_name": p_name,
            "parameter_value": p_val,
            "seed": seed,
            "prediction_agreement": metrics.prediction_agreement,
            "cosine_similarity": metrics.cosine_similarity,
            "kl_divergence": metrics.kl_divergence,
            "matching_predictions": metrics.matching_predictions,
            "mismatching_predictions": metrics.mismatching_predictions,
            "decisions": {},
        }

        # Step 2: Evaluate decisions across all candidate threshold configurations
        for cfg_key, cfg_info in CANDIDATE_CONFIGS.items():
            thresholds: AuthenticationThresholds = cfg_info["thresholds"]
            report: AuthenticationReport = authenticator.evaluate_thresholds(metrics, thresholds)
            record["decisions"][cfg_key] = {
                "prediction_pass": report.prediction_pass,
                "cosine_pass": report.cosine_pass,
                "kl_pass": report.kl_pass,
                "authenticated": report.authenticated,
                "decision_outcome": (
                    ("true_acceptance" if report.authenticated else "false_rejection")
                    if gt == "genuine"
                    else ("false_acceptance" if report.authenticated else "true_rejection")
                ),
            }

        model_records.append(record)
        print(
            f"[{idx:02d}/26] {var_id:30} | Agree={metrics.prediction_agreement:.4f} "
            f"| Cos={metrics.cosine_similarity:.6f} | KL={metrics.kl_divergence:.6e}"
        )

    # Step 3: Compute Summary Metrics across Candidates
    print("\n--- Competing Configuration Performance Summary ---")
    config_summaries: Dict[str, Any] = {}

    for cfg_key, cfg_info in CANDIDATE_CONFIGS.items():
        th = cfg_info["thresholds"]

        # Decisions
        gen_decisions = [r["decisions"][cfg_key]["authenticated"] for r in model_records if r["ground_truth"] == "genuine"]
        unauth_decisions = [r["decisions"][cfg_key]["authenticated"] for r in model_records if r["ground_truth"] == "unauthorized"]
        rep_decisions = [r["decisions"][cfg_key]["authenticated"] for r in model_records if r["category"] == "impostor_replacement"]
        pert_decisions = [r["decisions"][cfg_key]["authenticated"] for r in model_records if r["category"] == "weight_perturbation"]
        prun_decisions = [r["decisions"][cfg_key]["authenticated"] for r in model_records if r["category"] == "magnitude_pruning"]

        far_frr_overall = calculate_far_frr(gen_decisions, unauth_decisions)
        far_rep = calculate_far_frr(gen_decisions, rep_decisions)
        far_pert = calculate_far_frr(gen_decisions, pert_decisions)
        far_prun = calculate_far_frr(gen_decisions, prun_decisions)

        # Perturbation breakdown by noise std
        pert_by_std = {}
        for std in perturb_strengths:
            sub_decisions = [
                r["decisions"][cfg_key]["authenticated"]
                for r in model_records
                if r["category"] == "weight_perturbation" and r["parameter_value"] == std
            ]
            rate_obj = calculate_far_frr(gen_decisions, sub_decisions)
            pert_by_std[str(std)] = {
                "unauthorized_models": len(sub_decisions),
                "false_acceptances": rate_obj["false_acceptances"],
                "far_numerator": rate_obj["far_numerator"],
                "far_denominator": rate_obj["far_denominator"],
                "far": rate_obj["far"],
                "far_percent": rate_obj["far_percent"],
            }

        # Pruning breakdown by sparsity
        prun_by_sparsity = {}
        for sp in pruning_levels:
            sub_decisions = [
                r["decisions"][cfg_key]["authenticated"]
                for r in model_records
                if r["category"] == "magnitude_pruning" and r["parameter_value"] == sp
            ]
            rate_obj = calculate_far_frr(gen_decisions, sub_decisions)
            prun_by_sparsity[str(sp)] = {
                "unauthorized_models": len(sub_decisions),
                "false_acceptances": rate_obj["false_acceptances"],
                "far_numerator": rate_obj["far_numerator"],
                "far_denominator": rate_obj["far_denominator"],
                "far": rate_obj["far"],
                "far_percent": rate_obj["far_percent"],
            }

        config_summaries[cfg_key] = {
            "name": cfg_info["name"],
            "short_name": cfg_info["short_name"],
            "thresholds": {
                "prediction_threshold": th.prediction_threshold,
                "cosine_threshold": th.cosine_threshold,
                "kl_threshold": th.kl_threshold,
            },
            "description": cfg_info["description"],
            "overall": {
                "genuine_attempts": far_frr_overall["total_genuine_attempts"],
                "false_rejections": far_frr_overall["false_rejections"],
                "frr_numerator": far_frr_overall["frr_numerator"],
                "frr_denominator": far_frr_overall["frr_denominator"],
                "frr": far_frr_overall["frr"],
                "frr_percent": far_frr_overall["frr_percent"],
                "unauthorized_attempts": far_frr_overall["total_unauthorized_attempts"],
                "false_acceptances": far_frr_overall["false_acceptances"],
                "far_numerator": far_frr_overall["far_numerator"],
                "far_denominator": far_frr_overall["far_denominator"],
                "far": far_frr_overall["far"],
                "far_percent": far_frr_overall["far_percent"],
            },
            "category_far": {
                "replacement_model": {
                    "unauthorized_models": far_rep["total_unauthorized_attempts"],
                    "false_acceptances": far_rep["false_acceptances"],
                    "far_numerator": far_rep["far_numerator"],
                    "far_denominator": far_rep["far_denominator"],
                    "far": far_rep["far"],
                    "far_percent": far_rep["far_percent"],
                },
                "weight_perturbations": {
                    "unauthorized_models": far_pert["total_unauthorized_attempts"],
                    "false_acceptances": far_pert["false_acceptances"],
                    "far_numerator": far_pert["far_numerator"],
                    "far_denominator": far_pert["far_denominator"],
                    "far": far_pert["far"],
                    "far_percent": far_pert["far_percent"],
                    "by_noise_strength": pert_by_std,
                },
                "magnitude_pruning": {
                    "unauthorized_models": far_prun["total_unauthorized_attempts"],
                    "false_acceptances": far_prun["false_acceptances"],
                    "far_numerator": far_prun["far_numerator"],
                    "far_denominator": far_prun["far_denominator"],
                    "far": far_prun["far"],
                    "far_percent": far_prun["far_percent"],
                    "by_sparsity": prun_by_sparsity,
                },
            },
        }

        print(
            f"Config {cfg_key:18} | FRR={far_frr_overall['frr_numerator']}/{far_frr_overall['frr_denominator']} "
            f"({far_frr_overall['frr_percent']:>6}) | FAR={far_frr_overall['far_numerator']:2d}/{far_frr_overall['far_denominator']:2d} "
            f"({far_frr_overall['far_percent']:>7}) | Pert={far_pert['false_acceptances']:2d}/18 | Prun={far_prun['false_acceptances']}/6 | Rep={far_rep['false_acceptances']}/1"
        )

    # Step 4: Discrepancy Analysis against Phase 8 Diagnostics
    print("\n--- Verifying Consistency with Phase 8 Diagnostic Projections ---")
    discrepancies: List[Dict[str, Any]] = []
    baseline_json_path = PROJECT_ROOT / "experiments" / "attack_evaluation" / "results_summary.json"
    if baseline_json_path.exists():
        with open(baseline_json_path, "r") as f:
            baseline_records = {r["variant_id"]: r for r in json.load(f)}

        for r in model_records:
            vid = r["variant_id"]
            if vid in baseline_records:
                b_r = baseline_records[vid]
                d_agr = abs(r["prediction_agreement"] - b_r["prediction_agreement"])
                d_cos = abs(r["cosine_similarity"] - b_r["cosine_similarity"])
                d_kl = abs(r["kl_divergence"] - b_r["kl_divergence"])
                if d_agr > 1e-6 or d_cos > 1e-6 or d_kl > 1e-6:
                    discrepancies.append({
                        "variant_id": vid,
                        "diff_agreement": d_agr,
                        "diff_cosine": d_cos,
                        "diff_kl": d_kl,
                    })

    if not discrepancies:
        print("Consistency check PASSED: Reproduced behavioral metrics match Phase 8 records identically (max diff <= 1e-6).")
    else:
        print(f"Notice: Found {len(discrepancies)} metric precision discrepancies between reproduced run and Phase 8 saved records.")

    # Step 5: Save JSON & CSV Outputs
    json_out = output_dir / "comparison_summary.json"
    csv_out = output_dir / "comparison_summary.csv"
    decisions_csv_out = output_dir / "variant_decisions.csv"

    # Save comparison_summary.json
    full_output_data = {
        "metadata": {
            "title": "Phase 9 Experimental Improved Detection Comparison",
            "role": "Member 4 – Model Tampering and Evaluation",
            "methodology": "Proposal Rank 1 – Threshold Recalibration via Existing Authenticator Interface",
            "status": "EXPERIMENTAL_ANALYSIS_NOT_PRODUCTION_THRESHOLDS",
            "total_model_variants": len(model_records),
            "genuine_models": 1,
            "unauthorized_models": 25,
            "num_challenges": 500,
            "discrepancies_with_phase8": discrepancies,
        },
        "configurations": config_summaries,
        "variant_evaluations": model_records,
    }

    with open(json_out, "w") as f:
        json.dump(full_output_data, f, indent=2)
    print(f"\nSaved JSON comparison summary to: {json_out}")

    # Save comparison_summary.csv (Tabular summary of all configs)
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration",
            "Agr_Thresh",
            "Cos_Thresh",
            "KL_Thresh",
            "Genuine_FRR",
            "Overall_FAR",
            "Replacement_FAR",
            "Perturbation_FAR",
            "Pert_s0.001",
            "Pert_s0.005",
            "Pert_s0.010",
            "Pert_s0.020",
            "Pert_s0.050",
            "Pert_s0.100",
            "Pruning_FAR",
            "Prun_0.05",
            "Prun_0.10",
            "Prun_0.20",
            "Prun_0.35",
            "Prun_0.50",
            "Prun_0.75",
        ])
        for cfg_key, summary in config_summaries.items():
            th = summary["thresholds"]
            ov = summary["overall"]
            cat = summary["category_far"]
            p_std = cat["weight_perturbations"]["by_noise_strength"]
            p_sp = cat["magnitude_pruning"]["by_sparsity"]
            writer.writerow([
                summary["name"],
                th["prediction_threshold"],
                th["cosine_threshold"],
                th["kl_threshold"],
                f"{ov['frr_numerator']}/{ov['frr_denominator']} ({ov['frr_percent']})",
                f"{ov['far_numerator']}/{ov['far_denominator']} ({ov['far_percent']})",
                f"{cat['replacement_model']['far_numerator']}/{cat['replacement_model']['far_denominator']} ({cat['replacement_model']['far_percent']})",
                f"{cat['weight_perturbations']['far_numerator']}/{cat['weight_perturbations']['far_denominator']} ({cat['weight_perturbations']['far_percent']})",
                f"{p_std['0.001']['false_acceptances']}/3",
                f"{p_std['0.005']['false_acceptances']}/3",
                f"{p_std['0.01']['false_acceptances']}/3",
                f"{p_std['0.02']['false_acceptances']}/3",
                f"{p_std['0.05']['false_acceptances']}/3",
                f"{p_std['0.1']['false_acceptances']}/3",
                f"{cat['magnitude_pruning']['far_numerator']}/{cat['magnitude_pruning']['far_denominator']} ({cat['magnitude_pruning']['far_percent']})",
                f"{p_sp['0.05']['false_acceptances']}/1",
                f"{p_sp['0.1']['false_acceptances']}/1",
                f"{p_sp['0.2']['false_acceptances']}/1",
                f"{p_sp['0.35']['false_acceptances']}/1",
                f"{p_sp['0.5']['false_acceptances']}/1",
                f"{p_sp['0.75']['false_acceptances']}/1",
            ])
    print(f"Saved CSV comparison summary to: {csv_out}")

    # Save variant_decisions.csv (Model-level decisions for all variants under all configs)
    with open(decisions_csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "Variant_ID",
            "Category",
            "Ground_Truth",
            "Param_Value",
            "Agreement",
            "Cosine_Similarity",
            "KL_Divergence",
        ] + [f"Auth_{k}" for k in CANDIDATE_CONFIGS.keys()]
        writer.writerow(header)
        for r in model_records:
            row = [
                r["variant_id"],
                r["category"],
                r["ground_truth"],
                r["parameter_value"],
                f"{r['prediction_agreement']:.4f}",
                f"{r['cosine_similarity']:.6f}",
                f"{r['kl_divergence']:.6e}",
            ] + [str(r["decisions"][k]["authenticated"]) for k in CANDIDATE_CONFIGS.keys()]
            writer.writerow(row)
    print(f"Saved variant-level decisions CSV to: {decisions_csv_out}")

    # Step 6: Generate Publication-Ready Comparison Plots
    generate_comparison_plots(config_summaries, model_records, plots_dir)

    return full_output_data


def generate_comparison_plots(
    config_summaries: Dict[str, Any],
    model_records: List[Dict[str, Any]],
    plots_dir: Path,
) -> None:
    """Generate side-by-side comparison plots visualizing detection improvements and trade-offs."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    cfg_names = [s["short_name"] for s in config_summaries.values()]
    far_vals = [s["overall"]["far"] * 100 for s in config_summaries.values()]
    frr_vals = [s["overall"]["frr"] * 100 for s in config_summaries.values()]

    # ─────────────────────────────────────────────────────────────
    # Plot 1: FAR vs FRR Trade-off across Configurations
    # ─────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(cfg_names))
    width = 0.35

    color_far = "#d9534f"  # Red
    color_frr = "#28a745"  # Green

    rects1 = ax1.bar(x - width/2, far_vals, width, label="Unauthorized FAR (%)", color=color_far, alpha=0.85, edgecolor="black")
    rects2 = ax1.bar(x + width/2, frr_vals, width, label="Genuine FRR (%)", color=color_frr, alpha=0.85, edgecolor="black")

    ax1.set_ylabel("Error Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Phase 9: False Acceptance (FAR) vs False Rejection (FRR) Trade-Off Across Configurations", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_names, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="darkgreen")

    fig.tight_layout()
    p1 = plots_dir / "far_frr_tradeoff.png"
    fig.savefig(p1)
    plt.close(fig)
    print(f"Generated Plot 1: {p1}")

    # ─────────────────────────────────────────────────────────────
    # Plot 2: Perturbation Detection Rate by Noise Strength
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    stds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    std_strs = [str(s) for s in stds]

    markers = ["o", "s", "^", "D", "v", "<", "p"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (cfg_key, summary) in enumerate(config_summaries.items()):
        by_std = summary["category_far"]["weight_perturbations"]["by_noise_strength"]
        # Detection Rate = 100% - FAR
        detection_rates = [(1.0 - by_std[s]["far"]) * 100 for s in std_strs]
        ax.plot(
            np.arange(len(stds)),
            detection_rates,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=f"{summary['short_name']}",
            linewidth=2,
            markersize=7,
        )

    ax.set_xticks(np.arange(len(stds)))
    ax.set_xticklabels([f"σ={s}" for s in stds], fontsize=10, fontweight="bold")
    ax.set_xlabel("Gaussian Noise Strength (Relative Standard Deviation σ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tampered Model Detection Rate (%) [100% - FAR]", fontsize=11, fontweight="bold")
    ax.set_title("Detection Rate of Gaussian Weight-Perturbed Models Across Noise Strengths", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    p2 = plots_dir / "perturbation_detection_by_sigma.png"
    fig.savefig(p2)
    plt.close(fig)
    print(f"Generated Plot 2: {p2}")

    # ─────────────────────────────────────────────────────────────
    # Plot 3: Category FAR Comparison across Candidate Configurations
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    x = np.arange(len(cfg_names))
    width = 0.25

    rep_fars = [s["category_far"]["replacement_model"]["far"] * 100 for s in config_summaries.values()]
    pert_fars = [s["category_far"]["weight_perturbations"]["far"] * 100 for s in config_summaries.values()]
    prun_fars = [s["category_far"]["magnitude_pruning"]["far"] * 100 for s in config_summaries.values()]

    r1 = ax.bar(x - width, rep_fars, width, label="Replacement Model (N=1)", color="#4682b4", alpha=0.85, edgecolor="black")
    r2 = ax.bar(x, pert_fars, width, label="Weight Perturbation (N=18)", color="#d9534f", alpha=0.85, edgecolor="black")
    r3 = ax.bar(x + width, prun_fars, width, label="Magnitude Pruning (N=6)", color="#f0ad4e", alpha=0.85, edgecolor="black")

    ax.set_ylabel("False Acceptance Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("False Acceptance Rate (FAR) by Attack Category Across Candidate Configurations", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.tight_layout()
    p3 = plots_dir / "category_far_comparison.png"
    fig.savefig(p3)
    plt.close(fig)
    print(f"Generated Plot 3: {p3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 Experimental Improved Detection Runner")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_IMPROVED_DIR), help="Output directory for Phase 9 artifacts")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute inference (cpu or cuda)")
    args = parser.parse_args()

    run_improved_detection_evaluation(output_dir=args.output_dir, device=args.device)
