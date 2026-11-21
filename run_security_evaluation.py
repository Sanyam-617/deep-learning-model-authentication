"""
Final Security Evaluation
=========================
Deep Learning Model Authentication Using Behavioral Challenge-Response

This script performs a rigorous security evaluation with proper methodology:

  1. CALIBRATION PHASE: Generate a separate set of genuine and impostor models,
     compute their prediction-agreement scores, and select the operating
     threshold that minimises calibration error (EER approximation).

  2. FREEZE THRESHOLD: Lock the threshold before touching the evaluation set.

  3. EVALUATION PHASE: Generate a held-out set of genuine and impostor models
     (using DIFFERENT random seeds), apply the frozen threshold, and record
     all authentication decisions.

  4. METRICS: Confusion matrix, FAR, FRR, Accuracy, Precision, Recall, F1.

  5. VISUALISATION: FAR/FRR-vs-threshold curve, confusion-matrix heatmap,
     per-attack-type comparison bar chart.

All genuine models are semantically identical to original_model.pth
(exact copies, re-serialised copies). All impostor models are constructed
by independent training, weight perturbation, random initialisation, or
layer replacement — each with unique seeds so that calibration and
evaluation sets are disjoint.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

RESULTS_DIR = PROJECT_ROOT / "results" / "security_evaluation"
MODELS_DIR  = PROJECT_ROOT / "models" / "security_evaluation"


# ──────────────────────────────────────────────
# 1.  Model Factories
# ──────────────────────────────────────────────

def _make_genuine_copy(original: CIFAR10CNN, label: str) -> CIFAR10CNN:
    """Return a deep copy (simulates load-and-save)."""
    m = copy.deepcopy(original)
    m.eval()
    return m


def _make_perturbed(original: CIFAR10CNN, frac: float, seed: int) -> CIFAR10CNN:
    """Perturb *frac* of all weight elements with Gaussian noise."""
    torch.manual_seed(seed)
    m = copy.deepcopy(original)
    with torch.no_grad():
        for p in m.parameters():
            mask = torch.rand_like(p.data) < frac
            if mask.any():
                std = max(p.data.std().item(), 0.01)
                p.data[mask] += torch.randn_like(p.data[mask]) * std
    m.eval()
    return m


def _make_random_init(seed: int) -> CIFAR10CNN:
    """Fresh random weights (untrained)."""
    torch.manual_seed(seed)
    m = CIFAR10CNN()
    m.eval()
    return m


def _make_head_replaced(original: CIFAR10CNN, seed: int) -> CIFAR10CNN:
    """Replace the final classification layer."""
    torch.manual_seed(seed)
    m = copy.deepcopy(original)
    head = nn.Linear(256, 10)
    with torch.no_grad():
        m.classifier[4].weight.copy_(head.weight)
        m.classifier[4].bias.copy_(head.bias)
    m.eval()
    return m


def _make_independently_trained(seed: int, device: torch.device,
                                num_batches: int = 100) -> CIFAR10CNN:
    """Train a fresh model on CIFAR-10 with a unique seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CIFAR10CNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=1e-3)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                      download=False, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    m.train()
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
    m.eval()
    return m


# ──────────────────────────────────────────────
# 2.  Trial Specification
# ──────────────────────────────────────────────

@dataclass
class Trial:
    trial_id: str
    label: str           # "genuine" or "impostor"
    attack_type: str     # human-readable category
    description: str
    seed: int
    build_fn: str        # factory function name (for logging)
    frac: float = 0.0    # perturbation fraction (only for perturbed)


def _calibration_trials() -> List[Trial]:
    """Calibration set — used ONLY to select the threshold."""
    trials = []
    # 4 genuine
    for i in range(4):
        trials.append(Trial(
            trial_id=f"CAL-G{i+1}", label="genuine",
            attack_type="Legitimate Copy",
            description=f"Genuine copy #{i+1} of original model",
            seed=2000+i, build_fn="genuine_copy"))

    # 12 impostor (diverse seeds)
    # 2 × independently trained
    for i, s in enumerate([3001, 3002]):
        trials.append(Trial(
            trial_id=f"CAL-I-TRAIN{i+1}", label="impostor",
            attack_type="Independent Training",
            description=f"Independently trained model (seed={s})",
            seed=s, build_fn="independently_trained"))
    # 2 × random init
    for i, s in enumerate([4001, 4002]):
        trials.append(Trial(
            trial_id=f"CAL-I-RAND{i+1}", label="impostor",
            attack_type="Random Initialisation",
            description=f"Randomly initialised weights (seed={s})",
            seed=s, build_fn="random_init"))
    # 2 × head replaced
    for i, s in enumerate([5001, 5002]):
        trials.append(Trial(
            trial_id=f"CAL-I-HEAD{i+1}", label="impostor",
            attack_type="Head Replacement",
            description=f"Classification head reinitialised (seed={s})",
            seed=s, build_fn="head_replaced"))
    # 6 × weight perturbation (various fracs)
    for frac, s in [(0.01, 6001), (0.02, 6002), (0.05, 6003),
                    (0.10, 6004), (0.25, 6005), (0.50, 6006)]:
        trials.append(Trial(
            trial_id=f"CAL-I-PERT{int(frac*100)}",
            label="impostor",
            attack_type=f"Weight Perturbation ({frac*100:.0f}%)",
            description=f"{frac*100:.0f}% weights perturbed (seed={s})",
            seed=s, build_fn="perturbed", frac=frac))
    return trials


def _evaluation_trials() -> List[Trial]:
    """Held-out evaluation set — DIFFERENT seeds from calibration."""
    trials = []
    # 6 genuine
    for i in range(6):
        trials.append(Trial(
            trial_id=f"EVAL-G{i+1}", label="genuine",
            attack_type="Legitimate Copy",
            description=f"Genuine copy #{i+1} of original model",
            seed=8000+i, build_fn="genuine_copy"))

    # 14 impostor
    # 2 × independently trained
    for i, s in enumerate([9001, 9002]):
        trials.append(Trial(
            trial_id=f"EVAL-I-TRAIN{i+1}", label="impostor",
            attack_type="Independent Training",
            description=f"Independently trained model (seed={s})",
            seed=s, build_fn="independently_trained"))
    # 2 × random init
    for i, s in enumerate([9101, 9102]):
        trials.append(Trial(
            trial_id=f"EVAL-I-RAND{i+1}", label="impostor",
            attack_type="Random Initialisation",
            description=f"Randomly initialised weights (seed={s})",
            seed=s, build_fn="random_init"))
    # 2 × head replaced
    for i, s in enumerate([9201, 9202]):
        trials.append(Trial(
            trial_id=f"EVAL-I-HEAD{i+1}", label="impostor",
            attack_type="Head Replacement",
            description=f"Classification head reinitialised (seed={s})",
            seed=s, build_fn="head_replaced"))
    # 8 × weight perturbation
    for frac, s in [(0.005, 9301), (0.01, 9302), (0.02, 9303),
                    (0.05, 9304), (0.10, 9305), (0.15, 9306),
                    (0.25, 9307), (0.50, 9308)]:
        trials.append(Trial(
            trial_id=f"EVAL-I-PERT{frac*100:.1f}".replace(".", "p"),
            label="impostor",
            attack_type=f"Weight Perturbation ({frac*100:g}%)",
            description=f"{frac*100:g}% weights perturbed (seed={s})",
            seed=s, build_fn="perturbed", frac=frac))
    return trials


# ──────────────────────────────────────────────
# 3.  Build a Model from a Trial Spec
# ──────────────────────────────────────────────

def build_model_for_trial(trial: Trial, original: CIFAR10CNN,
                          device: torch.device) -> CIFAR10CNN:
    if trial.build_fn == "genuine_copy":
        return _make_genuine_copy(original, trial.trial_id)
    elif trial.build_fn == "perturbed":
        return _make_perturbed(original, trial.frac, trial.seed)
    elif trial.build_fn == "random_init":
        return _make_random_init(trial.seed)
    elif trial.build_fn == "head_replaced":
        return _make_head_replaced(original, trial.seed)
    elif trial.build_fn == "independently_trained":
        return _make_independently_trained(trial.seed, device)
    else:
        raise ValueError(f"Unknown build function: {trial.build_fn}")


# ──────────────────────────────────────────────
# 4.  Run a Batch of Trials → Raw Scores
# ──────────────────────────────────────────────

def run_trial_batch(
    trials: List[Trial],
    original: CIFAR10CNN,
    ref_fp: Dict[str, Any],
    challenge_path: Path,
    device: torch.device,
    phase_name: str,
) -> List[Dict[str, Any]]:
    """Run all trials and return per-trial metric records."""
    auth = ModelAuthenticator()      # default thresholds (irrelevant here)
    records: List[Dict[str, Any]] = []

    print(f"\n{'─'*70}")
    print(f"  {phase_name}: running {len(trials)} trials")
    print(f"{'─'*70}")

    for idx, t in enumerate(trials, 1):
        t0 = time.time()
        try:
            model = build_model_for_trial(t, original, device)
            fp = extract_fingerprint(model, str(challenge_path), device, batch_size=64)
            validate_fingerprint_alignment(ref_fp, fp)
            metrics = auth.compare_metrics(ref_fp, fp)
            elapsed = time.time() - t0

            rec = {
                "trial_id": t.trial_id,
                "label": t.label,
                "attack_type": t.attack_type,
                "description": t.description,
                "prediction_agreement": metrics.prediction_agreement,
                "cosine_similarity": metrics.cosine_similarity,
                "kl_divergence": metrics.kl_divergence,
                "matching_predictions": metrics.matching_predictions,
                "mismatching_predictions": metrics.mismatching_predictions,
                "num_challenges": metrics.num_challenges,
                "error": None,
                "time_sec": round(elapsed, 3),
            }
        except Exception as e:
            elapsed = time.time() - t0
            rec = {
                "trial_id": t.trial_id,
                "label": t.label,
                "attack_type": t.attack_type,
                "description": t.description,
                "prediction_agreement": None,
                "cosine_similarity": None,
                "kl_divergence": None,
                "matching_predictions": None,
                "mismatching_predictions": None,
                "num_challenges": None,
                "error": str(e),
                "time_sec": round(elapsed, 3),
            }

        records.append(rec)
        tag = "GEN" if t.label == "genuine" else "IMP"
        agr = f"{rec['prediction_agreement']*100:5.1f}%" if rec['prediction_agreement'] is not None else "ERR"
        cos = f"{rec['cosine_similarity']:.4f}" if rec['cosine_similarity'] is not None else "ERR"
        kl  = f"{rec['kl_divergence']:.4f}" if rec['kl_divergence'] is not None else "ERR"
        print(f"  [{idx:2d}/{len(trials)}] {tag} {t.trial_id:<22} Agr={agr}  Cos={cos}  KL={kl}  ({elapsed:.2f}s)")

    return records


# ──────────────────────────────────────────────
# 5.  Threshold Calibration
# ──────────────────────────────────────────────

def calibrate_threshold(
    cal_records: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """Sweep prediction-agreement thresholds over calibration data.

    Returns the threshold that minimises ``|FAR − FRR|`` (operating point
    closest to the Equal Error Rate) and the full sweep data.
    """
    genuine_scores = [r["prediction_agreement"] for r in cal_records
                      if r["label"] == "genuine" and r["prediction_agreement"] is not None]
    impostor_scores = [r["prediction_agreement"] for r in cal_records
                       if r["label"] == "impostor" and r["prediction_agreement"] is not None]

    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    assert n_gen > 0, "No genuine calibration samples"
    assert n_imp > 0, "No impostor calibration samples"

    thresholds = np.linspace(0.0, 1.0, 1001)
    sweep: List[Dict[str, float]] = []
    best_t, best_diff = 0.5, 999.0

    for t in thresholds:
        # genuine accepted if score >= t
        fr = sum(1 for s in genuine_scores if s < t)
        fa = sum(1 for s in impostor_scores if s >= t)
        frr = fr / n_gen * 100
        far = fa / n_imp * 100
        diff = abs(far - frr)
        sweep.append({"threshold": float(t), "far": far, "frr": frr})
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)

    cal_info = {
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "selected_threshold": best_t,
        "eer_approx_diff": best_diff,
        "sweep": sweep,
    }
    return best_t, cal_info


# ──────────────────────────────────────────────
# 6.  Apply Frozen Threshold → Decisions
# ──────────────────────────────────────────────

def apply_threshold(records: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Assign accept/reject decisions with the frozen threshold."""
    for r in records:
        if r["prediction_agreement"] is not None:
            r["accepted"] = r["prediction_agreement"] >= threshold
        else:
            r["accepted"] = False

        is_gen = r["label"] == "genuine"
        if is_gen and r["accepted"]:
            r["outcome"] = "TA"   # True Acceptance
        elif is_gen and not r["accepted"]:
            r["outcome"] = "FR"   # False Rejection
        elif not is_gen and not r["accepted"]:
            r["outcome"] = "TR"   # True Rejection
        else:
            r["outcome"] = "FA"   # False Acceptance

        r["frozen_threshold"] = threshold
    return records


# ──────────────────────────────────────────────
# 7.  Compute Final Metrics
# ──────────────────────────────────────────────

def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Confusion matrix, FAR, FRR, Accuracy, Precision, Recall, F1."""
    ta = sum(1 for r in records if r["outcome"] == "TA")
    fr = sum(1 for r in records if r["outcome"] == "FR")
    tr = sum(1 for r in records if r["outcome"] == "TR")
    fa = sum(1 for r in records if r["outcome"] == "FA")

    n_gen = ta + fr
    n_imp = tr + fa
    total = n_gen + n_imp

    far = (fa / n_imp * 100) if n_imp > 0 else 0.0
    frr = (fr / n_gen * 100) if n_gen > 0 else 0.0
    accuracy = ((ta + tr) / total * 100) if total > 0 else 0.0

    # Precision / Recall / F1 (treating "Authentic" as the positive class)
    precision = (ta / (ta + fa)) if (ta + fa) > 0 else 0.0
    recall    = (ta / (ta + fr)) if (ta + fr) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "confusion_matrix": {"TA": ta, "FR": fr, "TR": tr, "FA": fa},
        "total_genuine": n_gen,
        "total_impostor": n_imp,
        "total_trials": total,
        "FAR_pct": round(far, 4),
        "FRR_pct": round(frr, 4),
        "accuracy_pct": round(accuracy, 4),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
    }


# ──────────────────────────────────────────────
# 8.  Per-Attack-Type Breakdown
# ──────────────────────────────────────────────

def per_attack_breakdown(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group evaluation impostor records by attack type."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r["label"] == "impostor":
            groups[r["attack_type"]].append(r)

    breakdown = []
    for atype, recs in sorted(groups.items()):
        n = len(recs)
        detected = sum(1 for r in recs if r["outcome"] == "TR")
        bypassed = sum(1 for r in recs if r["outcome"] == "FA")
        avg_agr  = np.mean([r["prediction_agreement"] for r in recs
                            if r["prediction_agreement"] is not None])
        avg_cos  = np.mean([r["cosine_similarity"] for r in recs
                            if r["cosine_similarity"] is not None])
        avg_kl   = np.mean([r["kl_divergence"] for r in recs
                            if r["kl_divergence"] is not None])
        breakdown.append({
            "attack_type": atype,
            "n_trials": n,
            "detected": detected,
            "bypassed": bypassed,
            "detection_rate_pct": round(detected / n * 100, 2) if n > 0 else 0,
            "avg_prediction_agreement": round(float(avg_agr), 6),
            "avg_cosine_similarity": round(float(avg_cos), 6),
            "avg_kl_divergence": round(float(avg_kl), 6),
        })
    return breakdown


# ──────────────────────────────────────────────
# 9.  Visualisation
# ──────────────────────────────────────────────

def plot_far_frr_curve(cal_info: Dict, frozen_threshold: float,
                       out_path: Path) -> None:
    """Plot FAR and FRR as functions of the decision threshold."""
    sweep = cal_info["sweep"]
    ts   = [s["threshold"] for s in sweep]
    fars = [s["far"] for s in sweep]
    frrs = [s["frr"] for s in sweep]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, fars, color="#d62728", linewidth=2, label="FAR (False Acceptance Rate)")
    ax.plot(ts, frrs, color="#1f77b4", linewidth=2, label="FRR (False Rejection Rate)")
    ax.axvline(frozen_threshold, color="#2ca02c", linestyle="--", linewidth=2,
               label=f"Frozen Threshold = {frozen_threshold:.3f}")
    ax.set_xlabel("Prediction Agreement Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("FAR and FRR vs Decision Threshold\n(Calibration Data)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(metrics: Dict, out_path: Path) -> None:
    """Plot a 2×2 confusion-matrix heatmap."""
    cm = metrics["confusion_matrix"]
    mat = np.array([[cm["TA"], cm["FR"]],
                    [cm["FA"], cm["TR"]]])
    labels_pred = ["Accepted\n(Authentic)", "Rejected"]
    labels_true = ["Genuine", "Impostor"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_pred, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_true, fontsize=11)
    ax.set_xlabel("System Decision", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Identity", fontsize=12, fontweight="bold")
    ax.set_title("Authentication Confusion Matrix\n(Evaluation Set — Frozen Threshold)",
                 fontsize=13, fontweight="bold")
    outcome_labels = [["TA (True\nAcceptance)", "FR (False\nRejection)"],
                      ["FA (False\nAcceptance)", "TR (True\nRejection)"]]
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            txt = f"{outcome_labels[i][j]}\n{val}"
            color = "white" if val > mat.max() * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_attack_comparison(breakdown: List[Dict], out_path: Path) -> None:
    """Grouped bar chart of average metrics by attack type."""
    types = [b["attack_type"] for b in breakdown]
    agrs  = [b["avg_prediction_agreement"] * 100 for b in breakdown]
    coss  = [b["avg_cosine_similarity"] for b in breakdown]
    kls   = [b["avg_kl_divergence"] for b in breakdown]
    det   = [b["detection_rate_pct"] for b in breakdown]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(types))
    w = 0.55

    # Agreement
    ax = axes[0, 0]
    bars = ax.bar(x, agrs, w, color="#4e79a7", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg Agreement (%)", fontweight="bold")
    ax.set_title("Prediction Agreement by Attack Type", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.1f}%",
                ha="center", fontsize=8, fontweight="bold")

    # Cosine
    ax = axes[0, 1]
    bars = ax.bar(x, coss, w, color="#59a14f", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg Cosine Similarity", fontweight="bold")
    ax.set_title("Cosine Similarity by Attack Type", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.3f}",
                ha="center", fontsize=8, fontweight="bold")

    # KL
    ax = axes[1, 0]
    bars = ax.bar(x, kls, w, color="#e15759", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg KL Divergence (nats)", fontweight="bold")
    ax.set_title("KL Divergence by Attack Type", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.3f}",
                ha="center", fontsize=8, fontweight="bold")

    # Detection Rate
    ax = axes[1, 1]
    colors = ["#2ca02c" if d == 100 else "#d62728" for d in det]
    bars = ax.bar(x, det, w, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Detection Rate (%)", fontweight="bold")
    ax.set_title("Detection Rate by Attack Type", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f"{b.get_height():.0f}%",
                ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Per-Attack-Type Comparison (Evaluation Set)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_score_distributions(cal_info: Dict, frozen_threshold: float,
                             out_path: Path) -> None:
    """Histogram of genuine vs impostor agreement scores."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(cal_info["genuine_scores"], bins=bins, alpha=0.7,
            color="#2ca02c", label="Genuine", edgecolor="black")
    ax.hist(cal_info["impostor_scores"], bins=bins, alpha=0.7,
            color="#d62728", label="Impostor", edgecolor="black")
    ax.axvline(frozen_threshold, color="#1f77b4", linestyle="--", linewidth=2,
               label=f"Threshold = {frozen_threshold:.3f}")
    ax.set_xlabel("Prediction Agreement", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Score Distributions (Calibration Set)\nGenuine vs Impostor",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────
# 10. Main Pipeline
# ──────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  FINAL SECURITY EVALUATION")
    print("  Deep Learning Model Authentication — Behavioral Challenge-Response")
    print("=" * 80)
    print(f"[+] Device: {device}")
    print(f"[+] Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # ── Load reference ──
    print("\n[1] Loading original model and reference fingerprint ...")
    original = load_model(str(DEFAULT_MODEL_PATH), device)
    ref_fp = torch.load(DEFAULT_REF_FP_PATH, map_location="cpu", weights_only=False)
    print(f"    Reference fingerprint: {ref_fp['num_challenges']} challenges.")

    # ── CALIBRATION PHASE ──
    print("\n" + "=" * 80)
    print("  PHASE A: CALIBRATION (threshold selection)")
    print("=" * 80)
    cal_trials = _calibration_trials()
    cal_records = run_trial_batch(cal_trials, original, ref_fp,
                                 DEFAULT_CHALLENGE_PATH, device,
                                 "CALIBRATION")
    frozen_threshold, cal_info = calibrate_threshold(cal_records)
    print(f"\n  ★ CALIBRATED THRESHOLD (prediction agreement): {frozen_threshold:.3f}")
    print(f"    EER approximation |FAR−FRR| at threshold: {cal_info['eer_approx_diff']:.2f}%")
    print(f"    Calibration genuine scores : {cal_info['genuine_scores']}")
    print(f"    Calibration impostor scores: {[round(s,4) for s in cal_info['impostor_scores']]}")

    # ── FREEZE ──
    print(f"\n  >>> THRESHOLD FROZEN AT {frozen_threshold:.3f} <<<")
    print(f"  >>> No further changes to threshold permitted <<<")

    # ── EVALUATION PHASE ──
    print("\n" + "=" * 80)
    print("  PHASE B: HELD-OUT EVALUATION (frozen threshold)")
    print("=" * 80)
    eval_trials = _evaluation_trials()
    eval_records = run_trial_batch(eval_trials, original, ref_fp,
                                  DEFAULT_CHALLENGE_PATH, device,
                                  "EVALUATION")

    # ── Apply frozen threshold ──
    eval_records = apply_threshold(eval_records, frozen_threshold)
    metrics = compute_metrics(eval_records)
    breakdown = per_attack_breakdown(eval_records)

    # ── Print results ──
    cm = metrics["confusion_matrix"]
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS (Frozen Threshold = {frozen_threshold:.3f})")
    print(f"{'='*70}")
    print(f"                       System Decision")
    print(f"                 Accepted         Rejected")
    print(f"  Genuine      TA = {cm['TA']:<5d}       FR = {cm['FR']:<5d}   (total: {metrics['total_genuine']})")
    print(f"  Impostor     FA = {cm['FA']:<5d}       TR = {cm['TR']:<5d}   (total: {metrics['total_impostor']})")
    print()
    print(f"  FAR = {metrics['FAR_pct']:.4f}%   (False Acceptances / Total Impostor × 100)")
    print(f"  FRR = {metrics['FRR_pct']:.4f}%   (False Rejections  / Total Genuine × 100)")
    print(f"  Authentication Accuracy = {metrics['accuracy_pct']:.4f}%")
    print(f"  Precision = {metrics['precision']:.6f}")
    print(f"  Recall    = {metrics['recall']:.6f}")
    print(f"  F1 Score  = {metrics['f1_score']:.6f}")

    print(f"\n{'─'*70}")
    print(f"  Per-Attack-Type Breakdown")
    print(f"{'─'*70}")
    print(f"  {'Attack Type':<28} {'N':>3}  {'Det':>3}  {'Byp':>3}  {'Det%':>6}  {'AvgAgr':>8}  {'AvgCos':>8}  {'AvgKL':>8}")
    for b in breakdown:
        print(f"  {b['attack_type']:<28} {b['n_trials']:>3}  {b['detected']:>3}  "
              f"{b['bypassed']:>3}  {b['detection_rate_pct']:>5.1f}%  "
              f"{b['avg_prediction_agreement']*100:>7.2f}%  "
              f"{b['avg_cosine_similarity']:>8.4f}  {b['avg_kl_divergence']:>8.4f}")

    # ── Save all outputs ──
    # JSON report
    full_report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "frozen_threshold": frozen_threshold,
        "calibration": {
            "n_genuine": cal_info["n_genuine"],
            "n_impostor": cal_info["n_impostor"],
            "genuine_scores": cal_info["genuine_scores"],
            "impostor_scores": cal_info["impostor_scores"],
            "eer_approx_diff": cal_info["eer_approx_diff"],
        },
        "evaluation_metrics": metrics,
        "per_attack_breakdown": breakdown,
        "evaluation_records": eval_records,
        "calibration_records": cal_records,
        "limitations": (
            "The evaluation set contains 20 trials (6 genuine, 14 impostor). "
            "While sufficient to demonstrate correct system behaviour across "
            "all tested attack categories, the sample size is too small to "
            "derive statistically significant confidence intervals for FAR/FRR. "
            "Production deployment would require hundreds or thousands of "
            "trials per category for reliable error-rate estimation."
        ),
    }
    json_path = RESULTS_DIR / "security_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n[+] Saved JSON report         : {json_path}")

    # CSV
    csv_path = RESULTS_DIR / "evaluation_decisions.csv"
    fieldnames = [
        "trial_id", "label", "attack_type", "outcome", "accepted",
        "prediction_agreement", "cosine_similarity", "kl_divergence",
        "frozen_threshold", "description",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in eval_records:
            w.writerow(r)
    print(f"[+] Saved evaluation CSV      : {csv_path}")

    # Plots
    plot_far_frr_curve(cal_info, frozen_threshold,
                       RESULTS_DIR / "far_frr_vs_threshold.png")
    print(f"[+] Saved FAR/FRR curve       : {RESULTS_DIR / 'far_frr_vs_threshold.png'}")

    plot_confusion_matrix(metrics, RESULTS_DIR / "confusion_matrix.png")
    print(f"[+] Saved confusion matrix    : {RESULTS_DIR / 'confusion_matrix.png'}")

    plot_attack_comparison(breakdown, RESULTS_DIR / "attack_type_comparison.png")
    print(f"[+] Saved attack comparison   : {RESULTS_DIR / 'attack_type_comparison.png'}")

    plot_score_distributions(cal_info, frozen_threshold,
                             RESULTS_DIR / "score_distributions.png")
    print(f"[+] Saved score distributions : {RESULTS_DIR / 'score_distributions.png'}")

    print(f"\n{'='*80}")
    print(f"  SECURITY EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

