"""
Unit and Integration Tests for Member 4 Attack Evaluator
========================================================
Tests:
  - Confusion matrix calculations (TP, TN, FP, FN)
  - Biometric rates (FAR, FRR, Accuracy, Precision, Recall, F1)
  - Genuine vs. impostor classification logic
  - Missing file handling
  - Plot generation routines (smoke tests)
  - Actual E1–E9 evaluation consistency
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.attack_evaluator import (
    compute_security_metrics,
    evaluate_attack_suite,
    generate_confusion_matrix_plot,
    generate_decisions_plot,
    generate_metrics_comparison_plot,
    load_experiment_results,
)


@pytest.fixture
def sample_perfect_records():
    """Synthetic dataset: 2 genuine accepted, 7 impostor rejected."""
    return [
        {"experiment_id": "E1", "ground_truth_category": "genuine", "authenticated": True},
        {"experiment_id": "E2", "ground_truth_category": "genuine", "authenticated": True},
        {"experiment_id": "E3", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "E4", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "E5", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "E6", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "E7", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "E8", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "E9", "ground_truth_category": "impostor", "authenticated": False},
    ]


@pytest.fixture
def sample_imperfect_records():
    """Synthetic dataset with 1 False Acceptance and 1 False Rejection."""
    return [
        {"experiment_id": "G1", "ground_truth_category": "genuine", "authenticated": True},
        {"experiment_id": "G2", "ground_truth_category": "genuine", "authenticated": False},  # FR (FN)
        {"experiment_id": "I1", "ground_truth_category": "impostor", "authenticated": False},
        {"experiment_id": "I2", "ground_truth_category": "impostor", "authenticated": True},   # FA (FP)
    ]


def test_compute_security_metrics_perfect(sample_perfect_records):
    """Test metrics on perfect classification (100% accuracy, 0% FAR, 0% FRR)."""
    metrics = compute_security_metrics(sample_perfect_records)

    cm = metrics["confusion_matrix"]
    assert cm["TP_TrueAcceptance"] == 2
    assert cm["FN_FalseRejection"] == 0
    assert cm["FP_FalseAcceptance"] == 0
    assert cm["TN_TrueRejection"] == 7

    counts = metrics["counts"]
    assert counts["total_genuine"] == 2
    assert counts["total_impostor"] == 7
    assert counts["total_trials"] == 9

    rates = metrics["rates"]
    assert rates["FAR_percent"] == 0.0
    assert rates["FRR_percent"] == 0.0
    assert rates["accuracy_percent"] == 100.0
    assert rates["precision"] == 1.0
    assert rates["recall"] == 1.0
    assert rates["f1_score"] == 1.0


def test_compute_security_metrics_imperfect(sample_imperfect_records):
    """Test metrics when false acceptance and false rejection occur."""
    metrics = compute_security_metrics(sample_imperfect_records)

    cm = metrics["confusion_matrix"]
    assert cm["TP_TrueAcceptance"] == 1
    assert cm["FN_FalseRejection"] == 1
    assert cm["FP_FalseAcceptance"] == 1
    assert cm["TN_TrueRejection"] == 1

    rates = metrics["rates"]
    # FAR = 1 FA / 2 Impostors = 50.0%
    assert rates["FAR_percent"] == 50.0
    # FRR = 1 FR / 2 Genuines = 50.0%
    assert rates["FRR_percent"] == 50.0
    # Accuracy = (1 + 1) / 4 = 50.0%
    assert rates["accuracy_percent"] == 50.0
    # Precision = 1 / (1 + 1) = 0.5
    assert rates["precision"] == 0.5
    # Recall = 1 / (1 + 1) = 0.5
    assert rates["recall"] == 0.5
    # F1 = 0.5
    assert rates["f1_score"] == 0.5


def test_compute_security_metrics_empty():
    """Empty list should not trigger division by zero."""
    metrics = compute_security_metrics([])
    assert metrics["counts"]["total_trials"] == 0
    assert metrics["rates"]["FAR_percent"] == 0.0
    assert metrics["rates"]["FRR_percent"] == 0.0
    assert metrics["rates"]["accuracy_percent"] == 0.0


def test_load_experiment_results_real_files():
    """Verify loading actual E1–E9 results from disk."""
    records = load_experiment_results()
    assert len(records) == 9

    # Check ordering E1 through E9
    ids = [r["experiment_id"] for r in records]
    assert ids == ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]

    # Verify category assignments
    assert records[0]["ground_truth_category"] == "genuine"
    assert records[1]["ground_truth_category"] == "genuine"
    for r in records[2:]:
        assert r["ground_truth_category"] == "impostor"

    # Verify decisions
    assert records[0]["decision"] == "AUTHENTIC"
    assert records[1]["decision"] == "AUTHENTIC"
    for r in records[2:]:
        assert r["decision"] == "REJECTED"


def test_load_experiment_results_missing_file():
    """Missing file should raise FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Missing result file"):
            load_experiment_results(Path(tmpdir))


def test_plot_generation_smoke():
    """Ensure all three plotting routines create valid image files without crashing."""
    records = load_experiment_results()
    metrics = compute_security_metrics(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        cm_path = tmppath / "cm.png"
        metrics_path = tmppath / "metrics.png"
        decisions_path = tmppath / "decisions.png"

        generate_confusion_matrix_plot(metrics, cm_path)
        generate_metrics_comparison_plot(records, metrics_path)
        generate_decisions_plot(records, decisions_path)

        assert cm_path.exists() and cm_path.stat().st_size > 1000
        assert metrics_path.exists() and metrics_path.stat().st_size > 1000
        assert decisions_path.exists() and decisions_path.stat().st_size > 1000


def test_evaluate_attack_suite_integration():
    """Verify end-to-end evaluate_attack_suite function."""
    res = evaluate_attack_suite()
    assert "report" in res
    assert "files_saved" in res
    rep = res["report"]
    assert rep["security_metrics"]["counts"]["total_trials"] == 9
    assert rep["security_metrics"]["rates"]["FAR_percent"] == 0.0
    assert rep["security_metrics"]["rates"]["FRR_percent"] == 0.0
    assert rep["security_metrics"]["rates"]["accuracy_percent"] == 100.0

