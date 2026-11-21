"""
Member 4 – Attack Evaluation Module
====================================
Deep Learning Model Authentication Using Behavioral Challenge–Response

Provides attack evaluation, security metrics computation (FAR, FRR, Accuracy, F1),
confusion matrix generation, and visualization routines across evaluated model variants.
"""

from src.attacks.attack_evaluator import (
    compute_security_metrics,
    evaluate_attack_suite,
    load_experiment_results,
    generate_confusion_matrix_plot,
    generate_metrics_comparison_plot,
    generate_decisions_plot,
)

__all__ = [
    "compute_security_metrics",
    "evaluate_attack_suite",
    "load_experiment_results",
    "generate_confusion_matrix_plot",
    "generate_metrics_comparison_plot",
    "generate_decisions_plot",
]
