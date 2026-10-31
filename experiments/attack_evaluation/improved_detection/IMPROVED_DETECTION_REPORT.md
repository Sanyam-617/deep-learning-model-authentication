# Phase 9: Experimental Improved Detection Report (Proposal Rank 1)

**Role:** Member 4 — Model Tampering and Evaluation  
**Status:** Experimental Analysis (Non-Production / Research Findings)  
**Date:** October 30, 2026  
**Evaluation Scope:** 26 Model Variants $\times$ 500 Fixed Challenges $\times$ 7 Threshold Configurations  

---

## Executive Summary

In Phase 8, our diagnostic audit identified that Member 3's baseline authentication rule (`prediction_agreement >= 0.85`, `cosine_similarity >= 0.90`, and `kl_divergence <= 0.15`) resulted in a **100.00% False Acceptance Rate (FAR)** across all 18 Gaussian weight-perturbed model variants ($21/25 = 84.00\%$ overall unauthorized FAR). 

In Phase 9, as authorized by the team, Member 4 implemented **Proposal Rank 1**: an isolated, non-destructive experimental evaluation of six progressively stricter threshold configurations compared directly against the baseline.

### Key Experimental Findings:
1. **Zero Impact on Genuine Acceptance:** Across all seven candidate configurations—including the Zero-Tolerance candidate—the genuine positive control model (`models/original_model.pth`) achieved **0.00% False Rejection Rate (FRR: 0/1)**.
2. **Dramatic FAR Reduction:** Overall unauthorized FAR decreases progressively from **84.00% (Baseline)** down to **16.00% (High-Sensitivity)**, **12.00% (Ultra-Sensitive)**, and **0.00% (Zero-Tolerance Candidate)**.
3. **Pruning Attacks Fully Neutralized:** Starting at the **Strict** configuration (`Agr >= 0.99, Cos >= 0.99, KL <= 0.001`), magnitude pruning FAR drops from **50.00% to 0.00% (0/6)**.
4. **Subtle Gaussian Perturbations Detected:**
   - At **Strict**, perturbations with $\sigma \ge 0.01$ are 100% rejected.
   - At **High-Sensitivity**, perturbations with $\sigma \ge 0.01$ are 100% rejected, and $\sigma = 0.005$ is 66.7% rejected.
   - At **Ultra-Sensitive**, perturbations with $\sigma \ge 0.005$ are 100% rejected.
   - At **Zero-Tolerance**, even the microscopic $\sigma = 0.001$ perturbations (which flip zero discrete challenge predictions) are 100% rejected due to KL divergence thresholding at $1.0 \times 10^{-6}$.
5. **Exact Empirical Reproduction:** All behavioral metrics reproduced identically with zero discrepancy relative to Phase 8 diagnostic projections (max absolute difference $\le 10^{-6}$).

---

## 1. Candidate Threshold Configurations

All configurations were evaluated strictly through Member 3's existing `ModelAuthenticator` and `AuthenticationThresholds` interface without modifying the engine source code:

| Configuration Key | Configuration Name | Prediction Agreement ($A \ge$) | Cosine Similarity ($C \ge$) | KL Divergence ($D_{KL} \le$) | Intent & Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `baseline` | **Baseline (Member 3 Default)** | $0.85$ | $0.90$ | $0.15$ | Coarse development default; catches replacement and severe pruning. |
| `moderate` | **Moderate** | $0.95$ | $0.95$ | $0.05$ | Modest tightening; filters severe noise ($\sigma=0.10$). |
| `balanced` | **Balanced** | $0.98$ | $0.99$ | $0.01$ | Rejects moderate noise ($\sigma \ge 0.02$) and severe pruning. |
| `strict` | **Strict** | $0.99$ | $0.99$ | $0.001$ | Rejects mild noise ($\sigma \ge 0.01$) and all pruning levels ($s \ge 0.05$). |
| `high_sensitivity` | **High-Sensitivity** | $0.99$ | $0.999$ | $0.0001$ | High probability-distribution sensitivity; rejects most $\sigma=0.005$. |
| `ultra_sensitive` | **Ultra-Sensitive** | $0.99$ | $0.9999$ | $0.00001$ | Rejects all $\sigma \ge 0.005$; only microscopic $\sigma=0.001$ passes. |
| `zero_tolerance` | **Zero-Tolerance Candidate** | $0.999$ | $0.9999$ | $0.000001$ | Maximum distribution fidelity; detects any measurable parameter perturbation. |

---

## 2. Comprehensive Experimental Comparison Table

All 26 model variants were evaluated against the identical 500 challenges (`data/challenges/challenge_set.pt`) and reference fingerprint (`data/fingerprints/reference_fingerprint.json`):

| Metric / Category | Baseline (Default) | Moderate | Balanced | Strict | High-Sensitivity | Ultra-Sensitive | Zero-Tolerance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Agreement Threshold** | $\ge 0.85$ | $\ge 0.95$ | $\ge 0.98$ | $\ge 0.99$ | $\ge 0.99$ | $\ge 0.99$ | $\ge 0.999$ |
| **Cosine Threshold** | $\ge 0.90$ | $\ge 0.95$ | $\ge 0.99$ | $\ge 0.99$ | $\ge 0.999$ | $\ge 0.9999$ | $\ge 0.9999$ |
| **KL Threshold** | $\le 0.15$ | $\le 0.05$ | $\le 0.01$ | $\le 0.001$ | $\le 0.0001$ | $\le 0.00001$ | $\le 0.000001$ |
| **Genuine FRR** | **0/1 (0.00%)** | **0/1 (0.00%)** | **0/1 (0.00%)** | **0/1 (0.00%)** | **0/1 (0.00%)** | **0/1 (0.00%)** | **0/1 (0.00%)** |
| **Overall Unauthorized FAR** | **21/25 (84.00%)** | **15/25 (60.00%)** | **11/25 (44.00%)** | **6/25 (24.00%)** | **4/25 (16.00%)** | **3/25 (12.00%)** | **0/25 (0.00%)** |
| **Replacement Model FAR** | 0/1 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) |
| **Magnitude Pruning FAR** | 3/6 (50.00%) | 2/6 (33.33%) | 2/6 (33.33%) | **0/6 (0.00%)** | **0/6 (0.00%)** | **0/6 (0.00%)** | **0/6 (0.00%)** |
| - $s = 0.05$ | 1/1 (Pass) | 1/1 (Pass) | 1/1 (Pass) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) |
| - $s = 0.10$ | 1/1 (Pass) | 1/1 (Pass) | 1/1 (Pass) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) |
| - $s = 0.20$ | 1/1 (Pass) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) |
| - $s = 0.35$ | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) |
| - $s = 0.50$ | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) |
| - $s = 0.75$ | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) | 0/1 (Reject) |
| **Weight Perturbation FAR** | 18/18 (100.00%) | 13/18 (72.22%) | 9/18 (50.00%) | 6/18 (33.33%) | 4/18 (22.22%) | 3/18 (16.67%) | **0/18 (0.00%)** |
| - $\sigma = 0.001$ (3 seeds) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | **0/3 (Reject)** |
| - $\sigma = 0.005$ (3 seeds) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | **1/3 (Pass)** | **0/3 (Reject)** | **0/3 (Reject)** |
| - $\sigma = 0.010$ (3 seeds) | 3/3 (Pass) | 3/3 (Pass) | 3/3 (Pass) | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** |
| - $\sigma = 0.020$ (3 seeds) | 3/3 (Pass) | 3/3 (Pass) | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** |
| - $\sigma = 0.050$ (3 seeds) | 3/3 (Pass) | 1/3 (Pass) | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** |
| - $\sigma = 0.100$ (3 seeds) | 3/3 (Pass) | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** | **0/3 (Reject)** |

---

## 3. Discrepancy Analysis

The experimental evaluation runner performed automated consistency checks comparing the reproduced metrics against the Phase 8 diagnostic projections:
- **Maximum Absolute Agreement Difference:** $0.000000$ (Identical)
- **Maximum Absolute Cosine Similarity Difference:** $0.000000$ (Identical)
- **Maximum Absolute KL Divergence Difference:** $0.000000$ (Identical)
- **Result:** **Zero discrepancies detected.** All 26 model variants produced bitwise-identical classification outputs and softmax probabilities.

---

## 4. Visualizations & Graphical Trade-Off Analysis

Three high-resolution figures were generated and saved under `experiments/attack_evaluation/improved_detection/plots/`:

1. **`far_frr_tradeoff.png` (FAR vs FRR Trade-off):**
   Illustrates overall unauthorized FAR dropping monotonically from $84.00\%$ to $0.00\%$ across candidate configurations, while genuine FRR stays flat at $0.00\%$.
2. **`perturbation_detection_by_sigma.png` (Perturbation Detection Rate by Noise $\sigma$):**
   Demonstrates the step-wise detection curve ($1 - \text{FAR}$) as noise strength $\sigma$ increases from $0.001$ to $0.100$. Shows how tighter KL divergence thresholds shift the detection boundary toward subtle noise levels.
3. **`category_far_comparison.png` (FAR by Attack Category):**
   A grouped bar chart comparing error rates across the three unauthorized model categories (Replacement, Weight Perturbation, Magnitude Pruning) for each configuration.

---

## 5. Verification & Safety Audits

- **Syntax Verification:** `python -m py_compile src/attacks/run_improved_detection.py` passed with code 0.
- **Unit Test Suite:** All 7 unit tests in `src/attacks/attack_tests.py` passed cleanly in $0.507\text{s}$:
  ```
  Ran 7 tests in 0.507s ... OK
  ```
- **Preservation Check:**
  - `src/authentication/authenticator.py` was untouched.
  - `src/fingerprint/fingerprint.py` was untouched.
  - `data/challenges/challenge_set.pt` and `data/fingerprints/reference_fingerprint.json` were untouched.
  - `models/original_model.pth` and `models/test_models/` were untouched.
  - Existing baseline artifacts (`experiments/attack_evaluation/results_summary.json`, `far_frr_metrics.json`, etc.) remain identical and untouched.
  - No Git commits, pushes, or merges were performed.

---

## 6. Scientific Limitations & Disclosure

> [!WARNING]
> **Experimental Results Notice:**  
> The threshold configurations reported here are experimental findings evaluated on a single genuine positive control model ($N=1$) and 25 synthetic attack variants. They do **not** represent finalized or validated production thresholds.

### Critical Considerations for Future Phases:
1. **Sample Size ($N=1$ Genuine Model):**  
   The $0.00\%$ FRR achieved on the genuine model is a necessary condition, but an $N=1$ test cannot guarantee $0.00\%$ FRR across diverse operational environments. Factors such as cross-platform floating-point implementations (CPU vs CUDA vs Apple Silicon MPS), PyTorch runtime variations, or inference batch sizes can introduce minute numerical differences on the order of $10^{-6}\text{--}10^{-7}$.
2. **Calibration Risks of Zero-Tolerance:**  
   The `zero_tolerance` candidate threshold ($D_{KL} \le 1.0 \times 10^{-6}$) successfully detected all $\sigma = 0.001$ models, but its threshold buffer is narrow ($1.0 \times 10^{-6}$ vs genuine $3.98 \times 10^{-8}$). In deployment, harmless environmental noise or quantization could trigger false rejections.
3. **Recommended Production Compromise:**  
   The **High-Sensitivity** (`Agr >= 0.99, Cos >= 0.999, KL <= 0.0001`) configuration provides an optimal balance: it achieves $16.00\%$ overall FAR (rejecting all pruning, all replacement, and all $\sigma \ge 0.01$ perturbations), while providing an ample $2,500\times$ safety margin above the genuine model's KL divergence.
