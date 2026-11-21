# Deep Learning Model Authentication Using Behavioral Challenge–Response

An end-to-end security framework that verifies whether a deployed deep learning model is the genuine registered model or an impostor (tampered, independently trained, or substitute model) using black-box behavioral challenge–response fingerprinting.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Completed Project Modules](#completed-project-modules)
3. [Dataset](#dataset)
4. [CNN Architecture](#cnn-architecture)
5. [Project Directory Structure](#project-directory-structure)
6. [Setup & Virtual Environment](#setup--virtual-environment)
7. [Running the Automated Test Suite](#running-the-automated-test-suite)
8. [End-to-End Model Authentication Pipeline](#end-to-end-model-authentication-pipeline)
9. [Experimental Benchmark (E1–E9)](#experimental-benchmark-e1e9)
10. [Outputs & Evaluation Reports](#outputs--evaluation-reports)
11. [License](#license)

---

## Project Overview

The objective of this project is to provide a reliable, non-intrusive model authentication protocol. By presenting a registered deep learning model with a fixed, carefully generated set of challenge inputs (original and transformed CIFAR-10 validation images) and recording its full output probability distributions, we construct a unique **behavioral fingerprint**.

Candidate models submitted for authentication are subjected to the identical challenge set. Three independent, complementary behavioral metrics are calculated:
1. **Prediction Agreement**: Discrete class decision concordance ($\ge 85.0\%$).
2. **Cosine Similarity**: Angular alignment of output probability distributions ($\ge 0.9000$).
3. **KL Divergence**: Information-theoretic relative entropy ($\le 0.1500$).

Under the conjunction decision rule (`all_metrics`), a candidate model is classified as **`AUTHENTIC`** only if all three metric thresholds are simultaneously satisfied; otherwise, it is **`REJECTED`**.

---

## Completed Project Modules

All four modules of the project are fully implemented and integrated:

| Module | Component | Directory | Status | Description |
|---|---|---|:---:|---|
| **Module 1** | Challenge Set Generation | `src/challenges/` | **COMPLETED** | Generates a reproducible, class-balanced challenge set from CIFAR-10 validation data using 5 controlled transformations (original, Gaussian noise, rotation, brightness, crop & resize). |
| **Module 2** | Behavioral Fingerprint Extraction | `src/fingerprint/` | **COMPLETED** | Extracts softmax probability distributions, argmax predicted classes, and confidence scores under deterministic inference (`model.eval()`, `torch.no_grad()`). |
| **Module 3** | Authentication Engine | `src/authentication/` | **COMPLETED** | Validates challenge alignment, calculates the three behavioral metrics, and evaluates baseline threshold criteria. |
| **Module 4** | Attack Evaluation & Security Analysis | `src/attacks/` | **COMPLETED** | Evaluates security against weight perturbation, independent training, and substitution; computes confusion matrix, FAR, FRR, accuracy, and F1-score. |

---

## Dataset

| Property | Value |
|---|---|
| **Name** | CIFAR-10 |
| **Images** | 60,000 colour images (32 × 32 px) |
| **Classes** | 10 — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| **Train / Test Split** | 50,000 / 10,000 (official split) |
| **Validation** | 10% of training set (5,000 images) held out using reproducible seed `42` |

**Data Augmentation (training only):**
- Random horizontal flip
- Random crop (32 × 32 with 4 px padding)

**Normalization (all splits & challenges):**
- Mean: `(0.4914, 0.4822, 0.4465)`
- Std:  `(0.2470, 0.2435, 0.2616)`

---

## CNN Architecture

```
Input (3 × 32 × 32)
 │
 ├─ Conv Block 1
 │    Conv2d(3 → 32, 3×3, pad=1) → ReLU
 │    Conv2d(32 → 32, 3×3, pad=1) → ReLU
 │    MaxPool2d(2×2)
 │    Dropout2d(0.25)
 │
 ├─ Conv Block 2
 │    Conv2d(32 → 64, 3×3, pad=1) → ReLU
 │    Conv2d(64 → 64, 3×3, pad=1) → ReLU
 │    MaxPool2d(2×2)
 │    Dropout2d(0.25)
 │
 ├─ Classifier
 │    Flatten
 │    Linear(64 × 8 × 8 [4096] → 256) → ReLU
 │    Dropout(0.5)
 │    Linear(256 → 10)
 │
 └─ Output (10 logits)
```

| Metric | Value |
|---|---|
| Total parameters | **1,116,970 (~1.12 M)** |
| Trainable parameters | **1,116,970 (~1.12 M)** |
| Expected validation accuracy | ~78 – 83% (10 epochs) |

---

## Project Directory Structure

```
AI ML Project/
├── data/
│   ├── challenges/
│   │   ├── challenge_set.pt            # 100 challenge images (shape: [100, 3, 32, 32])
│   │   └── metadata.json               # Provenance, transform parameters, and seeds
│   ├── fingerprints/
│   │   └── reference_fingerprint.pt    # Registered baseline behavioral fingerprint
│   └── cifar-10-batches-py/            # Raw CIFAR-10 data batches
├── models/
│   ├── original_model.pth              # Registered legitimate CNN weights
│   ├── model_config.json               # Model architecture hyperparameters
│   ├── exact_copy_model.pth            # E2: Exact serialization copy
│   ├── modified_model_1pct.pth         # E3: 1% weight perturbation checkpoint
│   ├── modified_model_5pct.pth         # E4: 5% weight perturbation checkpoint
│   ├── modified_model_10pct.pth        # E5: 10% weight perturbation checkpoint
│   ├── modified_model_25pct.pth        # E6: 25% weight perturbation checkpoint
│   ├── modified_model_50pct.pth        # E7: 50% weight perturbation checkpoint
│   ├── independent_model.pth           # E8: Independently trained model (seed 808)
│   ├── random_model.pth                # E9: Randomly initialized model (seed 909)
│   └── modified_models_manifest.json   # Full parameter modification audit manifest
├── results/
│   └── experiments/                    # JSON, CSV, and PNG evaluation reports
├── src/
│   ├── challenges/                     # Module 1: Challenge set generation
│   │   ├── __init__.py
│   │   └── challenge_generator.py
│   ├── fingerprint/                    # Module 2: Fingerprint extraction
│   │   ├── __init__.py
│   │   └── fingerprint.py
│   ├── authentication/                 # Module 3: Authentication engine
│   │   ├── __init__.py
│   │   └── authenticator.py
│   └── attacks/                        # Module 4: Attack evaluation & metrics
│       ├── __init__.py
│       └── attack_evaluator.py
├── tests/                              # Automated test suite
│   ├── test_challenge_generator.py     # Unit tests for Module 1
│   ├── test_fingerprint.py             # Unit tests for Module 2
│   ├── test_authenticator.py           # Unit tests for Module 3
│   ├── test_pipeline.py                # Integration tests for end-to-end pipeline
│   └── test_attack_evaluator.py        # Unit tests for Module 4
├── run_authentication.py               # Central CLI authentication pipeline
├── train.py                            # Model training script
├── verify_model.py                     # Deterministic inference verification
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## Setup & Virtual Environment

### 1. Set Up Environment
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Automated Test Suite

Execute the complete test suite across all modules:
```bash
python -m pytest -v
```

---

## End-to-End Model Authentication Pipeline

The central pipeline runner (`run_authentication.py`) coordinates candidate model validation, fingerprint extraction, and metric evaluation.

### Execution Commands

#### 1. Authenticate the Genuine Original Model (E1)
```bash
python run_authentication.py --models models/original_model.pth
```

#### 2. Evaluate Multiple Candidate Models Simultaneously
```bash
python run_authentication.py --models models/original_model.pth models/modified_model_1pct.pth models/independent_model.pth
```

#### 3. Run with Custom Thresholds and Output Paths
```bash
python run_authentication.py \
    --models models/original_model.pth \
    --prediction_threshold 0.85 \
    --cosine_threshold 0.90 \
    --kl_threshold 0.15 \
    --output_dir results/experiments/
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--models` | `models/original_model.pth` | One or more candidate model paths (`.pth`) |
| `--original_model` | `models/original_model.pth` | Registered original model checkpoint |
| `--challenges` | `data/challenges/challenge_set.pt` | Challenge set file |
| `--reference_fingerprint` | `data/fingerprints/reference_fingerprint.pt` | Reference fingerprint file |
| `--prediction_threshold` | `0.85` | Agreement cutoff ($\ge 85\%$) |
| `--cosine_threshold` | `0.90` | Cosine similarity cutoff ($\ge 0.90$) |
| `--kl_threshold` | `0.15` | KL divergence cutoff ($\le 0.15$) |
| `--output_dir` | `results/` | Output directory for reports |
| `--output_json` | None | Custom JSON output path |
| `--output_csv` | None | Custom CSV output path |
| `--batch_size` | `64` | Inference batch size |
| `--device` | Auto-detect | Compute device (`cpu` or `cuda`) |

---

## Experimental Benchmark (E1–E9)

The protocol was evaluated across nine controlled model conditions:

| Exp | Model Condition | Agreement (>=85%) | Cosine (>=0.90) | KL Div (<=0.15) | Decision | Result |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **E1** | Original registered model | **100.00%** (PASS) | **1.0000** (PASS) | **0.0000** (PASS) | **AUTHENTIC** | True Acceptance |
| **E2** | Exact copy (re-saved) | **100.00%** (PASS) | **1.0000** (PASS) | **0.0000** (PASS) | **AUTHENTIC** | True Acceptance |
| **E3** | 1% weight perturbation | **83.00%** (FAIL) | **0.9887** (PASS) | **0.0194** (PASS) | **REJECTED** | True Rejection |
| **E4** | 5% weight perturbation | **74.00%** (FAIL) | **0.9562** (PASS) | **0.0837** (PASS) | **REJECTED** | True Rejection |
| **E5** | 10% weight perturbation | **72.00%** (FAIL) | **0.9007** (PASS) | **0.1902** (FAIL) | **REJECTED** | True Rejection |
| **E6** | 25% weight perturbation | **43.00%** (FAIL) | **0.7651** (FAIL) | **0.5326** (FAIL) | **REJECTED** | True Rejection |
| **E7** | 50% weight perturbation | **42.00%** (FAIL) | **0.5362** (FAIL) | **2.0234** (FAIL) | **REJECTED** | True Rejection |
| **E8** | Independently trained model | **37.00%** (FAIL) | **0.6878** (FAIL) | **0.8177** (FAIL) | **REJECTED** | True Rejection |
| **E9** | Randomly initialized model | **14.00%** (FAIL) | **0.5453** (FAIL) | **1.0979** (FAIL) | **REJECTED** | True Rejection |

To run the Member 4 attack evaluation and regenerate all summary figures:
```bash
python -m src.attacks.attack_evaluator
```

---

## Outputs & Evaluation Reports

| File | Description |
|---|---|
| `results/experiments/e1_e9_evaluation_report.json` | Comprehensive JSON report with confusion matrix, FAR, FRR, and per-trial metrics |
| `results/experiments/e1_e9_summary.csv` | Spreadsheet summary of all E1–E9 evaluation decisions and scores |
| `results/experiments/e1_e9_confusion_matrix.png` | 2×2 confusion matrix heatmap visualization |
| `results/experiments/e1_e9_metrics_comparison.png` | 3-panel behavioral metrics profile across all test conditions |
| `results/experiments/e1_e9_decisions.png` | Final authentication decision bar chart |

---

## License

This project is developed for academic and educational research purposes.
