# Deep Learning Model Authentication Using Behavioral Challenge-Response

## Phase 1 – CIFAR-10 CNN Training

This phase trains a lightweight Convolutional Neural Network (CNN) on the **CIFAR-10** dataset. The trained model will serve as the *original model* for authentication experiments in later phases.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [CNN Architecture](#cnn-architecture)
4. [Training Process](#training-process)
5. [Project Structure](#project-structure)
6. [Setup & Execution](#setup--execution)
7. [Outputs](#outputs)
8. [Future Phases](#future-phases)

---

## Project Overview

The goal of this project is to explore **model authentication** by using behavioral challenge-response protocols — verifying that a deployed deep learning model is the genuine, authorized model and not a stolen or substitute copy.

**Phase 1** focuses solely on training a baseline CNN whose learned behavior (weights, internal representations, and output distributions) will later be fingerprinted and used for authentication.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | CIFAR-10 |
| **Images** | 60,000 colour images (32 × 32 px) |
| **Classes** | 10 — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| **Train / Test Split** | 50,000 / 10,000 (official split) |
| **Validation** | 10% of the training set (5,000 images) is held out |

**Data Augmentation (training only):**
- Random horizontal flip
- Random crop (32 × 32 with 4 px padding)

**Normalization (all splits):**
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
 │    Linear(64 × 8 × 8 → 256) → ReLU
 │    Dropout(0.5)
 │    Linear(256 → 10)
 │
 └─ Output (10 logits)
```

| Metric | Value |
|---|---|
| Total parameters | ~430 K |
| Trainable parameters | ~430 K |
| Expected validation accuracy | ~78 – 83% (10 epochs) |

The architecture is intentionally kept small so it can train in a few minutes on a laptop CPU or in under a minute on a Colab GPU.

---

## Training Process

| Hyper-parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 10 |
| Validation Split | 10% of training data |

- The model with the **best validation accuracy** across all epochs is saved.
- Training and validation loss curves plus validation accuracy are plotted and saved as `training_curves.png`.

---

## Project Structure

```
AI ML Project/
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── models/
│   ├── original_model.pth   # Best model weights (Phase 1)
│   └── model_config.json    # Architecture & class-name metadata
├── train.py                 # Training script (this phase)
├── training_curves.png      # Loss & accuracy plots (generated)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Setup & Execution

### 1. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you have an NVIDIA GPU and want CUDA acceleration, install the appropriate PyTorch build from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 3. Train the Model

```bash
python train.py
```

The script will:
1. Auto-detect GPU/CPU.
2. Download CIFAR-10 (first run only, ~170 MB).
3. Train for 10 epochs, printing loss and accuracy each epoch.
4. Save the best model to `models/original_model.pth`.
5. Save architecture metadata to `models/model_config.json`.
6. Save training curves to `training_curves.png`.

### Google Colab

Upload the project files, then run:

```python
!pip install -r requirements.txt
!python train.py
```

---

## Outputs

| File | Description |
|---|---|
| `models/original_model.pth` | PyTorch state dict of the best-performing model |
| `models/model_config.json` | JSON with architecture name, class names, and hyper-parameters |
| `training_curves.png` | Matplotlib figure with loss and accuracy over epochs |

---

## Future Phases

| Phase | Description |
|---|---|
| **Phase 2** | Behavioral fingerprinting – extract challenge-response pairs |
| **Phase 3** | Authentication protocol – verify model identity |
| **Phase 4** | Attack simulation – test against model theft and substitution |

---

## License

This project is for academic / research purposes.
