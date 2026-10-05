# BloodMNIST CNN Classifier

A lightweight image classification project built for a resource-constrained,
edge-deployment setting. Two custom convolutional neural networks are trained
and compared on **BloodMNIST** (part of the MedMNIST+ collection) — a
**standard CNN (Model A)** and a **depthwise-separable lightweight CNN
(Model B)** — implemented from scratch in PyTorch.

> Developed for EN3150 Assignment 03 — *Resource-Constrained CNN for Edge
> Image Classification*.

## Overview

- **Dataset:** BloodMNIST (MedMNIST+), 8 classes of peripheral blood cell
  images, 64×64 RGB
- **Custom split:** stratified 70% / 15% / 15% train/val/test
- **Model A:** standard 2D convolutions interleaved with max-pooling
- **Model B:** depthwise separable convolutions for a far smaller parameter
  footprint, aimed at edge/microcontroller-scale deployment
- **Evaluation:** macro-averaged precision/recall/F1/ROC-AUC (to account for
  class imbalance), plus a full per-class breakdown and confusion matrix

## Project Structure

```
.
├── Data_prep.py          # Downloads BloodMNIST and builds the custom 70/15/15 split
├── Data_loader.py        # PyTorch Dataset + DataLoader (normalization, augmentation)
├── Data_Analysis.py      # EDA on the prepared split (class balance, pixel stats)
├── NodeConvs_A_Net.py    # Model A -- standard CNN
├── NodeConvs_B_Net.py    # Model B -- depthwise-separable lightweight CNN
├── Train.py               # Training loop, checkpointing, metric tracking
├── Test.py                 # Test-set evaluation and reporting
├── BloodMNIST/              # (create this) prepared dataset split
│   └── bloodmnist_70_15_15.npz
├── Weights/                  # (create this) model checkpoints
│   └── Best_modelA_fine_tuned.pth
│   └── Best_modelB_fine_tuned.pth
|   
└── README.md
```

## Dataset

[BloodMNIST](https://medmnist.com/) contains peripheral blood cell
microscopy images across 8 classes: `basophil`, `eosinophil`,
`erythroblast`, `immature granulocytes`, `lymphocyte`, `monocyte`,
`neutrophil`, `platelet`. Images are downscaled to 64×64 RGB.

Rather than using MedMNIST's official split directly, `Data_prep.py` builds
a custom **70/15/15** split: the official train split (already ≈70% of the
data) is kept untouched, and the official val + test splits are merged and
re-divided 50/50 (stratified) into the new val/test sets. This avoids
disturbing any train-boundary separation in the original data while still
hitting the target ratios.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- medmnist

```bash
pip install torch numpy scikit-learn matplotlib medmnist
```

## Setup

### 1. Prepare the dataset

```bash
mkdir BloodMNIST
cd BloodMNIST
python ../Data_prep.py
cd ..
```

This downloads BloodMNIST via the `medmnist` package and writes
`bloodmnist_70_15_15.npz` inside `BloodMNIST/`, matching the path
`Data_loader.py` expects by default.

### 2. (Optional) Explore the data

```bash
python Data_Analysis.py
```

Prints shape/dtype sanity checks, a NaN/blank-image check, class
distribution per split, the class-imbalance ratio, and the per-channel
normalization statistics used by `Data_loader.py`.

### 3. Add pretrained weights

```bash
mkdir Weights
```

Download the pretrained Model checkpoints from Google Drive:

**[Download Model Weights](https://drive.google.com/drive/folders/1Ne8R_WzfamynKc-2GJeZ9bEGDqFK0CG-?usp=sharing)**

Place the downloaded file inside the `Weights/` folder so the path matches
what `Test.py` expects:

```
Weights/Best_modelA_fine_tuned.pth
```

### 4. Train a model

```bash
python Train.py
```

Trains whichever model is currently selected in `Train.py` (`model_A` or
`model_B`), tracks loss, accuracy, macro ROC-AUC, and macro F1 per epoch,
and checkpoints the best-performing weights (by validation macro F1) to
disk.

### 5. Evaluate on the test set

```bash
python Test.py
```

Loads the Model A checkpoint from `Weights/Best_modelA_fine_tuned.pth` and
reports test accuracy, inference throughput, macro ROC-AUC/F1/precision/
recall, a full per-class classification report, and the confusion matrix.

## Model Architectures

### Model A — Standard CNN (`NodeConvs_A_Net.py`)

Each level runs two `Conv2d → BatchNorm2d → ReLU` stages before
max-pooling, ending in Global Average Pooling and a fully-connected
classifier head (outputs raw logits, paired with `CrossEntropyLoss`).

| | |
|---|---|
| Configuration used in `Train.py` / `Test.py` | `base_channels=32`, `levels=3` |
| Trainable parameters | **≈ 289,000** |

### Model B — Lightweight CNN (`NodeConvs_B_Net.py`)

Same overall structure as Model A, but every convolution is replaced with a
**depthwise separable convolution** (a per-channel depthwise pass followed
by a 1×1 pointwise pass), substantially cutting multiply-accumulate
operations for edge deployment.

| | |
|---|---|
| Configuration used in `Train.py` / `Test.py` | `base_channels=32`, `levels=3` |
| Trainable parameters | **≈ 37,000** (well under the 100K edge-deployment target) |

## Evaluation Metrics

The dataset has a moderate class imbalance (~2.7× between the largest and
smallest classes), so model selection and evaluation lean on
**macro-averaged** precision, recall, F1, and ROC-AUC — each class
contributes equally, so performance isn't dominated by the majority
classes. `Test.py` also prints a full per-class `classification_report`
and the confusion matrix for a complete breakdown.

## Known Issues / Before You Run

A few things in the current scripts need attention before the commands
above will run cleanly:

- **Import name mismatch:** `Train.py` and `Test.py` both import
  `from NodeConvs_Net import NodeConvs_Net`, but the file in this repo is
  named `NodeConvs_A_Net.py`. Update that line to
  `from NodeConvs_A_Net import NodeConvs_Net` in both files.
- **`Train.py` expects an existing checkpoint:** it currently loads
  `Best_modelB_2.pth` via `torch.load(...)` before training even starts, so
  a fresh run with no prior checkpoint will fail. Remove or comment out
  that block if you're training from scratch.
- **Checkpoint location consistency:** `Train.py` saves/loads checkpoints
  from the working directory directly, while `Test.py` expects them inside
  `Weights/`. Move a checkpoint into `Weights/` manually before running
  `Test.py` on it.