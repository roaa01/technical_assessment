# Point-Supervised Remote Sensing Segmentation

**Technical Assessment** — Partial Focal CE Loss + U-Net + Experimental Study

---

## Project Structure

```
technical_assessment/
├── dataset/                  # Place real Potsdam data here (optional)
│   └── readme.md
├── src/
│   ├── __init__.py
│   ├── loss.py               # Task 1 — Partial Focal CE Loss
│   ├── dataset_sim.py        # Task 2 — Point-label simulation & dataset
│   ├── model.py              # Task 2 — U-Net segmentation network
│   └── train.py              # Task 3 — Training + experimental runner
├── notebooks/
│   └── exploration_and_experiments.ipynb
├── results/                  # Auto-created; holds JSON results & PNG plots
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run a single training experiment
```bash
cd technical_assessment
python src/train.py --experiment single --density 0.02 --gamma 2.0 --epochs 30
```

### 3. Reproduce Factor 1 (point density sweep)
```bash
python src/train.py --experiment density --epochs 25
```

### 4. Reproduce Factor 2 (focal gamma sweep)
```bash
python src/train.py --experiment gamma --epochs 25
```

### 5. Run all experiments
```bash
python src/train.py --experiment all --epochs 25
```

### 6. Open the Jupyter notebook
```bash
cd notebooks
jupyter notebook exploration_and_experiments.ipynb
```

---

## CLI Reference (`train.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `single` | `single` / `density` / `gamma` / `all` |
| `--density` | `0.02` | Fraction of pixels labeled (used in `single` mode) |
| `--gamma` | `2.0` | Focal γ exponent (used in `single` mode) |
| `--epochs` | `30` | Training epochs per run |
| `--batch_size` | `8` | Batch size |
| `--image_size` | `128` | Square image side length |
| `--n_train` | `200` | Training samples (synthetic) |
| `--n_val` | `40` | Validation samples (synthetic) |
| `--base_ch` | `32` | U-Net feature width multiplier |
| `--data_root` | `./dataset` | Path to real Potsdam data (falls back to synthetic) |
| `--out_dir` | `./results` | Where to save JSON result files |
| `--seed` | `42` | Random seed |

---

## Key Components

### `src/loss.py` — `PartialFocalCELoss`

Implements:
```
pfCE = Σ( FocalLoss(pred, GT) × MASK_labeled ) / Σ MASK_labeled
```
- `ignore_index` pixels are excluded from both numerator and denominator
- `gamma=0` recovers vanilla partial cross-entropy
- Supports `reduction='mean'|'sum'|'none'`

### `src/dataset_sim.py` — `SyntheticPotsdamDataset`

- Generates synthetic 6-class remote sensing scenes (ISPRS Potsdam taxonomy)
- `simulate_point_labels()` converts dense masks → sparse point annotations
- Two sampling strategies: `'random'` and `'stratified'` (class-proportional)

### `src/model.py` — `UNet`

- Encoder–decoder with skip connections, bilinear upsampling, BN+ReLU
- `base_ch=32` → ~7.8 M parameters; `base_ch=16` → ~2 M parameters
- Optional spatial dropout before the segmentation head

---

## Results Summary

### Factor 1 — Point Density (γ=2 fixed)

| Density | Best mIoU |
|---------|-----------|
| 0.5%    | 0.831     |
| 1.0%    | 0.869     |
| **2.0%**| **0.895** |
| 5.0%    | 0.867     |
| 10.0%   | 0.887     |

### Factor 2 — Focal γ (density=2% fixed)

| γ   | Best mIoU |
|-----|-----------|
| 0.0 | 0.857     |
| 0.5 | 0.878     |
| 1.0 | 0.890     |
| **2.0** | **0.895** |
| 5.0 | 0.873     |

**Best config:** density=2%, γ=2.0 → mIoU **0.895**

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision, numpy, matplotlib, pillow, scikit-learn, tqdm
