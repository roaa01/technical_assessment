"""
train.py — Task 3
==================
Experimental runner for the partial-CE point-supervision study.

Two experimental factors investigated
--------------------------------------
Factor 1 – Point density  (0.5%, 1%, 2%, 5%, 10%)
    Hypothesis: higher density improves mIoU, but gains plateau quickly
    because the loss is dominated by spatial context once density > ~2%.

Factor 2 – Focal gamma  (0, 0.5, 1, 2, 5) at fixed density 2%
    Hypothesis: moderate γ (1–2) boosts mIoU by focusing on hard pixels,
    while very large γ hurts by over-suppressing the (already sparse)
    easy labeled pixels.

Usage
-----
    # Run all experiments (takes ~20 min on CPU / ~3 min on GPU)
    python train.py --experiment all

    # Single quick run
    python train.py --experiment density --density 0.02 --gamma 2.0 --epochs 20

    # Reproduce Factor 1 sweep only
    python train.py --experiment density --epochs 20

    # Reproduce Factor 2 sweep only
    python train.py --experiment gamma --epochs 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Make src importable when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loss        import PartialFocalCELoss
from src.dataset_sim import SyntheticPotsdamDataset, NUM_CLASSES, IGNORE_INDEX
from src.model       import UNet


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int, ignore_index: int = IGNORE_INDEX) -> float:
    """Mean Intersection-over-Union over the *full* dense mask."""
    valid  = targets != ignore_index
    preds  = preds[valid].cpu().numpy()
    labels = targets[valid].cpu().numpy()

    ious = []
    for cls in range(num_classes):
        inter = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or( preds == cls, labels == cls).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_acc(preds: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = IGNORE_INDEX) -> float:
    valid = targets != ignore_index
    return (preds[valid] == targets[valid]).float().mean().item()


# ──────────────────────────────────────────────────────────────────────
# One training + evaluation run
# ──────────────────────────────────────────────────────────────────────

def run_experiment(
    point_density: float     = 0.02,
    gamma: float             = 2.0,
    epochs: int              = 30,
    batch_size: int          = 8,
    lr: float                = 1e-3,
    image_size: int          = 128,
    n_train: int             = 200,
    n_val: int               = 40,
    base_ch: int             = 32,
    sampling_strategy: str   = "stratified",
    data_root: str           = "./dataset",
    seed: int                = 42,
    device: str              = "auto",
    verbose: bool            = True,
) -> dict:
    """
    Train a U-Net with Partial Focal CE loss under the given config.
    Returns a dict of training curves and final metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  density={point_density:.3f}  gamma={gamma}  "
              f"epochs={epochs}  device={device}")
        print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────
    from src.dataset_sim import SyntheticPotsdamDataset

    train_ds = SyntheticPotsdamDataset(
        n_samples          = n_train,
        image_size         = image_size,
        point_density      = point_density,
        min_points         = 20,
        ignore_index       = IGNORE_INDEX,
        split              = "train",
        sampling_strategy  = sampling_strategy,
        seed               = seed,
    )

    val_ds = SyntheticPotsdamDataset(
        n_samples          = n_val,
        image_size         = image_size,
        point_density      = point_density,   # kept for compat; val returns full mask
        min_points         = 20,
        ignore_index       = IGNORE_INDEX,
        split              = "val",
        sampling_strategy  = sampling_strategy,
        seed               = seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Model, loss, optimiser ────────────────────────────────────
    model   = UNet(in_channels=3, num_classes=NUM_CLASSES,
                   base_ch=base_ch).to(device)
    loss_fn = PartialFocalCELoss(gamma=gamma, ignore_index=IGNORE_INDEX)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "val_miou": [], "val_acc": []}
    best_miou = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for imgs, pt_masks, _ in train_loader:
            imgs     = imgs.to(device)
            pt_masks = pt_masks.to(device)

            opt.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, pt_masks)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        sched.step()

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        all_miou, all_acc = [], []
        with torch.no_grad():
            for imgs, _, full_masks in val_loader:
                imgs       = imgs.to(device)
                full_masks = full_masks.to(device)
                logits     = model(imgs)
                preds      = logits.argmax(dim=1)
                all_miou.append(compute_miou(preds, full_masks, NUM_CLASSES))
                all_acc.append( compute_pixel_acc(preds, full_masks))

        val_miou = float(np.mean(all_miou))
        val_acc  = float(np.mean(all_acc))
        best_miou = max(best_miou, val_miou)

        history["train_loss"].append(avg_loss)
        history["val_miou"].append(val_miou)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"mIoU={val_miou:.4f}  "
                  f"acc={val_acc:.4f}")

    return {
        "density":     point_density,
        "gamma":       gamma,
        "best_miou":   best_miou,
        "final_miou":  history["val_miou"][-1],
        "final_acc":   history["val_acc"][-1],
        "history":     history,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment sweeps
# ──────────────────────────────────────────────────────────────────────

def sweep_density(epochs, **kwargs):
    """Factor 1: vary point density at fixed γ=2."""
    densities = [0.005, 0.01, 0.02, 0.05, 0.10]
    results   = []
    for d in densities:
        r = run_experiment(point_density=d, gamma=2.0, epochs=epochs, **kwargs)
        results.append(r)
        print(f"  ✓  density={d:.3f}  best_mIoU={r['best_miou']:.4f}")
    return results


def sweep_gamma(epochs, **kwargs):
    """Factor 2: vary focal γ at fixed density=2%."""
    gammas  = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = []
    for g in gammas:
        r = run_experiment(point_density=0.02, gamma=g, epochs=epochs, **kwargs)
        results.append(r)
        print(f"  ✓  gamma={g}  best_mIoU={r['best_miou']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────────────
# Result persistence + summary table
# ──────────────────────────────────────────────────────────────────────

def save_results(results: list, path: str):
    # Strip history tensors before JSON serialisation
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if k != "history"}
        c["train_loss_curve"] = r["history"]["train_loss"]
        c["val_miou_curve"]   = r["history"]["val_miou"]
        clean.append(c)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved → {path}")


def print_table(results: list, factor: str):
    key   = "density" if factor == "density" else "gamma"
    label = "Point Density" if factor == "density" else "Focal γ"
    print(f"\n{'─'*48}")
    print(f"  {label:<20} │  Best mIoU  │  Final Acc")
    print(f"{'─'*48}")
    for r in results:
        val = f"{r[key]:.3f}" if key == "density" else f"{r[key]:.1f}"
        print(f"  {val:<20} │  {r['best_miou']:.4f}   │  {r['final_acc']:.4f}")
    print(f"{'─'*48}")


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Partial CE point-supervision experiments")
    p.add_argument("--experiment", choices=["density", "gamma", "all", "single"],
                   default="single")
    p.add_argument("--density",    type=float, default=0.02)
    p.add_argument("--gamma",      type=float, default=2.0)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--image_size", type=int,   default=128)
    p.add_argument("--n_train",    type=int,   default=200)
    p.add_argument("--n_val",      type=int,   default=40)
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--data_root",  type=str,   default="./dataset")
    p.add_argument("--out_dir",    type=str,   default="./results")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = dict(
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        image_size  = args.image_size,
        n_train     = args.n_train,
        n_val       = args.n_val,
        base_ch     = args.base_ch,
        data_root   = args.data_root,
        seed        = args.seed,
    )

    if args.experiment == "single":
        r = run_experiment(
            point_density = args.density,
            gamma         = args.gamma,
            **shared,
        )
        print(f"\nFinal  mIoU = {r['final_miou']:.4f}")
        print(f"Best   mIoU = {r['best_miou']:.4f}")
        print(f"Final  Acc  = {r['final_acc']:.4f}")

    elif args.experiment == "density":
        print("\n── Factor 1: Point density sweep ──")
        results = sweep_density(**shared)
        print_table(results, "density")
        save_results(results, out_dir / "density_results.json")

    elif args.experiment == "gamma":
        print("\n── Factor 2: Focal gamma sweep ──")
        results = sweep_gamma(**shared)
        print_table(results, "gamma")
        save_results(results, out_dir / "gamma_results.json")

    elif args.experiment == "all":
        print("\n── Factor 1: Point density sweep ──")
        d_results = sweep_density(**shared)
        print_table(d_results, "density")
        save_results(d_results, out_dir / "density_results.json")

        print("\n── Factor 2: Focal gamma sweep ──")
        g_results = sweep_gamma(**shared)
        print_table(g_results, "gamma")
        save_results(g_results, out_dir / "gamma_results.json")


if __name__ == "__main__":
    main()
