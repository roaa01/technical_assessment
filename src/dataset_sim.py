"""
dataset_sim.py — Task 2
========================
Simulates a remote sensing image segmentation dataset with point-level
(sparse) supervision.

Dataset used: ISPRS Potsdam (6-class land-use segmentation).
If the real dataset is unavailable, a **synthetic generator** is
provided so the pipeline can run end-to-end without downloads.

Point-label simulation strategy
---------------------------------
For each training image we randomly sample `point_density` fraction
of the *labeled* pixels and set all others to `ignore_index`.
This mimics the real-world scenario where annotators place a single
click per object rather than drawing full masks.

Classes (Potsdam):
    0 – Impervious surfaces (roads, pavements)
    1 – Building
    2 – Low vegetation
    3 – Tree
    4 – Car
    5 – Clutter / background
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

NUM_CLASSES  = 6
IGNORE_INDEX = -1

# RGB colour map used by ISPRS Potsdam for visualisation
POTSDAM_PALETTE = np.array([
    [255, 255, 255],   # 0 Impervious surfaces
    [0,   0,   255],   # 1 Building
    [0,   255, 255],   # 2 Low vegetation
    [0,   255, 0  ],   # 3 Tree
    [255, 255, 0  ],   # 4 Car
    [255, 0,   0  ],   # 5 Clutter
], dtype=np.uint8)


def rgb_mask_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an H×W×3 RGB mask to an H×W integer class map."""
    h, w, _ = mask_rgb.shape
    class_map = np.zeros((h, w), dtype=np.int64)
    for cls_idx, colour in enumerate(POTSDAM_PALETTE):
        match = np.all(mask_rgb == colour, axis=-1)
        class_map[match] = cls_idx
    return class_map


def simulate_point_labels(
    full_mask: torch.Tensor,
    point_density: float = 0.02,
    min_points: int = 10,
    ignore_index: int = IGNORE_INDEX,
    strategy: str = "random",           # 'random' | 'stratified'
) -> torch.Tensor:
    """
    Convert a full dense segmentation mask into a sparse point-label mask.

    Parameters
    ----------
    full_mask     : (H, W) integer tensor  — ground-truth class per pixel
    point_density : fraction of pixels to keep as labeled
    min_points    : absolute minimum labeled pixels per image
    ignore_index  : value assigned to unlabeled pixels
    strategy      : sampling strategy
        'random'     — uniform random sample over all labeled pixels
        'stratified' — sample proportionally per class (guarantees each
                       class is represented if present)

    Returns
    -------
    point_mask : (H, W) integer tensor — sparse labels
    """
    H, W    = full_mask.shape
    n_total = H * W
    n_keep  = max(min_points, int(n_total * point_density))

    point_mask = torch.full_like(full_mask, fill_value=ignore_index)

    if strategy == "stratified":
        # Sample proportionally from each class
        selected = []
        classes  = full_mask.unique()
        per_cls  = max(1, n_keep // len(classes))
        for cls in classes:
            indices = (full_mask == cls).nonzero(as_tuple=False)
            n_pick  = min(per_cls, len(indices))
            perm    = torch.randperm(len(indices))[:n_pick]
            selected.append(indices[perm])
        selected = torch.cat(selected, dim=0)
        point_mask[selected[:, 0], selected[:, 1]] = \
            full_mask[selected[:, 0], selected[:, 1]]
    else:
        # Uniform random
        flat_indices  = torch.randperm(n_total)[:n_keep]
        row_idx = flat_indices // W
        col_idx = flat_indices %  W
        point_mask[row_idx, col_idx] = full_mask[row_idx, col_idx]

    return point_mask


# ──────────────────────────────────────────────────────────────────────
# Synthetic dataset (no download required)
# ──────────────────────────────────────────────────────────────────────

class SyntheticPotsdamDataset(Dataset):
    """
    Procedurally generated remote sensing imagery that mimics the
    spatial statistics of ISPRS Potsdam without requiring the real data.

    Each image contains smooth blobs of different land-cover classes
    rendered with class-typical RGB colours + Gaussian noise.
    """

    def __init__(
        self,
        n_samples: int = 200,
        image_size: int = 256,
        point_density: float = 0.02,
        min_points: int = 50,
        ignore_index: int = IGNORE_INDEX,
        transform: Optional[Callable] = None,
        split: str = "train",
        sampling_strategy: str = "stratified",
        seed: int = 42,
    ):
        self.n_samples          = n_samples
        self.image_size         = image_size
        self.point_density      = point_density
        self.min_points         = min_points
        self.ignore_index       = ignore_index
        self.transform          = transform
        self.split              = split
        self.sampling_strategy  = sampling_strategy
        self.seed               = seed

        # Pre-generate all images for reproducibility
        rng = np.random.default_rng(seed + (0 if split == "train" else 9999))
        self.images, self.full_masks = self._generate_all(rng)

    # ── generation helpers ─────────────────────────────────────────
    def _generate_all(self, rng):
        images, masks = [], []
        for _ in range(self.n_samples):
            img, mask = self._make_scene(rng)
            images.append(img)
            masks.append(mask)
        return images, masks

    def _make_scene(self, rng):
        """Generate one synthetic H×W scene."""
        S   = self.image_size
        mask_rgb = np.zeros((S, S, 3), dtype=np.uint8)

        # Fill background with class 0 colour
        mask_rgb[:] = POTSDAM_PALETTE[0]

        # Drop random elliptical blobs for each class
        n_blobs = rng.integers(8, 20)
        for _ in range(n_blobs):
            cls  = rng.integers(0, NUM_CLASSES)
            cy   = rng.integers(0, S)
            cx   = rng.integers(0, S)
            ry   = rng.integers(S // 16, S // 4)
            rx   = rng.integers(S // 16, S // 4)
            ys, xs = np.ogrid[:S, :S]
            blob = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1
            mask_rgb[blob] = POTSDAM_PALETTE[cls]

        # Build integer class map
        class_map = rgb_mask_to_class(mask_rgb)

        # Synthesise RGB image: class mean colour + Gaussian noise
        img = np.zeros((S, S, 3), dtype=np.float32)
        for cls_idx, colour in enumerate(POTSDAM_PALETTE):
            where = class_map == cls_idx
            img[where] = colour / 255.0
        img += rng.normal(0, 0.05, img.shape).astype(np.float32)
        img  = img.clip(0, 1)

        return img, class_map

    # ── Dataset interface ──────────────────────────────────────────
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_np   = self.images[idx]                           # H×W×3 float32
        full_msk = torch.from_numpy(self.full_masks[idx]).long()

        # Convert image to tensor
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)    # 3×H×W

        if self.transform is not None:
            img_t = self.transform(img_t)

        # Simulate point labels (only for training split)
        if self.split == "train":
            point_msk = simulate_point_labels(
                full_msk,
                point_density      = self.point_density,
                min_points         = self.min_points,
                ignore_index       = self.ignore_index,
                strategy           = self.sampling_strategy,
            )
        else:
            # Validation / test: use the full mask for evaluation
            point_msk = full_msk

        return img_t, point_msk, full_msk


# ──────────────────────────────────────────────────────────────────────
# Real Potsdam dataset wrapper (optional — needs local data)
# ──────────────────────────────────────────────────────────────────────

class PotsdamDataset(Dataset):
    """
    Wraps the real ISPRS Potsdam dataset.

    Expected directory layout:
        root/
          images/   *.png   (RGB or IRRG tiles)
          labels/   *.png   (RGB label masks)

    Falls back to SyntheticPotsdamDataset if root does not exist.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        point_density: float = 0.02,
        min_points: int = 50,
        image_size: int = 256,
        ignore_index: int = IGNORE_INDEX,
        sampling_strategy: str = "stratified",
    ):
        root = Path(root)
        if not root.exists():
            print(
                f"[PotsdamDataset] '{root}' not found — "
                "using synthetic data instead."
            )
            n = 200 if split == "train" else 40
            self._delegate = SyntheticPotsdamDataset(
                n_samples          = n,
                image_size         = image_size,
                point_density      = point_density,
                min_points         = min_points,
                ignore_index       = ignore_index,
                split              = split,
                sampling_strategy  = sampling_strategy,
            )
        else:
            self._delegate = None
            self._load_real(
                root, split, point_density, min_points,
                image_size, ignore_index, sampling_strategy,
            )

    def _load_real(self, root, split, point_density, min_points,
                   image_size, ignore_index, sampling_strategy):
        from PIL import Image

        img_dir  = root / "images"
        lbl_dir  = root / "labels"
        all_imgs = sorted(img_dir.glob("*.png"))
        n        = len(all_imgs)
        split_n  = int(n * 0.8)

        if split == "train":
            self._paths = all_imgs[:split_n]
        else:
            self._paths = all_imgs[split_n:]

        self.lbl_dir           = lbl_dir
        self.image_size        = image_size
        self.point_density     = point_density
        self.min_points        = min_points
        self.ignore_index      = ignore_index
        self.split             = split
        self.sampling_strategy = sampling_strategy

    def __len__(self):
        if self._delegate:
            return len(self._delegate)
        return len(self._paths)

    def __getitem__(self, idx):
        if self._delegate:
            return self._delegate[idx]

        from PIL import Image
        import torchvision.transforms.functional as TF

        img_path = self._paths[idx]
        lbl_path = self.lbl_dir / img_path.name

        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("RGB")

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        lbl = lbl.resize((self.image_size, self.image_size), Image.NEAREST)

        img_t  = TF.to_tensor(img)
        lbl_np = np.array(lbl)
        full_msk = torch.from_numpy(rgb_mask_to_class(lbl_np)).long()

        if self.split == "train":
            point_msk = simulate_point_labels(
                full_msk,
                point_density  = self.point_density,
                min_points     = self.min_points,
                ignore_index   = self.ignore_index,
                strategy       = self.sampling_strategy,
            )
        else:
            point_msk = full_msk

        return img_t, point_msk, full_msk


# ──────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = SyntheticPotsdamDataset(n_samples=4, image_size=128,
                                 point_density=0.02, split="train")
    img, pt_msk, full_msk = ds[0]
    labeled = (pt_msk != IGNORE_INDEX).float().mean().item()
    print(f"Image shape : {img.shape}")
    print(f"Full mask   : {full_msk.shape}  classes {full_msk.unique().tolist()}")
    print(f"Point mask  : {pt_msk.shape}   labeled {labeled*100:.2f}% of pixels")
