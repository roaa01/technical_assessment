"""
loss.py — Task 1
================
Implements Partial Focal Cross-Entropy Loss for point-supervised
remote sensing image segmentation.

Formula (from the assessment brief):
    pfCE = Σ( FocalLoss(pred, GT) × MASK_labeled ) / Σ MASK_labeled

The key idea: the segmentation mask is **incomplete** (only a sparse
set of pixels carry a label). The MASK_labeled binary tensor gates the
loss so that *only labeled pixels* contribute to the gradient.

Focal modulation (Lin et al., 2017) is added on top to down-weight
easy, well-classified pixels and focus training on the hard ones,
which is especially valuable when labels are sparse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFocalCELoss(nn.Module):
    """
    Partial Focal Cross-Entropy Loss.

    Parameters
    ----------
    gamma : float
        Focal exponent γ.  γ=0 reduces to vanilla partial CE.
        γ=2 is the standard focal default.
    alpha : float or list[float] or None
        Per-class weight α.  If a list, length must equal num_classes.
        None → uniform weights.
    ignore_index : int
        Label value marking *unlabeled* pixels.  These pixels are
        **excluded** from the loss — this is the "partial" mechanism.
    reduction : str
        'mean'  → divide by number of labeled pixels  (default)
        'sum'   → raw sum over labeled pixels
        'none'  → returns a per-pixel loss map (unlabeled = 0)
    eps : float
        Numerical stability clamp to avoid log(0).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        ignore_index: int = -1,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

        # Register alpha as a buffer so it moves with .to(device)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)  — raw network output (un-normalised)
        targets : (B, H, W)     — integer class labels; pixels with
                                  value == ignore_index are unlabeled
        Returns
        -------
        Scalar loss (or per-pixel map if reduction='none').
        """
        B, C, H, W = logits.shape

        # ── 1. Build the binary labeled-pixel mask ─────────────────────
        # MASK_labeled  shape: (B, H, W),  dtype bool
        mask_labeled = targets != self.ignore_index          # True where labeled

        if mask_labeled.sum() == 0:
            # No labeled pixels in this batch → zero loss, no NaN
            return logits.sum() * 0.0

        # ── 2. Cross-entropy per pixel ─────────────────────────────────
        # We compute log-softmax manually so we can also get probs for
        # the focal weight.
        log_prob = F.log_softmax(logits, dim=1)              # (B, C, H, W)
        prob     = log_prob.exp()                            # (B, C, H, W)

        # Clamp targets at 0 for gather (ignore_index pixels will be
        # masked out afterwards, so their gathered value doesn't matter).
        safe_targets = targets.clone()
        safe_targets[~mask_labeled] = 0                      # harmless placeholder

        # Gather log-prob and prob of the ground-truth class
        # shape after gather + squeeze: (B, H, W)
        gt_idx       = safe_targets.unsqueeze(1)             # (B, 1, H, W)
        log_pt       = log_prob.gather(1, gt_idx).squeeze(1) # (B, H, W)
        pt           = prob.gather(1, gt_idx).squeeze(1)     # (B, H, W)

        # ── 3. Focal weight  (1 − p_t)^γ ─────────────────────────────
        focal_weight = (1.0 - pt.clamp(self.eps, 1.0 - self.eps)) ** self.gamma

        # ── 4. Per-class alpha weight ──────────────────────────────────
        if self.alpha is not None:
            alpha_map = self.alpha[safe_targets]             # (B, H, W)
        else:
            alpha_map = torch.ones_like(pt)

        # ── 5. Per-pixel focal CE  ─────────────────────────────────────
        per_pixel_loss = -alpha_map * focal_weight * log_pt  # (B, H, W)

        # ── 6. Apply MASK_labeled  (the "partial" gating) ─────────────
        #   pfCE = Σ( FocalLoss(pred, GT) × MASK_labeled ) / Σ MASK_labeled
        masked_loss = per_pixel_loss * mask_labeled.float()

        if self.reduction == "none":
            return masked_loss

        total_labeled = mask_labeled.float().sum()

        if self.reduction == "sum":
            return masked_loss.sum()

        # default: mean over labeled pixels
        return masked_loss.sum() / total_labeled.clamp(min=1.0)


# ──────────────────────────────────────────────────────────────────────
# Quick sanity-check (run directly: python loss.py)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 2, 6, 64, 64
    IGNORE = -1
    POINT_DENSITY = 0.05          # 5 % of pixels labeled

    logits  = torch.randn(B, C, H, W)
    targets = torch.full((B, H, W), IGNORE, dtype=torch.long)

    # Randomly label ~5 % of pixels
    mask = torch.rand(B, H, W) < POINT_DENSITY
    targets[mask] = torch.randint(0, C, (int(mask.sum().item()),))

    # ── Test γ sweep ────────────────────────────────────────────────
    print("=" * 55)
    print("Partial Focal CE Loss — sanity check")
    print("=" * 55)
    print(f"Batch: {B}×{C}×{H}×{W}   labeled pixels: {mask.sum().item()}")
    print("-" * 55)
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        loss_fn = PartialFocalCELoss(gamma=gamma, ignore_index=IGNORE)
        val     = loss_fn(logits, targets)
        print(f"  γ = {gamma:.1f}   loss = {val.item():.6f}")

    # ── Test reduction modes ────────────────────────────────────────
    print("-" * 55)
    for mode in ["mean", "sum", "none"]:
        loss_fn = PartialFocalCELoss(gamma=2.0, ignore_index=IGNORE, reduction=mode)
        val     = loss_fn(logits, targets)
        shape   = val.shape if hasattr(val, "shape") else "()"
        print(f"  reduction='{mode}'   shape={shape}   val={val.mean().item():.4f}")

    # ── Edge-case: all pixels unlabeled ────────────────────────────
    print("-" * 55)
    all_unlabeled = torch.full((B, H, W), IGNORE, dtype=torch.long)
    loss_fn = PartialFocalCELoss(gamma=2.0, ignore_index=IGNORE)
    val = loss_fn(logits, all_unlabeled)
    print(f"  All-unlabeled batch → loss = {val.item():.6f}  (should be 0.0)")
    print("=" * 55)
