"""
Anatomically Guided Masking for iBOT pre-training on CT axial slices.

Implementation based on Curia-2 (arXiv 2604.01987):
    "We improve upon this strategy by implementing a Gaussian-weighted prior
     centered on anatomical regions. To maintain spatial consistency, the
     Gaussian centers are adjusted to account for the offsets introduced by
     global cropping."

Algorithm
---------
1. The body centroid (cy, cx) is detected in the original slice and mapped
   into crop-pixel coordinates by ContentAwareCropV2 (stored in `crop_meta`).
2. This mapped centroid is converted to patch-grid coordinates.
3. A 2-D Gaussian centered on the mapped centroid is used as the sampling
   probability for block-mask anchors — biasing masked regions towards
   anatomically informative patches rather than background.
4. Blocks are accumulated (RCC style) until mask_ratio × n_patches is reached.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


class AnatomicallyGuidedMasker:
    """
    Gaussian-weighted block mask generator (CURIA-2 style).

    Args:
        mask_ratio:       Target fraction of masked patches.
        gaussian_sigma:   Std of the Gaussian prior in patch units.
                          Defaults to 25 % of the grid width, giving a broad
                          but anatomically centred distribution.
        min_aspect:       Min aspect ratio of a rectangular block.
        max_aspect:       Max aspect ratio of a rectangular block.
        max_block_area:   Max fraction of total patches in one block.
        n_fallback_iters: Extra uniform-random iterations if target not met.
    """

    def __init__(
        self,
        mask_ratio:       float = 0.40,
        gaussian_sigma:   float | None = None,   # None → 0.25 × grid_width
        min_aspect:       float = 0.3,
        max_aspect:       float = 3.3,
        max_block_area:   float = 0.35,
        n_fallback_iters: int   = 50,
    ):
        self.mask_ratio       = mask_ratio
        self.gaussian_sigma   = gaussian_sigma
        self.min_aspect       = min_aspect
        self.max_aspect       = max_aspect
        self.max_block_area   = max_block_area
        self.n_fallback_iters = n_fallback_iters

    # ------------------------------------------------------------------

    def _gaussian_prior(
        self,
        n_h: int,
        n_w: int,
        cy_patch: float,
        cx_patch: float,
        sigma: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        2-D Gaussian over the patch grid, centred at (cy_patch, cx_patch).
        Returns a flat (n_h * n_w,) probability tensor.
        """
        ys = torch.arange(n_h, dtype=torch.float32, device=device)
        xs = torch.arange(n_w, dtype=torch.float32, device=device)
        # (n_h, n_w) grid of squared distances from the centroid
        dist2 = (ys[:, None] - cy_patch) ** 2 + (xs[None, :] - cx_patch) ** 2
        g = torch.exp(-0.5 * dist2 / (sigma ** 2))
        g = g.flatten()
        total = g.sum()
        if total < 1e-8:
            # Degenerate case: centroid far outside grid → uniform
            return torch.ones(n_h * n_w, device=device) / (n_h * n_w)
        return g / total

    def _sample_block(
        self,
        n_h: int,
        n_w: int,
        probs: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        """Sample a rectangular block using probs to bias the anchor patch."""
        n      = n_h * n_w
        anchor = int(torch.multinomial(probs, num_samples=1).item())
        ar, ac = anchor // n_w, anchor % n_w

        max_area = max(1, int(self.max_block_area * n))
        area     = random.randint(1, max_area)
        aspect   = random.uniform(self.min_aspect, self.max_aspect)

        h_blk = max(1, min(int((area * aspect) ** 0.5), n_h))
        w_blk = max(1, min(int((area / aspect) ** 0.5), n_w))

        top  = max(0, min(ar - h_blk // 2, n_h - h_blk))
        left = max(0, min(ac - w_blk // 2, n_w - w_blk))

        return top, left, h_blk, w_blk

    # ------------------------------------------------------------------

    def __call__(
        self,
        crop_meta: dict,
        patch_size: int,
        crop_size:  int,
    ) -> torch.Tensor:
        """
        Args:
            crop_meta:  dict produced by ContentAwareCropV2.__call__, containing
                        'body_cy_crop' and 'body_cx_crop' (centroid in crop pixels).
            patch_size: ViT patch size in pixels (e.g. 8).
            crop_size:  Side length of the global crop in pixels (e.g. 512).

        Returns:
            bool tensor (n_h * n_w,), True = masked.
        """
        n_h = crop_size // patch_size
        n_w = crop_size // patch_size
        n   = n_h * n_w
        target_masked = max(1, int(n * self.mask_ratio))

        device = torch.device("cpu")

        # Map centroid from crop pixels → patch grid
        cy_patch = crop_meta.get("body_cy_crop", n_h / 2.0) / patch_size
        cx_patch = crop_meta.get("body_cx_crop", n_w / 2.0) / patch_size

        sigma = self.gaussian_sigma if self.gaussian_sigma is not None \
                else 0.25 * n_w

        probs  = self._gaussian_prior(n_h, n_w, cy_patch, cx_patch, sigma, device)
        mask2d = torch.zeros(n_h, n_w, dtype=torch.bool)
        n_masked = 0

        for _ in range(100):
            if n_masked >= target_masked:
                break
            top, left, h_blk, w_blk = self._sample_block(n_h, n_w, probs)
            mask2d[top : top + h_blk, left : left + w_blk] = True
            n_masked = int(mask2d.sum())

        # Fallback: uniform random blocks
        if n_masked < target_masked:
            for _ in range(self.n_fallback_iters):
                if n_masked >= target_masked:
                    break
                aspect = random.uniform(self.min_aspect, self.max_aspect)
                area   = random.randint(1, max(1, int(self.max_block_area * n)))
                h_blk  = max(1, min(int((area * aspect) ** 0.5), n_h))
                w_blk  = max(1, min(int((area / aspect) ** 0.5), n_w))
                top    = random.randint(0, n_h - h_blk)
                left   = random.randint(0, n_w - w_blk)
                mask2d[top : top + h_blk, left : left + w_blk] = True
                n_masked = int(mask2d.sum())

        return mask2d.flatten()
