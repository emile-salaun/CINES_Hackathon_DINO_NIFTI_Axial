"""
DINOv3 loss functions.

Components:
  - DINOLoss:    cross-entropy between teacher and student CLS token distributions.
  - iBOTLoss:    cross-entropy between teacher and student patch token distributions
                 (at masked positions only).
  - sigreg_loss: covariance regularisation on L2-normalised embeddings
                 (replaces KoLeo from FlexiCT — Modification 3).
  - gram_loss:   Gram-matrix consistency for hi-res fine-tuning (Phase 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


# ---------------------------------------------------------------------------
# DINO loss (CLS-level)
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """
    Cross-entropy between teacher and student softmax distributions.

    Teacher outputs are sharpened (low temperature) and centred.
    Student outputs use a higher temperature.

    Centering is updated via EMA of teacher batch statistics (distributed-safe).
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.teacher_temp    = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        """EMA update of the centering buffer."""
        # teacher_logits: (B, K) from one teacher crop
        batch_center = teacher_logits.mean(dim=0, keepdim=True)    # (1, K)
        batch_center = _all_reduce_mean(batch_center)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(
        self,
        teacher_logits: list[torch.Tensor],  # n_teacher crops × (B, K)
        student_logits: list[torch.Tensor],  # n_student crops × (B, K)
    ) -> torch.Tensor:
        """
        Each teacher crop is matched against every student crop except itself.
        """
        n_teacher = len(teacher_logits)
        n_student = len(student_logits)

        # Update centre with teacher batch statistics
        for tl in teacher_logits:
            self._update_center(tl.detach())

        total_loss = torch.tensor(0.0, device=teacher_logits[0].device)
        n_pairs = 0

        for t_idx, tl in enumerate(teacher_logits):
            centered = (tl - self.center) / self.teacher_temp
            t_prob = F.softmax(centered, dim=-1).detach()
            for s_idx, sl in enumerate(student_logits):
                if s_idx == t_idx:
                    continue
                s_log_prob = F.log_softmax(sl / self.student_temp, dim=-1)
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss = total_loss + loss
                n_pairs += 1

        return total_loss / max(n_pairs, 1)


# ---------------------------------------------------------------------------
# iBOT loss (patch-level)
# ---------------------------------------------------------------------------

class iBOTLoss(nn.Module):
    """
    Cross-entropy between teacher patch token distributions and student predictions
    at masked positions.

    Uses a separate centering buffer from DINOLoss (patch vs CLS statistics differ).
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.teacher_temp    = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def _update_center(self, teacher_patch_logits: torch.Tensor) -> None:
        # teacher_patch_logits: (B, N, K)
        batch_center = teacher_patch_logits.mean(dim=(0, 1), keepdim=True)  # (1, 1, K)
        batch_center = _all_reduce_mean(batch_center)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(
        self,
        teacher_patch_logits: list[torch.Tensor],  # n_global × (B, N, K)
        student_patch_logits: list[torch.Tensor],  # n_global × (B, N, K)
        masks: list[torch.Tensor],                 # n_global × (B, N) bool  (True = masked)
    ) -> torch.Tensor:
        total_loss  = torch.tensor(0.0, device=teacher_patch_logits[0].device)
        total_count = 0

        for tl, sl, mask in zip(teacher_patch_logits, student_patch_logits, masks):
            # tl, sl: (B, N, K);  mask: (B, N) bool
            self._update_center(tl.detach())

            t_prob    = F.softmax((tl - self.center) / self.teacher_temp, dim=-1).detach()
            s_logprob = F.log_softmax(sl / self.student_temp, dim=-1)

            # Only compute loss at masked positions
            # mask: (B, N) → (B, N, 1) for broadcasting
            mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
            token_loss = -(t_prob * s_logprob).sum(dim=-1)  # (B, N)
            masked_loss = (token_loss * mask.float()).sum()
            count       = mask.float().sum().clamp(min=1)

            total_loss  = total_loss + masked_loss
            total_count += count.item()

        return total_loss / max(total_count, 1)


# ---------------------------------------------------------------------------
# KoLeo regulariser  (Kozachenko-Leonenko estimator)
#
# Reference implementation: github.com/facebookresearch/dinov3
#
# Distributed note: for multi-GPU training all_gather is applied before NN
# search so that each GPU sees the full cross-replica batch.
# ---------------------------------------------------------------------------

class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic regulariser (DINOv2/v3 / FlexiCT).

    Pushes embeddings to span the hypersphere uniformly.
    Weight in FlexiCT: 0.1  (same as the previous sigreg_weight).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2, eps=eps)
        self.eps   = eps

    def _pairwise_nn_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Return index of nearest neighbour for each row of x (L2-normed)."""
        dots = torch.mm(x, x.t())              # cosine similarities
        n    = x.shape[0]
        # Zero out the diagonal so a point isn't its own nearest neighbour
        dots.view(-1)[:: (n + 1)].fill_(-1)
        _, indices = torch.max(dots, dim=1)
        return indices

    def forward(self, student_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_output: (B, D) — all student CLS tokens (all views × batch).
        Returns:
            Scalar KoLeo loss.
        """
        with torch.autocast("cuda", enabled=False):
            z = F.normalize(student_output.float(), p=2, dim=-1, eps=self.eps)

            # Gather across GPUs for full-batch NN search.
            # The local chunk is re-inserted with the original grad-connected
            # tensor because all_gather does not preserve autograd.
            if dist.is_available() and dist.is_initialized():
                ws     = dist.get_world_size()
                rank   = dist.get_rank()
                chunks = [torch.zeros_like(z) for _ in range(ws)]
                dist.all_gather(chunks, z)
                chunks[rank] = z          # restore grad-connected local shard
                z = torch.cat(chunks, dim=0)

            indices   = self._pairwise_nn_indices(z)
            distances = self.pdist(z, z[indices])          # (B_total,)
            return -torch.log(distances + self.eps).mean()


# ---------------------------------------------------------------------------
# Gram loss — Phase 2 hi-res fine-tuning
# ---------------------------------------------------------------------------

def gram_loss(
    student_patch: torch.Tensor,
    teacher_patch: torch.Tensor,
) -> torch.Tensor:
    """
    Encourages patch-token Gram matrices to be consistent across resolutions.

    Used during Phase 2 (384–512 px) hi-res fine-tuning so that the model
    learns resolution-invariant feature statistics rather than over-fitting
    to high-resolution artifacts.

    Args:
        student_patch: (B, N_s, D) — patch tokens from student at hi-res.
        teacher_patch: (B, N_t, D) — patch tokens from teacher (at 256 px).
    Returns:
        Scalar MSE loss between Gram matrices.
    """
    B, N_s, D = student_patch.shape
    _, N_t, _ = teacher_patch.shape

    G_s = torch.bmm(student_patch.transpose(1, 2), student_patch) / N_s  # (B, D, D)
    G_t = torch.bmm(teacher_patch.transpose(1, 2), teacher_patch) / N_t  # (B, D, D)

    return F.mse_loss(G_s, G_t)
