"""
DINOv3 teacher-student trainer for NIfTI axial slice pre-training.

Identical training dynamics to merlin_flexi_ct/training/trainer.py
(FlexiCT Phase 1 hyperparameters).  The only differences are:
  - Import paths point to nifti_dino_axial.training
  - No Phase 2 Gram-loss branch (kept as optional via gram_weight > 0)
  - Checkpoint includes 'masker_cfg' field for reproducibility

Losses: DINO (CLS) + iBOT (masked patches) + SigReg (reg, weight=0.1)
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

from .heads import DINOHead
from .losses import DINOLoss, iBOTLoss, gram_loss, KoLeoLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA & schedules (unchanged from merlin_flexi_ct)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for s_p, t_p in zip(student.parameters(), teacher.parameters()):
        t_p.data.mul_(momentum).add_((1.0 - momentum) * s_p.data)


def cosine_schedule(start: float, end: float, step: int, total: int) -> float:
    frac = step / max(total, 1)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * frac))


def get_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    return cosine_schedule(base_lr, 0.0, step - warmup, total - warmup)


def get_wd(step: int, total: int, wd_start: float, wd_end: float) -> float:
    return cosine_schedule(wd_start, wd_end, step, total)


def get_ema_momentum(step: int, total: int, start: float = 0.994) -> float:
    return cosine_schedule(start, 1.0, step, total)


# ---------------------------------------------------------------------------
# Flash-attention patch
# ---------------------------------------------------------------------------

# Note: FlexiCT's SelfAttention.compute_attention already uses
# F.scaled_dot_product_attention natively (attention.py:116).
# No attention patch needed.


# ---------------------------------------------------------------------------
# Layer-wise LR decay (LLRD)
# ---------------------------------------------------------------------------

def _build_llrd_param_groups(
    backbone:             nn.Module,
    base_lr:              float,
    llrd:                 float,
    patch_embed_lr_mult:  float,
    depth:                int,
) -> list[dict]:
    """
    Build AdamW parameter groups with layer-wise LR decay.

    FlexiCT paper: decay factor 0.9 per block from top, patch-embed lr × 0.2.

    Layers are numbered from the *output* end:
      - norm / cls_token / storage_tokens → scale 1.0  (top)
      - blocks[depth-1]                  → scale llrd^1
      - blocks[depth-2]                  → scale llrd^2
      - ...
      - blocks[0]                        → scale llrd^depth
      - patch_embed                      → base_lr × patch_embed_lr_mult
    """
    groups: dict[str, dict] = {}

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("patch_embed"):
            lr = base_lr * patch_embed_lr_mult
            key = "patch_embed"
        elif name.startswith("blocks."):
            # e.g. "blocks.11.attn.qkv.weight"
            block_idx = int(name.split(".")[1])
            # distance from top: depth-1 is closest to output
            dist_from_top = (depth - 1) - block_idx
            lr = base_lr * (llrd ** dist_from_top)
            key = f"block_{block_idx}"
        else:
            # cls_token, storage_tokens, norm, rope embeds, mask_token
            lr = base_lr
            key = "head_norm"

        if key not in groups:
            groups[key] = {"params": [], "lr": lr}
        groups[key]["params"].append(param)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Flexi patch-size helpers
# ---------------------------------------------------------------------------

def _adapt_masks(
    masks:   List[torch.Tensor],
    base_ps: int,
    target_ps: int,
) -> List[torch.Tensor]:
    """
    Resize boolean iBOT masks from the base-patch-size grid to a coarser grid.

    The dataset always generates masks at ``base_ps`` (e.g. 8).  When a flexi
    step samples a larger patch size (e.g. 16), each target patch covers
    ``(target_ps / base_ps)²`` base patches.  We use **max-pool** so that a
    target patch is masked if *any* of its constituent base patches were masked
    — this preserves the anatomically guided masking intent.

    Args:
        masks:     list[Tensor(B, N_base)]  bool  (one per global crop)
        base_ps:   patch size the masks were generated at (e.g. 8)
        target_ps: desired patch size (e.g. 16); must be an integer multiple
                   of base_ps

    Returns:
        list[Tensor(B, N_target)]  bool
    """
    if target_ps == base_ps:
        return masks
    assert target_ps % base_ps == 0, \
        f"target_ps ({target_ps}) must be a multiple of base_ps ({base_ps})"
    factor = target_ps // base_ps
    result = []
    for m in masks:
        B, N = m.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"mask is not square: {N} patches (H={H}, W={W})"
        # (B, N) → (B, 1, H, W) → max-pool → (B, 1, H/f, W/f) → (B, N_target)
        m2d  = m.float().view(B, 1, H, W)
        down = F.max_pool2d(m2d, kernel_size=factor, stride=factor)
        result.append(down.view(B, -1).bool())
    return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DINOv3Trainer:
    """
    Self-contained DINOv3 trainer.  Call trainer.train(dataloader) to start.

    Args:
        student_backbone : FlexiCT or DINOv2-compatible ViT backbone.
        backbone_factory : Zero-argument callable that returns a fresh backbone
                           instance.  Used to build the teacher without deepcopy
                           (deepcopy fails when nn.utils.weight_norm is applied).
        cfg              : Config dict loaded from configs/phase1.yaml.
        output_dir       : Where to save checkpoints.
        resume_path      : Optional checkpoint path to resume from.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        backbone_factory:  Callable[[], nn.Module],
        cfg:              dict,
        output_dir:       str,
        resume_path:      Optional[str] = None,
    ):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rank       = int(os.environ.get("RANK",       0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main    = (self.rank == 0)
        self.device     = torch.device(f"cuda:{self.local_rank}")

        # ---- Monitoring setup (rank 0 only) ----
        self._metrics_file = self.output_dir / "metrics.jsonl"
        if self.is_main:
            if _HAS_TB:
                self.tb_writer = _TBWriter(str(self.output_dir / "tb"))
                logger.info(f"[rank 0] TensorBoard writer → {self.output_dir / 'tb'}")
            else:
                self.tb_writer = None
                logger.info(
                    "[rank 0] TensorBoard not available "
                    "(pip install tensorboard to enable)"
                )
        else:
            self.tb_writer = None

        # ---- Student ----
        # self.student_backbone = student_backbone.to(self.device)
        self.student_backbone = student_backbone.to(self.device, dtype=torch.bfloat16)
        self.teacher_backbone = backbone_factory().to(self.device, dtype=torch.bfloat16)
        embed_dim       = cfg["model"]["embed_dim"]
        dino_out_dim    = cfg["loss"]["dino_out_dim"]
        head_hidden     = cfg["loss"].get("head_hidden",     2048)
        head_bottleneck = cfg["loss"].get("head_bottleneck",  256)

        self.dino_head = DINOHead(embed_dim, dino_out_dim,
                                  hidden_dim=head_hidden,
                                  bottleneck_dim=head_bottleneck).to(self.device , dtype=torch.bfloat16)
        self.ibot_head = DINOHead(embed_dim, dino_out_dim,
                                  hidden_dim=head_hidden,
                                  bottleneck_dim=head_bottleneck).to(self.device, dtype=torch.bfloat16)

        # self.dino_head = DINOHead(...).to(self.device)
        # self.ibot_head = DINOHead(...).to(self.device, dtype=torch.bfloat16)

        # ---- Teacher (EMA, no grad) ----
        # deepcopy fails when nn.utils.weight_norm is applied to any layer
        # (PyTorch bug).  Build fresh instances via the factory and copy
        # weights through state_dict to avoid the issue.
        self.teacher_backbone = backbone_factory().to(self.device)
        self.teacher_backbone.load_state_dict(student_backbone.state_dict())
        self.teacher_backbone.eval()   # no dropout / drop-path during EMA inference

        # self.teacher_dino_head = DINOHead(...).to(self.device, dtype=torch.bfloat16)
        # self.teacher_ibot_head = DINOHead(...).to(self.device, dtype=torch.bfloat16)

        self.teacher_dino_head = DINOHead(embed_dim, dino_out_dim,
                                          hidden_dim=head_hidden,
                                          bottleneck_dim=head_bottleneck).to(self.device, dtype=torch.bfloat16)
        self.teacher_dino_head.load_state_dict(self.dino_head.state_dict())
        self.teacher_ibot_head = DINOHead(embed_dim, dino_out_dim,
                                          hidden_dim=head_hidden,
                                          bottleneck_dim=head_bottleneck).to(self.device, dtype=torch.bfloat16)
        self.teacher_ibot_head.load_state_dict(self.ibot_head.state_dict())
        for p in (*self.teacher_backbone.parameters(),
                  *self.teacher_dino_head.parameters(),
                  *self.teacher_ibot_head.parameters()):
            p.requires_grad_(False)

        # Defensive: zero-out any params still NaN/Inf after state_dict loading,
        # AND clamp very-large-but-finite values that slip through isfinite().
        # FlexiCT allocates mask_token / cls_token / storage_tokens with torch.empty
        # (un-initialised GPU memory).  Values up to float32-max (~3.4e38) are
        # technically finite but overflow BF16 attention (Q·Kᵀ ≈ 1e74 → Inf → NaN).
        # The real fix is backbone.init_weights() before entering the trainer;
        # this clamp is a belt-and-suspenders safety net.
        _SAFE_MAX = 1e4   # larger than any reasonable weight, smaller than overflow
        for _mname, _module in [("student_backbone", self.student_backbone),
                                 ("teacher_backbone", self.teacher_backbone)]:
            for _pname, _p in _module.named_parameters():
                _bad = ~torch.isfinite(_p.data)
                if _bad.any():
                    n_bad = _bad.sum().item()
                    logger.warning(
                        f"[rank {self.rank}] {_mname}.{_pname}: "
                        f"{n_bad} NaN/Inf values after init "
                        f"(torch.empty not in checkpoint) — zeroing"
                    )
                    _p.data[_bad] = 0.0
                _large = _p.data.abs() > _SAFE_MAX
                if _large.any():
                    n_large = _large.sum().item()
                    logger.warning(
                        f"[rank {self.rank}] {_mname}.{_pname}: "
                        f"{n_large} very-large values (|x|>{_SAFE_MAX}) "
                        f"(torch.empty garbage) — zeroing"
                    )
                    _p.data[_large] = 0.0

        # Capture raw (non-DDP) student backbone for LLRD param-group building.
        # Must happen BEFORE the DDP wrap so named_parameters() returns bare
        # names ("blocks.11.attn.qkv.weight") not DDP-prefixed ones
        # ("module.blocks.11.attn.qkv.weight"), which would silently bypass
        # all LLRD matching and fall back to base_lr for every parameter.
        bb_raw = self.student_backbone

        # DDP wrap
        if self.world_size > 1:
            self.student_backbone = DDP(self.student_backbone, device_ids=[self.local_rank])
            self.dino_head        = DDP(self.dino_head,        device_ids=[self.local_rank])
            self.ibot_head        = DDP(self.ibot_head,        device_ids=[self.local_rank])

        # ---- Losses ----
        loss_cfg = cfg["loss"]
        self.dino_loss = DINOLoss(
            dino_out_dim,
            student_temp    = loss_cfg.get("student_temp",    0.1),
            teacher_temp    = loss_cfg.get("teacher_temp",    0.04),
            center_momentum = loss_cfg.get("center_momentum", 0.9),
        ).to(self.device,  dtype=torch.bfloat16)
        self.ibot_loss = iBOTLoss(
            dino_out_dim,
            student_temp    = loss_cfg.get("student_temp",    0.1),
            teacher_temp    = loss_cfg.get("teacher_temp",    0.04),
            center_momentum = loss_cfg.get("center_momentum", 0.9),
        ).to(self.device,  dtype=torch.bfloat16)

        self.koleo = KoLeoLoss().to(self.device, dtype=torch.bfloat16)

        # ---- Optimiser (AdamW + layer-wise LR decay, FlexiCT paper §3) ----
        optim_cfg = cfg["optim"]
        llrd      = optim_cfg.get("llrd", None)  # e.g. 0.9; None = no LLRD

        # bb_raw captured above (before DDP wrap) — see comment there.
        if llrd is not None:
            backbone_param_groups = _build_llrd_param_groups(
                bb_raw,
                base_lr        = optim_cfg["base_lr"],
                llrd           = llrd,
                patch_embed_lr_mult = optim_cfg.get("patch_embed_lr_mult", 0.2),
                depth          = cfg["model"].get("depth", 16),
            )
        else:
            backbone_param_groups = [{"params": list(bb_raw.parameters()),
                                      "lr": optim_cfg["base_lr"]}]

        head_params = (list(self.dino_head.parameters()) +
                       list(self.ibot_head.parameters()))
        self.optimizer = torch.optim.AdamW(
            backbone_param_groups + [{"params": head_params,
                                      "lr": optim_cfg["base_lr"]}],
            lr           = optim_cfg["base_lr"],
            weight_decay = optim_cfg.get("wd_start", 0.04),
            betas        = (0.9, 0.999),
        )

        # Store the per-group LR ratios so the cosine schedule can rescale them
        # uniformly while preserving LLRD proportions.
        self._pg_lr_ratios = [
            pg["lr"] / optim_cfg["base_lr"]
            for pg in self.optimizer.param_groups
        ]

        self.use_bf16          = cfg.get("bf16", False)
        self.grad_ckpt         = cfg.get("grad_ckpt", False)
        self.total_steps       = cfg["training"]["total_steps"]
        self.warmup_steps      = cfg["training"]["warmup_steps"]
        self.save_every        = cfg["training"].get("save_every",        5000)
        self.log_every         = cfg["training"].get("log_every",           50)
        # Keep only the last N full checkpoints (each ~3 GB) to avoid filling $WORK.
        # Model-only checkpoints (BF16 teacher backbone, ~300 MB each) are kept all.
        self.keep_checkpoints  = cfg["training"].get("keep_checkpoints",     3)
        self.grad_clip      = cfg["optim"].get("grad_clip",       3.0)
        self.koleo_weight   = loss_cfg.get("koleo_weight",        0.1)
        self.ibot_weight    = loss_cfg.get("ibot_weight",         1.0)
        self.gram_weight    = loss_cfg.get("gram_weight",         0.0)
        self.step           = 0

        # Flexi patch sizes: sample one per training step.
        # The dataset generates masks at base_patch_size; _adapt_masks() downsizes
        # them on-the-fly when a larger patch size is sampled.
        self.base_patch_size = cfg["model"]["patch_size"]
        _ps_cfg = cfg.get("patch_sizes", None)
        self.patch_sizes: list[int] = sorted(_ps_cfg) if _ps_cfg else [self.base_patch_size]
        if len(self.patch_sizes) > 1:
            logger.info(
                f"[rank {self.rank}] Flexi patch sizes: {self.patch_sizes}  "
                f"(base={self.base_patch_size})"
            )

        if resume_path:
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Flexi helpers
    # ------------------------------------------------------------------

    def _set_patch_size(self, patch_size: int) -> None:
        """
        Set the runtime patch size on both student and teacher PatchEmbedND layers.

        PatchEmbedND.set_patch_size() stores the value in _runtime_patch_size;
        on the next forward() it resamples the conv kernel via pseudo-inverse
        bicubic interpolation (see FlexiCT models.py _resample_conv_weight).
        """
        for backbone in (self.student_backbone, self.teacher_backbone):
            raw = backbone.module if isinstance(backbone, DDP) else backbone
            for attr in ("patch_embed_2D", "patch_embed_3D"):
                embed = getattr(raw, attr, None)
                if embed is not None and hasattr(embed, "set_patch_size"):
                    embed.set_patch_size(patch_size)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _run_backbone(
        self,
        backbone: nn.Module,
        crops:    list[torch.Tensor],
        masks:    list[torch.Tensor | None],
    ) -> list[dict]:
        raw = backbone.module if isinstance(backbone, DDP) else backbone

        if not self.grad_ckpt or not torch.is_grad_enabled():
            return raw.forward_features_list(crops, masks)

        # Gradient checkpointing: recompute block activations during backward
        # instead of storing them.  Saves ~70 % activation memory at 512² with
        # only a ~20 % compute overhead.
        #
        # All blocks share the same class (SelfAttentionBlock), so we capture
        # the original forward ONCE before patching and restore it ONCE after.
        # Patching per-block in a loop causes the captured orig_call to chain
        # through previously-patched versions, leading to 16-deep nested
        # grad_checkpoints and 16× backward overhead.

        block_cls = raw.blocks[0].__class__
        orig_block_forward = block_cls.forward

        def _checkpointed_forward(self_blk, x_list, rope_list):
            def _run(*flat):
                return orig_block_forward(self_blk, list(flat), rope_list)
            return grad_checkpoint(_run, *x_list, use_reentrant=False)

        block_cls.forward = _checkpointed_forward
        try:
            out = raw.forward_features_list(crops, masks)
        finally:
            block_cls.forward = orig_block_forward

        return out

    def _step(self, batch: dict) -> dict[str, torch.Tensor]:
        n_global   = batch["global_crops"].shape[0]
        n_regional = batch["regional_crops"].shape[0] if "regional_crops" in batch else 0
        n_local    = batch["local_crops"].shape[0]

        global_crops   = [batch["global_crops"][i].to(self.device)   for i in range(n_global)]
        regional_crops = [batch["regional_crops"][i].to(self.device) for i in range(n_regional)]
        local_crops    = [batch["local_crops"][i].to(self.device)    for i in range(n_local)]
        global_masks   = [batch["global_masks"][i].to(self.device)   for i in range(n_global)]

        # ── Flexi: sample a patch size for this step ──────────────────────────
        # The dataset always generates masks at base_patch_size.
        # When a larger patch size is sampled:
        #   • set_patch_size() resamples the PatchEmbedND conv kernel on-the-fly
        #     via pseudo-inverse bicubic interpolation (FlexiCT _resample_conv_weight)
        #   • _adapt_masks() downsizes the masks by max-pooling to match the
        #     coarser patch grid (e.g. 8→16: 4096 patches → 1024 patches)
        #
        # IMPORTANT: _set_patch_size() is called on EVERY step, including when
        # step_ps == base_patch_size.  PatchEmbedND stores the runtime size in
        # _runtime_patch_size; if we skipped the reset, the previous step's size
        # would linger and produce a token-count mismatch with the un-adapted masks.
        step_ps = random.choice(self.patch_sizes)
        self._set_patch_size(step_ps)
        if step_ps != self.base_patch_size:
            global_masks = _adapt_masks(global_masks, self.base_patch_size, step_ps)

        # Use autocast for mixed-precision instead of manually casting inputs,
        # so that model weights and inputs always agree on dtype.
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16)

        # Teacher — global crops only, no masking
        # Memory strategy: run everything in BF16 end-to-end.
        # At 256² / patch_size=8: n_patches=1024, out_dim=65536.
        # BF16 iBOT head output: batch × 1024 × 65536 × 2 bytes ≈ 6.7 GB at batch=50.
        # Keeping BF16 through the loss halves that vs float32.
        # The DINO head (CLS tokens only) stays in float32 — output is tiny (batch × 65536).
        with torch.no_grad(), amp_ctx:
            t_out   = self._run_backbone(self.teacher_backbone, global_crops, [None] * n_global)
            # t_cls   = [o["x_norm_clstoken"].float() for o in t_out]  # float32 — tiny

            t_cls   = [o["x_norm_clstoken"] for o in t_out]

            t_patch = [o["x_norm_patchtokens"]       for o in t_out]  # BF16 — large
        with torch.no_grad(), amp_ctx:
            t_dino = [self.teacher_dino_head(c) for c in t_cls]
        with torch.no_grad(), amp_ctx:                                 # BF16 iBOT
            t_ibot  = [self.teacher_ibot_head(p) for p in t_patch]

        # Student — global (avec masques iBOT) + locaux (sans masques)
        all_crops = global_crops + regional_crops + local_crops
        all_masks = list(global_masks) + [None] * (n_regional + n_local)
        with amp_ctx:
            s_out   = self._run_backbone(self.student_backbone, all_crops, all_masks)
            # s_cls   = [o["x_norm_clstoken"].float() for o in s_out]           # float32 — tiny
            s_cls   = [o["x_norm_clstoken"] for o in s_out]
            s_patch = [o["x_norm_patchtokens"]       for o in s_out[:n_global]]  # BF16 — large
        # DINO head in float32 (CLS tokens — no memory issue)
        with amp_ctx:
            s_dino = [self.dino_head(c) for c in s_cls]
        # iBOT head in BF16 — gradients flow through BF16 autocast correctly
        with amp_ctx:
            s_ibot  = [self.ibot_head(p) for p in s_patch]

        # Losses
        dino_l = self.dino_loss(t_dino, s_dino)
        ibot_l = self.ibot_loss(t_ibot, s_ibot, global_masks)

        koleo_l = self.koleo(torch.cat(s_cls, dim=0))

        total = dino_l + self.ibot_weight * ibot_l + self.koleo_weight * koleo_l

        gram_l = torch.tensor(0.0, device=self.device)
        if self.gram_weight > 0.0:
            for sp, tp in zip(s_patch, t_patch):
                gram_l = gram_l + gram_loss(sp, tp)
            gram_l = gram_l / n_global
            total  = total + self.gram_weight * gram_l

        return {
            "loss":       total,
            "dino_loss":  dino_l.detach(),
            "ibot_loss":  ibot_l.detach(),
            "koleo_loss": koleo_l.detach(),
            "gram_loss":  gram_l.detach(),
            "patch_size": step_ps,          # logged; not a tensor
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader: DataLoader) -> None:
        cfg       = self.cfg
        base_lr   = cfg["optim"]["base_lr"]
        wd_start  = cfg["optim"].get("wd_start",  0.04)
        wd_end    = cfg["optim"].get("wd_end",     0.40)
        ema_start = cfg["optim"].get("ema_start",  0.994)

        loader_iter  = iter(dataloader)
        _t_last      = time.time()   # wall-clock time at the start of the window
        _step_last   = self.step     # step counter at the start of the window

        while self.step < self.total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(self.step // max(len(dataloader), 1))
                loader_iter = iter(dataloader)
                batch = next(loader_iter)

            lr  = get_lr(self.step, self.total_steps, base_lr, self.warmup_steps)
            wd  = get_wd(self.step, self.total_steps, wd_start, wd_end)
            ema = get_ema_momentum(self.step, self.total_steps, ema_start)

            for pg, ratio in zip(self.optimizer.param_groups, self._pg_lr_ratios):
                pg["lr"] = lr * ratio
                pg["weight_decay"] = wd

            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self._step(batch)

            loss_dict["loss"].backward()

            nn.utils.clip_grad_norm_(
                list(self.student_backbone.parameters()) +
                list(self.dino_head.parameters()) +
                list(self.ibot_head.parameters()),
                self.grad_clip,
            )
            self.optimizer.step()

            # EMA teacher update
            def _bb(m: nn.Module) -> nn.Module:
                return m.module if isinstance(m, DDP) else m

            ema_update(_bb(self.student_backbone), self.teacher_backbone,  ema)
            ema_update(_bb(self.dino_head),        self.teacher_dino_head, ema)
            ema_update(_bb(self.ibot_head),        self.teacher_ibot_head, ema)

            self.step += 1

            if self.is_main and self.step % self.log_every == 0:
                _t_now       = time.time()
                _elapsed     = _t_now - _t_last
                _steps_done  = self.step - _step_last
                _it_s        = _steps_done / max(_elapsed, 1e-6)

                _eta_target_samples = self.cfg.get("eta_target_samples", 10_000_000)
                _samples_per_step   = self.cfg["training"]["local_batch_size"] * self.world_size
                _samples_done       = self.step * _samples_per_step
                _img_s              = _it_s * _samples_per_step
                _samples_remaining  = max(0, _eta_target_samples - _samples_done)
                _eta_s              = _samples_remaining / max(_img_s, 1e-6)
                _eta_h, _rem        = divmod(int(_eta_s), 3600)
                _eta_m              = _rem // 60
                _t_last             = _t_now
                _step_last          = self.step

                logger.info(
                    f"step={self.step:>7d}/{self.total_steps}"
                    f"  loss={loss_dict['loss'].item():.4f}"
                    f"  dino={loss_dict['dino_loss'].item():.4f}"
                    f"  ibot={loss_dict['ibot_loss'].item():.4f}"
                    f"  koleo={loss_dict['koleo_loss'].item():.4f}"
                    f"  gram={loss_dict['gram_loss'].item():.4f}"
                    f"  ps={loss_dict['patch_size']}"
                    f"  lr={lr:.2e}  ema={ema:.5f}"
                    f"  it/s={_it_s:.2f}  img/s={_img_s:.1f}"
                    f"  eta_10M={_eta_h}h{_eta_m:02d}m"
                )
                # ── JSONL + TensorBoard ────────────────────────────────────
                _m = {
                    "step":  self.step,
                    "loss":  round(loss_dict["loss"].item(),       6),
                    "dino":  round(loss_dict["dino_loss"].item(),  6),
                    "ibot":  round(loss_dict["ibot_loss"].item(),  6),
                    "koleo": round(loss_dict["koleo_loss"].item(), 6),
                    "gram":  round(loss_dict["gram_loss"].item(),  6),
                    "lr":    round(lr,  8),
                    "ema":   round(ema, 6),
                    "ps":    loss_dict["patch_size"],
                    "it_s":  round(_it_s, 4),
                }
                with self._metrics_file.open("a") as _f:
                    _f.write(json.dumps(_m) + "\n")
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("loss/total", _m["loss"],        self.step)
                    self.tb_writer.add_scalar("loss/dino",  _m["dino"],        self.step)
                    self.tb_writer.add_scalar("loss/ibot",  _m["ibot"],        self.step)
                    self.tb_writer.add_scalar("loss/koleo", _m["koleo"],       self.step)
                    self.tb_writer.add_scalar("optim/lr",   _m["lr"],          self.step)
                    self.tb_writer.add_scalar("optim/ema",  _m["ema"],         self.step)
                    self.tb_writer.add_scalar("optim/ps",   float(_m["ps"]),   self.step)
                    self.tb_writer.add_scalar("perf/it_s",  _m["it_s"],        self.step)

            if self.is_main and self.step % self.save_every == 0:
                self._save_checkpoint()

        if self.is_main:
            self._save_checkpoint(tag="final")
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Checkpointing (compatible with FlexiCT checkpoint format)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str | None = None) -> Path:
        tag  = tag or f"step{self.step:07d}"
        path = self.output_dir / f"ckpt_{tag}.pt"

        def _bb(m: nn.Module) -> nn.Module:
            return m.module if isinstance(m, DDP) else m

        # ── Full checkpoint (for resume) ──────────────────────────────────────
        torch.save(
            {
                "step": self.step,
                "teacher": {
                    f"backbone.{k}": v
                    for k, v in self.teacher_backbone.state_dict().items()
                },
                "student_backbone":  _bb(self.student_backbone).state_dict(),
                "dino_head":         _bb(self.dino_head).state_dict(),
                "ibot_head":         _bb(self.ibot_head).state_dict(),
                "teacher_dino_head": self.teacher_dino_head.state_dict(),
                "teacher_ibot_head": self.teacher_ibot_head.state_dict(),
                "optimizer":         self.optimizer.state_dict(),
                "dino_loss_center":  self.dino_loss.center,
                "ibot_loss_center":  self.ibot_loss.center,
                "cfg":               self.cfg,
            },
            path,
        )
        gb = path.stat().st_size / 1e9
        logger.info(f"Checkpoint saved : {path}  ({gb:.2f} GB)")

        # ── Lightweight model-only checkpoint (for evaluation / downstream) ───
        # Teacher backbone only, cast to BF16 — compatible with the FlexiCT
        # checkpoint format (keys "backbone.*" under "teacher").
        # ~144 M params × 2 bytes ≈ 290 MB vs ~3 GB for the full checkpoint.
        model_path = self.output_dir / f"model_{tag}.pt"
        torch.save(
            {
                "step": self.step,
                "teacher": {
                    f"backbone.{k}": v.bfloat16()
                    for k, v in self.teacher_backbone.state_dict().items()
                },
                "cfg": self.cfg,
            },
            model_path,
        )
        gb_m = model_path.stat().st_size / 1e9
        logger.info(f"Model checkpoint : {model_path}  ({gb_m:.2f} GB)")

        # ── Prune old full checkpoints ────────────────────────────────────────
        # Only ckpt_step*.pt files are pruned; ckpt_final.pt, ckpt_smoke.pt and
        # model_*.pt files are never deleted automatically.
        if self.keep_checkpoints > 0:
            step_ckpts = sorted(self.output_dir.glob("ckpt_step*.pt"))
            for old in step_ckpts[: -self.keep_checkpoints]:
                old.unlink()
                logger.info(f"Removed old checkpoint: {old}")

        return path

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.step = ckpt.get("step", 0)

        def _bb(m: nn.Module) -> nn.Module:
            return m.module if isinstance(m, DDP) else m

        if "student_backbone" in ckpt:
            _bb(self.student_backbone).load_state_dict(ckpt["student_backbone"])
        elif "teacher" in ckpt:
            sd = {k[len("backbone."):]: v for k, v in ckpt["teacher"].items()
                  if k.startswith("backbone.")}
            _bb(self.student_backbone).load_state_dict(sd, strict=False)

        if "teacher" in ckpt:
            self.teacher_backbone.load_state_dict(
                {k[len("backbone."):]: v for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")},
                strict=False,
            )

        # Restore projection heads — required for correct loss centering and
        # EMA teacher state after a preemption/resume.
        if "dino_head"         in ckpt: _bb(self.dino_head).load_state_dict(ckpt["dino_head"])
        if "ibot_head"         in ckpt: _bb(self.ibot_head).load_state_dict(ckpt["ibot_head"])
        if "teacher_dino_head" in ckpt: self.teacher_dino_head.load_state_dict(ckpt["teacher_dino_head"])
        if "teacher_ibot_head" in ckpt: self.teacher_ibot_head.load_state_dict(ckpt["teacher_ibot_head"])

        if "optimizer"         in ckpt: self.optimizer.load_state_dict(ckpt["optimizer"])
        if "dino_loss_center"  in ckpt: self.dino_loss.center.copy_(ckpt["dino_loss_center"])
        if "ibot_loss_center"  in ckpt: self.ibot_loss.center.copy_(ckpt["ibot_loss_center"])

        logger.info(f"Resumed from {path} at step {self.step}")