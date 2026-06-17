"""
Smoke test for nifti_dino_axial on 2 NVIDIA A6000 GPUs.

Loads real NIfTI files, runs 10 training steps with DDP, and verifies
shapes, loss finiteness, gradient flow, EMA update, and KoLeo behaviour.

Launch (2 GPUs):
    torchrun --nproc_per_node=2 nifti_dino_axial/smoke_test.py --nifti_dir /data/ct_nifti

Single-GPU debug:
    python nifti_dino_axial/smoke_test.py --nifti_dir /data/ct_nifti
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import logging
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

_ROOT = Path(__file__).parents[1]  # parent of the package dir (e.g. $WORK)
sys.path.insert(0, str(_ROOT))

_FLEXICT_CANDIDATES = [
    Path(__file__).parent / "models_pretrained" / "flexiCT" / "FlexiCT",  # submodule
    _ROOT / "models_pretrained" / "flexiCT" / "FlexiCT",                  # legacy
]


for _p in _FLEXICT_CANDIDATES:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

from flexi_ct.models import flexi_ct_backbone_base  # type: ignore

from nifti_dino_axial.data.dataset import NiftiAxialDataset, axial_dinov_collate
from nifti_dino_axial.masking import AnatomicallyGuidedMasker
from nifti_dino_axial.training.trainer import DINOv3Trainer, ema_update
from nifti_dino_axial.transforms import AxialDINOvTransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smoke-test hyperparameters (intentionally small)
# ---------------------------------------------------------------------------
N_STEPS     = 10
LOCAL_BATCH = 2      # 512² crops: 2 per GPU keeps peak VRAM under ~20 GB on A6000
N_GLOBAL    = 2
N_LOCAL     = 4
GLOBAL_SIZE = 512    # target resolution for Phase 1 training
PATCH_SIZE  = 8

CFG = {
    "phase": 1,
    "model": {"embed_dim": 864, "patch_size": PATCH_SIZE, "depth": 16, "num_heads": 12},
    "training": {
        "total_steps":      N_STEPS,
        "warmup_steps":     2,
        "local_batch_size": LOCAL_BATCH,
        "save_every":       9999,
        "log_every":        1,
    },
    "optim": {
        "base_lr": 2e-4, "wd_start": 0.04, "wd_end": 0.4,
        "ema_start": 0.994, "grad_clip": 3.0,
        "llrd": 0.9, "patch_embed_lr_mult": 0.2,
    },
    "loss": {
        "dino_out_dim": 65536, "head_hidden": 2048, "head_bottleneck": 256,
        "student_temp": 0.1, "teacher_temp": 0.04, "center_momentum": 0.9,
        "koleo_weight": 0.1, "ibot_weight": 1.0, "gram_weight": 0.0,
    },
    "bf16": True,
    "patch_sizes": [8, 16],   # flexi: alternate 8×8 and 16×16 patches each step
}

BACKBONE_KWARGS = dict(
    patch_size=PATCH_SIZE, in_chans=1, n_storage_tokens=4,
    qkv_bias=False, mask_k_bias=True, drop_path_rate=0.2, layerscale_init=1e-5,
)

# FLEXICT_2D_CKPT = str(
#     Path(__file__).parent / "models_pretrained" / "flexiCT" / "2D_final_model.pth"
# )
FLEXICT_2D_CKPT = str("/lus/work/CT3/cad17796/SHARED/New_Adastra_Hackathon/nifti_dino_axial/models_pretrained/flexiCT/2D_final_model.pth")


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_shape(name: str, tensor: torch.Tensor, expected: tuple) -> None:
    assert tensor.shape == torch.Size(expected), \
        f"[FAIL] {name}: expected {expected}, got {tuple(tensor.shape)}"
    logger.info(f"  [OK] shape {name}: {tuple(tensor.shape)}")


def assert_finite(name: str, value: torch.Tensor | float) -> None:
    v = float(value) if isinstance(value, torch.Tensor) else value
    assert math.isfinite(v) and abs(v) < 1e6, \
        f"[FAIL] {name} is not finite or too large: {v}"
    logger.info(f"  [OK] value {name}: {v:.4f}")


def assert_grad_nonzero(name: str, param: torch.nn.Parameter) -> None:
    assert param.grad is not None, f"[FAIL] {name}: gradient is None"
    assert param.grad.abs().sum().item() > 0, f"[FAIL] {name}: gradient is all zeros"
    logger.info(f"  [OK] gradient {name}: norm={param.grad.norm().item():.4e}")


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(rank: int, world_size: int, nifti_dir: Path,
                   pretrained: str | None, max_volumes: int = 5,
                   output_dir: str | None = None) -> None:
    is_main = (rank == 0)
    device  = torch.device(f"cuda:{rank}")

    if is_main:
        logger.info("=" * 60)
        logger.info(f"Smoke test  —  world_size={world_size}, device={device}")
        logger.info(f"NIfTI dir  : {nifti_dir}")
        logger.info(f"Global crop: {GLOBAL_SIZE}×{GLOBAL_SIZE}  |  patch_size={PATCH_SIZE}")
        logger.info(f"Patches per global crop: {GLOBAL_SIZE // PATCH_SIZE}² = "
                    f"{(GLOBAL_SIZE // PATCH_SIZE)**2}")
        logger.info("=" * 60)

    # ── Dataset ──────────────────────────────────────────────────────────────
    transform = AxialDINOvTransform(
        n_global=N_GLOBAL, n_local=N_LOCAL,
        global_size=GLOBAL_SIZE, hflip_prob=0.5, filter_prob=0.5,
    )
    masker = AnatomicallyGuidedMasker(mask_ratio=0.40)
    dataset = NiftiAxialDataset(
        nifti_dir   = nifti_dir,
        transform   = transform,
        masker      = masker,
        patch_size  = PATCH_SIZE,
        index_json  = None,        # rebuild each run — no stale cache
        max_volumes = max_volumes,
    )

    if is_main:
        assert len(dataset) > 0, "[FAIL] dataset is empty — check --nifti_dir"
        logger.info(f"Dataset: {len(dataset):,} valid axial slices")

    sampler    = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size  = LOCAL_BATCH,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = axial_dinov_collate,
    )

    # ── Batch shape checks ────────────────────────────────────────────────────
    if is_main:
        logger.info("\n--- Batch shape checks ---")
    batch     = next(iter(dataloader))
    n_patches = (GLOBAL_SIZE // PATCH_SIZE) ** 2

    assert_shape("global_crops", batch["global_crops"],
                 (N_GLOBAL, LOCAL_BATCH, 1, GLOBAL_SIZE, GLOBAL_SIZE))
    assert_shape("local_crops",  batch["local_crops"],
                 (N_LOCAL,  LOCAL_BATCH, 1, 112, 112))
    assert_shape("global_masks", batch["global_masks"],
                 (N_GLOBAL, LOCAL_BATCH, n_patches))

    actual_ratio = batch["global_masks"].float().mean().item()
    assert 0.20 <= actual_ratio <= 0.65, \
        f"[FAIL] mask ratio out of range: {actual_ratio:.3f}"
    if is_main:
        logger.info(f"  [OK] mask ratio: {actual_ratio:.3f} (target 0.40)")

    gc = batch["global_crops"][0]
    assert gc.dtype == torch.float32
    mean_abs = gc.abs().mean().item()
    assert mean_abs < 5.0, f"[FAIL] global crops look un-normalised: mean_abs={mean_abs:.2f}"
    if is_main:
        logger.info(f"  [OK] global crop normalisation: mean_abs={mean_abs:.3f}")

    # ── Model build + pretrained init ─────────────────────────────────────────
    if is_main:
        logger.info("\n--- Model build ---")

    backbone = flexi_ct_backbone_base(**BACKBONE_KWARGS)
    backbone.init_weights()   # cls_token / storage_tokens / mask_token are torch.empty
    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    if is_main:
        logger.info(f"  [OK] backbone: {n_params:.1f}M parameters")

    ckpt_path = pretrained or (FLEXICT_2D_CKPT if Path(FLEXICT_2D_CKPT).exists() else None)
    # Load on ALL ranks — not just rank 0.  The trainer copies student→teacher
    # via state_dict(), so if rank 1's student is uninitialized (torch.empty),
    # the teacher on GPU 1 inherits garbage NaN weights before any training step.
    if ckpt_path:
        from nifti_dino_axial.train import load_pretrained_weights
        load_pretrained_weights(backbone, ckpt_path)
        if is_main:
            logger.info(f"  [OK] Loaded pretrained weights: {ckpt_path}")

    # ── Training loop (N_STEPS) ───────────────────────────────────────────────
    if is_main:
        logger.info(f"\n--- Training for {N_STEPS} steps ---")

    def _backbone_factory() -> torch.nn.Module:
        m = flexi_ct_backbone_base(**BACKBONE_KWARGS)
        m.init_weights()   # must come before load_pretrained_weights
        if ckpt_path:
            from nifti_dino_axial.train import load_pretrained_weights
            load_pretrained_weights(m, ckpt_path)
        return m

    import tempfile
    _ckpt_tmp = output_dir or tempfile.mkdtemp(prefix="smoke_ckpt_")
    if output_dir:
        Path(_ckpt_tmp).mkdir(parents=True, exist_ok=True)
    if is_main:
        logger.info(f"Smoke output dir: {_ckpt_tmp}")
    trainer = DINOv3Trainer(
        student_backbone = backbone,
        backbone_factory = _backbone_factory,
        cfg              = CFG,
        output_dir       = _ckpt_tmp,
    )

    teacher_param_before = next(trainer.teacher_backbone.parameters()).clone()

    loss_history = []
    loader_iter  = iter(dataloader)

    def _bb(m):
        from torch.nn.parallel import DistributedDataParallel as DDP
        return m.module if isinstance(m, DDP) else m

    for step in range(N_STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            batch = next(loader_iter)

        trainer.optimizer.zero_grad(set_to_none=True)
        loss_dict = trainer._step(batch)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(trainer.student_backbone.parameters()) +
            list(trainer.dino_head.parameters()) +
            list(trainer.ibot_head.parameters()),
            3.0,
        )
        trainer.optimizer.step()

        ema_update(_bb(trainer.student_backbone), trainer.teacher_backbone,  0.994)
        ema_update(_bb(trainer.dino_head),        trainer.teacher_dino_head, 0.994)
        ema_update(_bb(trainer.ibot_head),        trainer.teacher_ibot_head, 0.994)

        loss_val = loss_dict["loss"].item()
        loss_history.append(loss_val)

        if is_main:
            logger.info(
                f"  step {step+1:2d}/{N_STEPS}"
                f"  total={loss_val:.4f}"
                f"  dino={loss_dict['dino_loss'].item():.4f}"
                f"  ibot={loss_dict['ibot_loss'].item():.4f}"
                f"  koleo={loss_dict['koleo_loss'].item():.4f}"
                f"  ps={loss_dict['patch_size']}"
            )

    # ── Post-training checks ──────────────────────────────────────────────────
    if is_main:
        logger.info("\n--- Post-training checks ---")

    for name, val in loss_dict.items():
        assert_finite(name, val)

    _bb_s       = _bb(trainer.student_backbone)
    first_param = next(_bb_s.parameters())
    assert_grad_nonzero("student_backbone.first_param", first_param)

    teacher_param_after = next(trainer.teacher_backbone.parameters())
    diff = (teacher_param_after - teacher_param_before).abs().max().item()
    assert diff > 0, "[FAIL] Teacher weights unchanged — EMA not applied"
    if is_main:
        logger.info(f"  [OK] EMA teacher updated: max_param_diff={diff:.4e}")

    koleo_val = loss_dict["koleo_loss"].item()
    assert math.isfinite(koleo_val), f"[FAIL] KoLeo loss is not finite: {koleo_val}"
    if is_main:
        logger.info(f"  [OK] KoLeo loss: {koleo_val:.4f}")

    max_loss = max(loss_history)
    assert max_loss < 500.0, f"[FAIL] Loss exploded: max={max_loss:.2f}"
    if is_main:
        logger.info(f"  [OK] Loss range over {N_STEPS} steps: "
                    f"[{min(loss_history):.4f}, {max_loss:.4f}]")

    # ── Checkpoint save / load verification ──────────────────────────────────
    if is_main:
        logger.info("\n--- Checkpoint verification ---")

    if is_main:
        ckpt_path_saved = trainer._save_checkpoint("smoke")
        assert ckpt_path_saved.exists(), \
            f"[FAIL] _save_checkpoint returned {ckpt_path_saved} but file not found"
        logger.info(f"  [OK] checkpoint written: {ckpt_path_saved}")

        ckpt = torch.load(str(ckpt_path_saved), map_location="cpu", weights_only=False)

        # Keys
        required_keys = {
            "step", "teacher", "student_backbone",
            "dino_head", "ibot_head",
            "teacher_dino_head", "teacher_ibot_head",
            "optimizer", "cfg",
        }
        missing_keys = required_keys - set(ckpt.keys())
        assert not missing_keys, f"[FAIL] checkpoint missing keys: {missing_keys}"
        logger.info(f"  [OK] checkpoint keys present: {sorted(ckpt.keys())}")

        # Step counter — smoke test uses a manual _step() loop that does NOT
        # increment trainer.step (only trainer.train() does).  We just verify
        # that whatever the trainer's current step is, the checkpoint stores it
        # faithfully.
        assert ckpt["step"] == trainer.step, (
            f"[FAIL] checkpoint step mismatch: saved={ckpt['step']}, "
            f"trainer.step={trainer.step}"
        )
        logger.info(f"  [OK] checkpoint step={ckpt['step']} (== trainer.step)")

        # Weight consistency: spot-check first 3 keys of student_backbone
        live_sd = _bb(trainer.student_backbone).state_dict()
        spot_keys = list(ckpt["student_backbone"].keys())[:3]
        for k in spot_keys:
            saved = ckpt["student_backbone"][k].float()
            live  = live_sd[k].cpu().float()
            assert torch.allclose(saved, live, atol=1e-5), (
                f"[FAIL] student_backbone[{k}]: saved vs live mismatch "
                f"(max diff={(saved - live).abs().max().item():.2e})"
            )
        logger.info(f"  [OK] student_backbone weights consistent with live model "
                    f"(checked: {spot_keys})")

        # JSONL monitoring — write one entry manually to verify the path is
        # writable.  (In production trainer.train() writes automatically;
        # the smoke test drives the loop manually so nothing is auto-written.)
        _test_entry = {"step": trainer.step, "loss": float(loss_dict["loss"].item()),
                       "smoke": True}
        with trainer._metrics_file.open("a") as _f:
            _f.write(json.dumps(_test_entry) + "\n")
        _readback = json.loads(trainer._metrics_file.read_text().strip())
        assert "step" in _readback and "loss" in _readback, \
            f"[FAIL] metrics.jsonl readback missing fields: {_readback}"
        logger.info(f"  [OK] metrics.jsonl writable: {trainer._metrics_file}")
        logger.info(f"       entry={_readback}")

        # TensorBoard
        if trainer.tb_writer is not None:
            trainer.tb_writer.add_scalar("smoke/loss", _test_entry["loss"], trainer.step)
            trainer.tb_writer.flush()
            logger.info(f"  [OK] TensorBoard writer active → {trainer.output_dir / 'tb'}")
        else:
            logger.info(
                "  [--] TensorBoard not available "
                "(run: pip install tensorboard  in $WORK/venv_dino to enable)"
            )

    if world_size > 1:
        dist.barrier()

    if is_main:
        logger.info("\n" + "=" * 60)
        logger.info("  ALL CHECKS PASSED — smoke test successful")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test — DINOv3 NIfTI axial training pipeline"
    )
    parser.add_argument(
        "--nifti_dir", required=True,
        help="Directory containing .nii / .nii.gz files",
    )
    parser.add_argument(
        "--pretrained", default=None,
        help="Optional pretrained checkpoint path (defaults to FlexiCT 2D if present)",
    )
    parser.add_argument(
        "--max_volumes", type=int, default=5,
        help="Number of NIfTI volumes to load (default: 5 — keeps the smoke test fast)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write checkpoints, metrics.jsonl and TensorBoard events. "
             "Defaults to a temporary directory (/tmp) that is deleted after the job. "
             "Pass e.g. $WORK/checkpoints/smoke_test to keep outputs.",
    )
    args = parser.parse_args()

    nifti_dir = Path(args.nifti_dir)
    assert nifti_dir.is_dir(), f"--nifti_dir does not exist: {nifti_dir}"

    # Distributed init
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank = local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    try:
        run_smoke_test(rank, world_size, nifti_dir, args.pretrained,
                       max_volumes=args.max_volumes,
                       output_dir=args.output_dir)
    except Exception:
        logger.error(f"[rank {rank}] Smoke test FAILED:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()