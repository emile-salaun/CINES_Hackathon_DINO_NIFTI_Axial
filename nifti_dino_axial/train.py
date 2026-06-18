"""
Main training entry point — DINOv3 Phase 1 on axial NIfTI CT slices.

Launch (single node, 8 GPUs):
    python -m torch.distributed.run --nproc_per_node=8 nifti_dino_axial/train.py \
        --config nifti_dino_axial/configs/phase1.yaml \
        --output_dir /checkpoints/axial_p1

Single-GPU debug:
    python nifti_dino_axial/train.py \
        --config nifti_dino_axial/configs/phase1.yaml \
        --output_dir /checkpoints/axial_p1_debug

Profiling (rank 0 only, 13 steps then stops):
    python -m torch.distributed.run --nproc_per_node=8 nifti_dino_axial/train.py \
        --config nifti_dino_axial/configs/phase1.yaml \
        --output_dir /checkpoints/axial_p1 \
        --profile
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from torch.profiler import profile, ProfilerActivity

_ROOT = Path(__file__).parents[1]  # parent of the package dir (e.g. $WORK)
sys.path.insert(0, str(_ROOT))

# FlexiCT: submodule path (models_pretrained/flexiCT/FlexiCT inside the repo)
# takes precedence over the legacy sibling-directory layout.
_FLEXICT_CANDIDATES = [
    Path(__file__).parent / "models_pretrained" / "flexiCT" / "FlexiCT",  # submodule
    _ROOT / "models_pretrained" / "flexiCT" / "FlexiCT",                  # legacy
]
for _p in _FLEXICT_CANDIDATES:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

from flexi_ct.models import flexi_ct_backbone_base  # type: ignore

from nifti_dino_axial.data.dataset import (
    NpyAxialDataset,
    axial_dinov_collate,
)
from nifti_dino_axial.masking import AnatomicallyGuidedMasker
from nifti_dino_axial.training.trainer import DINOv3Trainer
from nifti_dino_axial.transforms import AxialDINOvTransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlexiCT-Base backbone kwargs (unchanged from FlexiCT Phase 1)
# ---------------------------------------------------------------------------

_BACKBONE_KWARGS = dict(
    patch_size         = 8,
    in_chans           = 1,
    n_storage_tokens   = 4,
    qkv_bias           = False,
    mask_k_bias        = True,
    drop_path_rate     = 0.2,
    layerscale_init    = 1e-5,
)


def build_backbone() -> torch.nn.Module:
    m = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
    m.init_weights()   # cls_token / storage_tokens / mask_token are torch.empty by default
    return m


def load_pretrained_weights(backbone: torch.nn.Module, ckpt_path: str) -> None:
    """Load DINOv3 / FlexiCT ImageNet pretrained weights into the backbone.

    Supported checkpoint formats (tried in order):
      1. FlexiCT teacher checkpoint  — keys ``backbone.*`` under ``ckpt["teacher"]``
      2. Our own trainer checkpoint  — keys under ``ckpt["student_backbone"]``
      3. Raw state-dict              — top-level keys

    For 3-channel → 1-channel patch-embedding adaptation (e.g. DINOv2 ImageNet
    weights loaded into a 1-channel CT backbone), the three input-channel weights
    are averaged into one.

    Missing / unexpected keys are logged at WARNING level; training continues
    with those parameters randomly initialised.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "teacher" in ckpt:
        # FlexiCT teacher checkpoint: keys are "backbone.<param>"
        sd = {
            k[len("backbone."):]: v
            for k, v in ckpt["teacher"].items()
            if k.startswith("backbone.")
        }
    elif isinstance(ckpt, dict) and "student_backbone" in ckpt:
        sd = ckpt["student_backbone"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # --- 3-channel → 1-channel patch embedding adaptation ---
    # DINOv2/v3 ImageNet weights have proj.weight shape (D, 3, p, p).
    # FlexiCT CT backbone expects (D, 1, p, p).  Average across channels.
    for key in list(sd.keys()):
        if "patch_embed" in key and "proj.weight" in key:
            w = sd[key]
            if w.ndim == 4 and w.shape[1] == 3:
                sd[key] = w.mean(dim=1, keepdim=True)
                logger.info(f"  Averaged 3-ch → 1-ch for {key}")

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Pretrained init: {len(missing)} missing keys "
                       f"(will be randomly initialised): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"Pretrained init: {len(unexpected)} unexpected keys "
                       f"(ignored): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    logger.info(f"Loaded pretrained weights from {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINOv3 Phase 1 pre-training on axial NIfTI CT slices"
    )
    parser.add_argument("--config",       required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--resume",       default=None, help="Checkpoint to resume from")
    parser.add_argument("--pretrained",   default=None,
                        help="DINOv3 / FlexiCT ImageNet pretrained weights to initialise from")
    parser.add_argument("--max_volumes",  default=None, type=int,
                        help="Limit number of NIfTI volumes (useful for quick smoke tests)")
    parser.add_argument("--nifti_dir",   default=None,
                        help="Override data.nifti_dir from config (e.g. $SCRATCH/ct_nifti)")
    # [profiler] flag — activates torch profiler for a short window then stops.
    # Only rank 0 writes traces; other ranks run normally without profiler overhead.
    parser.add_argument("--profile",      action="store_true",
                        help="Run torch profiler (wait=5, warmup=5, active=3) on rank 0")
    parser.add_argument("--profile_with_stack", action="store_true",
                        help="Include Python call stack in profiler trace (higher overhead)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides — must come right after config load so all downstream code
    # (dataset build, trainer) sees the updated values.
    if args.nifti_dir:
        old_index = cfg["data"].get("index_json", None)
        cfg["data"]["nifti_dir"] = args.nifti_dir
        # Also relocate index_json into the new nifti_dir so we don't try to
        # write to the config's placeholder path (e.g. /data/ct_nifti/).
        if old_index:
            cfg["data"]["index_json"] = str(
                Path(args.nifti_dir) / Path(old_index).name
            )

    # ---- Distributed init ----
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    is_main = (int(os.environ.get("RANK", 0)) == 0)
    if is_main:
        logger.info(f"Config     : {args.config}")
        logger.info(f"Output dir : {args.output_dir}")

    # ---- Build transform + masker ----
    tf_cfg     = cfg.get("transforms", {})
    mask_cfg   = cfg.get("masking",    {})

    transform = AxialDINOvTransform(
        n_global    = cfg["data"].get("n_global",         2),
        n_regional  = cfg["data"].get("n_regional", 4),
        n_local     = cfg["data"].get("n_local",          8),
        global_size = cfg["data"].get("global_crop_size", 512),
        hflip_prob  = tf_cfg.get("hflip_prob",  0.5),
        filter_prob = tf_cfg.get("filter_prob", 0.5),
    )

    masker = AnatomicallyGuidedMasker(
        mask_ratio       = mask_cfg.get("mask_ratio",         0.40),
        min_aspect       = mask_cfg.get("min_aspect",         0.3),
        max_aspect       = mask_cfg.get("max_aspect",         3.3),
        max_block_area   = mask_cfg.get("max_block_area",     0.35),
        gaussian_sigma   = mask_cfg.get("gaussian_sigma",     None),
        n_fallback_iters = mask_cfg.get("n_fallback_iters",   50),
    )

    # ---- Dataset ----
    # Build only on rank 0 (scans NIfTI files + writes index_json), then
    # barrier so other ranks load the already-saved index — avoids a
    # multi-process write race on the JSON file.
    data_cfg = cfg["data"]
    index_json = "/lus/work/CT3/cad17796/SHARED/merlin_extracted/slice_index.json"

    if not Path(index_json).exists():
        raise FileNotFoundError(
            f"Slice index not found: {index_json}\n"
            "Run preprocess_nifti_to_npy.py before launching training."
        )

    dataset = NpyAxialDataset(
            index_json    = index_json,
            transform     = transform,
            masker        = masker,
            patch_size    = cfg["model"].get("patch_size",   8),
            bg_threshold  = data_cfg.get("bg_threshold",    -800.0),
            min_body_frac = data_cfg.get("min_body_frac",   0.05),
            cache_size    = data_cfg.get("cache_size",       8),
        )

    # if dist.is_initialized() and index_json and not Path(index_json).exists():
    #     if is_main:
    #         dataset = _build_dataset()   # rank 0 scans + writes the JSON
    #     dist.barrier()                   # all others wait
    #     if not is_main:
    #         dataset = _build_dataset()   # now the JSON exists → fast load
    # else:
    #     dataset = _build_dataset()

    if is_main:
        logger.info(f"Dataset: {len(dataset):,} valid axial slices")

    sampler     = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    num_workers = data_cfg.get("num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size          = cfg["training"]["local_batch_size"],
        sampler             = sampler,
        shuffle             = (sampler is None),
        num_workers         = num_workers,
        pin_memory          = True,
        drop_last           = True,
        collate_fn          = axial_dinov_collate,
        persistent_workers  = (num_workers > 0),
    )

    # ---- Build model ----
    student_backbone = build_backbone()

    # CLI --pretrained takes precedence; fall back to config field.
    pretrained_path = args.pretrained or cfg.get("pretrained_weights")
    if pretrained_path:
        if is_main:
            logger.info(f"Pretrained weights: {pretrained_path}")
        load_pretrained_weights(student_backbone, pretrained_path)

    # ---- Auto-resume: detect latest checkpoint if --resume not given ----
    # Allows the job to be requeued after preemption without manual intervention.
    if args.resume is None:
        output_path = Path(args.output_dir)
        if output_path.exists():
            ckpts = sorted(output_path.glob("ckpt_step*.pt"))
            if ckpts:
                args.resume = str(ckpts[-1])
                if is_main:
                    logger.info(f"Auto-resuming from {args.resume}")

    trainer = DINOv3Trainer(
        student_backbone = student_backbone,
        backbone_factory = build_backbone,
        cfg              = cfg,
        output_dir       = args.output_dir,
        resume_path      = args.resume,
    )

    # ---- Profiler setup ----
    # Active uniquement sur rank 0 pour éviter l'overhead sur tous les GPUs.
    # Schedule : 5 steps ignorés (JIT/compile chauffe) + 5 warmup + 3 actifs = 13 steps.
    # Les traces sont écrites dans output_dir/profiler/ au format TensorBoard.
    if args.profile and is_main:
        profile_dir = str(Path(args.output_dir) / "profiler")
        prof = profile(
            activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule        = torch.profiler.schedule(wait=5, warmup=5, active=3),
            on_trace_ready  = torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes   = True,
            with_stack      = args.profile_with_stack,
        )
        prof.start()
        logger.info(f"Profiler started — traces → {profile_dir}")
    else:
        prof = None

    trainer.train(dataloader, profiler=prof)

    if prof is not None:
        prof.stop()
        logger.info("Profiler stopped.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()