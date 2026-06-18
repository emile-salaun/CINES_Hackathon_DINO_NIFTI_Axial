"""
preprocess_nifti_to_npy.py

Pre-processing script: NIfTI → .npy (float16, RAS-reoriented, HU-clipped).

/!\ Run ONCE before training.  For each NIfTI file found under `nifti_dir`.

Usage
-----
    python preprocess_nifti_to_npy.py \
        --nifti_dir  /data/raw_ct \
        --out_dir    /data/preprocessed_npy \
        --index_json /data/preprocessed_npy/slice_index.json \
        --workers    192
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (mirror NiftiAxialDataset defaults)
# ---------------------------------------------------------------------------
HU_MIN:        float = -1000.0
HU_MAX:        float =  1000.0
BG_THRESHOLD:  float =  -800.0
MIN_BODY_FRAC: float =    0.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> list[dict] | None:
    """
    Process a single NIfTI file.

    Returns a list of valid slice records  [{"path": str, "z": int}, ...]
    or None if the file could not be loaded.
    """
    fpath, out_dir, nifti_root, hu_min, hu_max, bg_threshold, min_body_frac = args

    # ---- compute output path (mirrors input directory structure) -----------
    try:
        rel = fpath.relative_to(nifti_root)
    except ValueError:
        rel = Path(fpath.name)

    # strip .nii.gz or .nii suffix
    stem = rel.with_suffix("") if rel.suffix == ".gz" else rel
    stem = stem.with_suffix("")          # remove the remaining .nii
    out_path = out_dir / stem.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load & reorient --------------------------------------------------
    try:
        img  = nib.load(str(fpath))
        img  = nib.as_closest_canonical(img)          # → RAS
        data = np.array(img.get_fdata(dtype=np.float32))   # (X, Y, Z)
    except Exception as exc:
        logger.warning(f"Cannot load {fpath}: {exc}")
        return None

    # ---- clip + cast ------------------------------------------------------
    data = np.clip(data, hu_min, hu_max).astype(np.float16)  # (H, W, Z)

    # ---- save -------------------------------------------------------------
    np.save(str(out_path), data)

    # ---- build valid slice records ----------------------------------------
    records: list[dict] = []
    n_slices = data.shape[2]
    for z in range(n_slices):
        sl   = data[:, :, z].astype(np.float32)       # float32 for comparison
        frac = float((sl > bg_threshold).mean())
        if frac >= min_body_frac:
            records.append({"path": str(out_path), "z": z})

    return records



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NIfTI CT volumes to RAS-reoriented, HU-clipped .npy files."
    )
    parser.add_argument("--nifti_dir",    required=True,  help="Root dir with .nii/.nii.gz files")
    parser.add_argument("--out_dir",      required=True,  help="Output directory for .npy files")
    parser.add_argument("--index_json",   required=True,  help="Path to write the slice index JSON")
    parser.add_argument("--hu_min",       type=float, default=HU_MIN)
    parser.add_argument("--hu_max",       type=float, default=HU_MAX)
    parser.add_argument("--bg_threshold", type=float, default=BG_THRESHOLD)
    parser.add_argument("--min_body_frac",type=float, default=MIN_BODY_FRAC)
    parser.add_argument("--workers",      type=int,   default=max(1, os.cpu_count() - 2),
                        help="Number of parallel workers (default: nCPU-2)")
    parser.add_argument("--max_volumes",  type=int,   default=None,
                        help="Limit number of volumes (for debugging)")
    args = parser.parse_args()

    nifti_root = Path(args.nifti_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(set(
        list(nifti_root.rglob("*.nii.gz")) +
        list(nifti_root.rglob("*.nii"))
    ))
    if args.max_volumes is not None:
        files = files[: args.max_volumes]
    logger.info(f"Found {len(files):,} NIfTI files under {nifti_root}")

    worker_args = [
        (
            fpath,
            out_dir,
            nifti_root,
            args.hu_min,
            args.hu_max,
            args.bg_threshold,
            args.min_body_frac,
        )
        for fpath in files
    ]

    all_records: list[dict] = []
    failed = 0

    ctx = mp.get_context("spawn")   # safer with nibabel + numpy
    with ctx.Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_one, worker_args),
            total=len(files),
            desc="Converting volumes",
            unit="vol",
        ))

    for res in results:
        if res is None:
            failed += 1
        else:
            all_records.extend(res)

    index_path = Path(args.index_json)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(
        f"Done.  Volumes: {len(files) - failed:,} ok / {failed} failed  |  "
        f"Valid slices: {len(all_records):,}  |  "
        f"Index saved to {index_path}"
    )


if __name__ == "__main__":
    main()