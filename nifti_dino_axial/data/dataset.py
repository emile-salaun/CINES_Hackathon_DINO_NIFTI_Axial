"""
Dataset loading axial slices directly from NIfTI files.

This dataset reads raw NIfTI volumes (.nii / .nii.gz), reorients them to canonical RAS,
clips HU values, and extracts axial slices on the fly.

Index building
--------------
To avoid scanning every NIfTI at startup in distributed training, the dataset
can save/load a JSON slice index.  If `index_json` is provided and the file
exists, it is loaded directly.  Otherwise the index is built by iterating
over all NIfTI files and saved to `index_json` (only on main process).

Volume caching
--------------
Each DataLoader worker maintains its own in-process LRU volume cache
(default size = 4 volumes).  This amortises the I/O cost when multiple
slices from the same volume appear within the same worker's shard.
"""
from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from ..transforms import AxialDINOvTransform
from ..masking import AnatomicallyGuidedMasker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Volume loader with simple LRU cache (per-worker, no cross-process sharing)
# ---------------------------------------------------------------------------

class _LRUVolumeCache:
    def __init__(self, maxsize: int = 4):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        img  = nib.load(path)
        img  = nib.as_closest_canonical(img)   # reorient to RAS
        data = np.array(img.get_fdata(dtype=np.float32))
        # data shape after canonical: (X, Y, Z) ≡ (L-R, P-A, I-S)
        # axial plane = first two dimensions, Z is the slice axis

        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[path] = data
        return data


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def axial_dinov_collate(batch: list[dict]) -> dict:
    """
    Collate multi-crop dicts into batched tensors.

    Returns:
        global_crops   : (n_global,   B, 1, 512, 512)
        regional_crops : (n_regional, B, 1, 224, 224)  — omitted if n_regional == 0
        local_crops    : (n_local,    B, 1, 112, 112)
        global_masks   : (n_global,   B, N_patches) bool
    """
    n_global   = len(batch[0]["global_crops"])
    n_regional = len(batch[0].get("regional_crops", []))
    n_local    = len(batch[0]["local_crops"])

    global_crops = torch.stack(
        [torch.stack([b["global_crops"][i] for b in batch]) for i in range(n_global)]
    )
    local_crops = torch.stack(
        [torch.stack([b["local_crops"][i] for b in batch]) for i in range(n_local)]
    )
    global_masks = torch.stack(
        [torch.stack([b["global_masks"][i] for b in batch]) for i in range(n_global)]
    )
    out = {
        "global_crops": global_crops,
        "local_crops":  local_crops,
        "global_masks": global_masks,
    }
    if n_regional > 0:
        out["regional_crops"] = torch.stack(
            [torch.stack([b["regional_crops"][i] for b in batch]) for i in range(n_regional)]
        )
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NiftiAxialDataset(Dataset):
    """
    Loads axial CT slices on the fly from NIfTI files.

    Args:
        nifti_dir:      Root directory containing .nii or .nii.gz files
                        (searched recursively).
        transform:      AxialDINOvTransform instance.
        masker:         AnatomicallyGuidedMasker instance.
        patch_size:     ViT patch size (default 8), used by the masker.
        hu_clip:        (min, max) HU clipping range applied after loading.
        min_body_frac:  Slices with fewer non-background pixels than this
                        fraction are skipped during index building.
        bg_threshold:   HU below this is counted as background.
        index_json:     Optional path to save / reload the slice index
                        (avoids re-scanning on subsequent runs).
        cache_size:     Number of volumes to keep in per-worker LRU cache.
    """

    def __init__(
        self,
        nifti_dir:     str | Path,
        transform:     Optional[AxialDINOvTransform]      = None,
        masker:        Optional[AnatomicallyGuidedMasker] = None,
        patch_size:    int   = 8,
        hu_clip:       tuple[float, float] = (-1000.0, 1000.0),
        min_body_frac: float = 0.05,
        bg_threshold:  float = -800.0,
        index_json:    Optional[str | Path] = None,
        cache_size:    int   = 4,
        max_volumes:   Optional[int] = None,
    ):
        self.nifti_dir     = Path(nifti_dir)
        self.transform     = transform or AxialDINOvTransform()
        self.masker        = masker    or AnatomicallyGuidedMasker()
        self.patch_size    = patch_size
        self.hu_clip       = hu_clip
        self.min_body_frac = min_body_frac
        self.bg_threshold  = bg_threshold
        self.cache_size    = cache_size
        self.max_volumes   = max_volumes

        self._vol_cache = _LRUVolumeCache(maxsize=cache_size)

        # Build or load index
        if index_json is not None and Path(index_json).exists():
            self.records = self._load_index(Path(index_json))
            logger.info(f"Loaded slice index from {index_json}: {len(self.records):,} slices")
        else:
            self.records = self._build_index()
            logger.info(f"Built slice index: {len(self.records):,} slices "
                        f"from {self.nifti_dir}")
            if index_json is not None:
                self._save_index(Path(index_json))
                logger.info(f"Saved slice index to {index_json}")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _find_nifti_files(self) -> list[Path]:
        files = sorted(set(
            list(self.nifti_dir.rglob("*.nii.gz")) +
            list(self.nifti_dir.rglob("*.nii"))
        ))
        if self.max_volumes is not None:
            files = files[: self.max_volumes]
        return files

    def _build_index(self) -> list[dict]:
        """
        Scan all NIfTI files, load each volume, record valid slice indices.
        Valid = body fraction above min_body_frac.
        """
        files   = self._find_nifti_files()
        records = []

        for fpath in files:
            try:
                data = self._vol_cache.get(str(fpath))
            except Exception as e:
                logger.warning(f"Cannot load {fpath}: {e}")
                continue

            n_slices = data.shape[2]
            for z in range(n_slices):
                sl  = data[:, :, z]
                sl  = np.clip(sl, self.hu_clip[0], self.hu_clip[1])
                frac = float((sl > self.bg_threshold).mean())
                if frac >= self.min_body_frac:
                    records.append({"path": str(fpath), "z": z})

        return records

    def _save_index(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f)

    def _load_index(self, path: Path) -> list[dict]:
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        path   = rec["path"]
        z      = int(rec["z"])

        try:
            data = self._vol_cache.get(path)
        except Exception:
            # Corrupted file at runtime — resample a different index
            return self.__getitem__(random.randint(0, len(self) - 1))

        sl = data[:, :, z].copy()
        sl = np.clip(sl, self.hu_clip[0], self.hu_clip[1])
        hu_slice = torch.from_numpy(sl)  # (H, W) float32

        # Body fraction check (slice may be near-empty at slice boundary)
        frac = float((hu_slice > self.bg_threshold).float().mean())
        if frac < self.min_body_frac:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Multi-crop transform
        sample = self.transform(hu_slice)
        # sample keys: "global_crops", "local_crops", "global_hu", "global_meta"

        # Anatomically guided iBOT masks (CURIA-2 Gaussian prior)
        crop_size = self.transform.global_size
        global_masks = [
            self.masker(meta, self.patch_size, crop_size)
            for meta in sample["global_meta"]
        ]

        return {
            "global_crops":   sample["global_crops"],    # List[Tensor(1, 512, 512)]
            "regional_crops": sample["regional_crops"],  # List[Tensor(1, 224, 224)]
            "local_crops":    sample["local_crops"],     # List[Tensor(1, 112, 112)]
            "global_masks":   global_masks,              # List[Tensor(N_patches,)]
        }
