"""
Dataset loading axial slices from pre-processed .npy volumes.

Pre-requisite
-------------
Run  preprocess_nifti_to_npy.py  once to produce:
  • One  .npy  file per volume  (float16, RAS-reoriented, HU-clipped)
  • A JSON slice index  (list of {"path": str, "z": int})

At training time this dataset:
  - Memory-maps .npy volumes  (np.load(..., mmap_mode='r'))
  - Extracts axial slices as  volume[:, :, z]  — zero decompression overhead
  - Has zero nibabel / reorientation / HU-clipping logic

Volume caching
--------------
Each DataLoader worker keeps its own LRU cache of open memory-mapped arrays.
Because mmap_mode='r' opens a file descriptor rather than copying data into
RAM, cache entries are cheap.  The OS page cache does the real heavy lifting.
"""
from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..transforms import AxialDINOvTransform
from ..masking import AnatomicallyGuidedMasker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LRU cache for memory-mapped volumes (per-worker, no cross-process sharing)
# ---------------------------------------------------------------------------

class _MmapVolumeCache:
    """
    Keeps up to `maxsize` memory-mapped .npy arrays open.

    np.load(path, mmap_mode='r') returns a numpy memmap: the OS pages in
    only the bytes you actually access, so holding many open costs very
    little RSS unless you touch every slice.
    """

    def __init__(self, maxsize: int = 8):
        self._cache: OrderedDict[str, np.memmap] = OrderedDict()
        self._maxsize = maxsize

    def get(self, path: str) -> np.memmap:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        arr = np.load(path, mmap_mode="r")   # shape (H, W, Z), float16
        # No reorientation needed — already done at preprocessing time.

        if len(self._cache) >= self._maxsize:
            evicted_path, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted volume from mmap cache: {evicted_path}")

        self._cache[path] = arr
        return arr


# ---------------------------------------------------------------------------
# Collate function  (identical contract to the original NiftiAxialDataset)
# ---------------------------------------------------------------------------

def axial_dinov_collate(batch: list[dict]) -> dict:
    """
    Collate multi-crop dicts into batched tensors.

    Returns:
        global_crops : (n_global, B, 1, 256, 256)
        local_crops  : (n_local,  B, 1, 112, 112)
        global_masks : (n_global, B, N_patches) bool
    """
    n_global = len(batch[0]["global_crops"])
    n_local  = len(batch[0]["local_crops"])

    global_crops = torch.stack(
        [torch.stack([b["global_crops"][i] for b in batch]) for i in range(n_global)]
    )
    local_crops = torch.stack(
        [torch.stack([b["local_crops"][i] for b in batch]) for i in range(n_local)]
    )
    global_masks = torch.stack(
        [torch.stack([b["global_masks"][i] for b in batch]) for i in range(n_global)]
    )
    return {
        "global_crops": global_crops,
        "local_crops":  local_crops,
        "global_masks": global_masks,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NpyAxialDataset(Dataset):
    """
    Fast axial CT slice dataset backed by pre-processed .npy volumes.

    Compared to NiftiAxialDataset:
      • No nibabel dependency at training time
      • No reorientation (baked in at preprocessing)
      • No HU clipping (baked in at preprocessing)
      • Slices loaded via memory-map → only touched pages are paged in
      • Index must be provided (built by preprocess_nifti_to_npy.py)

    Args:
        index_json:     Path to the slice index JSON produced by the
                        preprocessing script.
        transform:      AxialDINOvTransform instance.
        masker:         AnatomicallyGuidedMasker instance.
        patch_size:     ViT patch size (default 8).
        bg_threshold:   HU below this is background (float16-safe value).
                        Used only for the runtime body-fraction guard.
        min_body_frac:  Minimum fraction of non-background pixels.
                        Usually not triggered because the index was already
                        filtered, but guards against edge cases.
        cache_size:     Number of mmap'd volumes to keep open per worker.
    """

    def __init__(
        self,
        index_json:    str | Path,
        transform:     Optional[AxialDINOvTransform]      = None,
        masker:        Optional[AnatomicallyGuidedMasker] = None,
        patch_size:    int   = 8,
        bg_threshold:  float = -800.0,
        min_body_frac: float = 0.05,
        cache_size:    int   = 8,
    ):
        self.transform     = transform or AxialDINOvTransform()
        self.masker        = masker    or AnatomicallyGuidedMasker()
        self.patch_size    = patch_size
        self.bg_threshold  = bg_threshold
        self.min_body_frac = min_body_frac

        self._vol_cache = _MmapVolumeCache(maxsize=cache_size)

        index_path = Path(index_json)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Slice index not found: {index_path}\n"
                "Run preprocess_nifti_to_npy.py first."
            )
        with open(index_path) as f:
            self.records: list[dict] = json.load(f)

        logger.info(f"Loaded slice index from {index_path}: {len(self.records):,} slices")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec  = self.records[idx]
        path = rec["path"]
        z    = int(rec["z"])

        # ---- load slice (mmap — only this page is read from disk) ---------
        try:
            vol = self._vol_cache.get(path)
        except Exception as exc:
            logger.warning(f"Cannot mmap {path}: {exc}. Resampling.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # vol shape: (H, W, Z_total), dtype float16
        # Axial slice — no copy until we need a writable float32 tensor
        sl = vol[:, :, z]                           # memmap view, float16

        # Cast to float32 for torch (float16 tensor ops are limited on CPU)
        # hu_slice = torch.from_numpy(sl.astype(np.float32))   # (H, W)
        hu_slice = torch.from_numpy(sl) 

        # Runtime body-fraction guard (almost always passes — index pre-filtered)
        frac = float((hu_slice > self.bg_threshold).float().mean())
        if frac < self.min_body_frac:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # ---- multi-crop transform -----------------------------------------
        sample = self.transform(hu_slice)
        # keys: "global_crops", "local_crops", "global_hu", "global_meta"

        # ---- anatomically guided iBOT masks --------------------------------
        crop_size    = self.transform.global_size
        global_masks = [
            self.masker(meta, self.patch_size, crop_size)
            for meta in sample["global_meta"]
        ]

        return {
            "global_crops": sample["global_crops"],   # List[Tensor(1,256,256)]
            "local_crops":  sample["local_crops"],    # List[Tensor(1,112,112)]
            "global_masks": global_masks,             # List[Tensor(N_patches,)]
        }