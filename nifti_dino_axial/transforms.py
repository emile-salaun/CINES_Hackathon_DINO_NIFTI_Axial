"""
Augmentation transforms for axial CT slices (NIfTI input).

Améliorations :
  - Fenêtrage global dynamique (50% Abdo, 50% Soft/Large/Bone).
  - Introduction de crops "régionaux" (224x224).
  - Fenêtrage local strict : 40% abdo, 20% soft (100-200), 20% bone (800-2000), 20% custom (-1000 à 1000).
  - Application systématique du ContentAwareCropV2 (Curia 2) sur tous les crops.
  - Exigence FORTE de contenu : min_importance = 0.70 (70% de contenu minimum).
"""
from __future__ import annotations

import random
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# CT windows  (center HU, width HU)
# ---------------------------------------------------------------------------
WINDOWS: dict[str, dict[str, float]] = {
    "abdo":   {"center":   60.0, "width":  360.0},   # abdominal
    "soft":   {"center":  100.0, "width":  200.0},   # soft tissue (100-200)
    "bone":   {"center":  800.0, "width": 2000.0},   # bone (800-2000)
    "large":  {"center": -500.0, "width": 1500.0},   # large pulmonary window
    "custom": {"center":    0.0, "width": 2000.0},   # Plage [-1000, 1000]
}

def apply_window(hu_patch: torch.Tensor, center: float, width: float) -> torch.Tensor:
    """Map HU → [0, 1] using a radiological display window."""
    lo = center - width / 2.0
    return torch.clamp((hu_patch - lo) / width, 0.0, 1.0)

def sample_global_window() -> str:
    """Privilégie l'abdo (50%) mais permet au modèle d'apprendre d'autres macro-contrastes."""
    r = random.random()
    if r < 0.50:
        return "abdo"
    elif r < 0.80:
        return "soft"
    elif r < 0.90:
        return "large"
    else:
        return "bone"

def sample_local_window() -> str:
    """
    Probabilités strictes pour les crops locaux :
    40% abdo
    20% soft (100-200)
    20% bone (800-2000)
    20% custom (-1000 à 1000)
    """
    r = random.random()
    if r < 0.50:
        return "abdo"
    elif r < 0.80:
        return "soft"
    elif r < 0.90:
        return "bone"
    else:
        return "custom"


# ---------------------------------------------------------------------------
# Tissue importance map
# ---------------------------------------------------------------------------
_TISSUE_TABLE: list[tuple[float, float, float]] = [
    (-1000.0,  -900.0,  0.00),   
    ( -900.0,  -500.0,  0.15),   
    ( -500.0,  -200.0,  0.40),   
    ( -200.0,   -50.0,  0.60),   
    (  -50.0,   200.0,  1.00),   
    (  200.0,   400.0,  1.30),   
    (  400.0,  3000.0,  0.90),   
]

def compute_importance_map(hu_slice: torch.Tensor) -> torch.Tensor:
    imp = torch.zeros_like(hu_slice)
    for lo, hi, w in _TISSUE_TABLE:
        mask = (hu_slice >= lo) & (hu_slice < hi)
        imp[mask] = w
    return imp


# ---------------------------------------------------------------------------
# ContentAwareCropV2
# ---------------------------------------------------------------------------
class ContentAwareCropV2:
    def __init__(
        self,
        output_size: int,
        scale: tuple[float, float],
        min_importance: float = 0.70, # Forcé à 70% de contenu
        bg_threshold: float = -800.0,
        max_attempts: int = 20,       # Augmenté pour donner plus de chances de trouver une zone dense
    ):
        self.output_size    = output_size
        self.scale          = scale
        self.min_importance = min_importance
        self.bg_threshold   = bg_threshold
        self.max_attempts   = max_attempts

    def _importance_centroid(self, imp: torch.Tensor) -> tuple[float, float, float, float]:
        H, W = imp.shape
        total = imp.sum().clamp(min=1e-6)

        ys = torch.arange(H, dtype=torch.float32, device=imp.device)
        xs = torch.arange(W, dtype=torch.float32, device=imp.device)

        cy = float((imp.sum(dim=1) * ys).sum() / total)
        cx = float((imp.sum(dim=0) * xs).sum() / total)

        var_r = float(((ys - cy) ** 2 * imp.sum(dim=1)).sum() / total)
        var_c = float(((xs - cx) ** 2 * imp.sum(dim=0)).sum() / total)
        sigma_r = max(math.sqrt(var_r), 16.0)
        sigma_c = max(math.sqrt(var_c), 16.0)

        return cy, cx, sigma_r, sigma_c

    def __call__(self, hu_slice: torch.Tensor) -> tuple[torch.Tensor, dict]:
        H, W = hu_slice.shape
        out  = self.output_size

        imp = compute_importance_map(hu_slice)
        cy, cx, sigma_r, sigma_c = self._importance_centroid(imp)

        best_crop:  tuple[int, int, int] | None = None
        best_score: float = -1.0

        for _ in range(self.max_attempts):
            area_frac = random.uniform(self.scale[0], self.scale[1])
            crop_side = max(out, int((H * W * area_frac) ** 0.5))
            crop_side = min(crop_side, H, W)
            half = crop_side // 2

            cy_s = torch.zeros(1).normal_(cy, sigma_r).clamp(half, H - half).item()
            cx_s = torch.zeros(1).normal_(cx, sigma_c).clamp(half, W - half).item()

            top  = max(0, min(int(cy_s) - half, H - crop_side))
            left = max(0, min(int(cx_s) - half, W - crop_side))

            patch_imp = imp[top : top + crop_side, left : left + crop_side]
            score = float(patch_imp.mean())

            if score > best_score:
                best_score = score
                best_crop  = (top, left, crop_side)

            # Filtre strict : Le crop doit avoir au moins 70% de "matière"
            if score >= self.min_importance:
                break

        assert best_crop is not None
        top, left, crop_side = best_crop
        patch = hu_slice[top : top + crop_side, left : left + crop_side]
        patch = TF.resize(
            patch.unsqueeze(0),
            [out, out],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        ).squeeze(0)

        scale = out / crop_side
        meta  = {
            "top":        top,
            "left":       left,
            "crop_side":  crop_side,
            "body_cy_crop": (cy - top) * scale,
            "body_cx_crop": (cx - left) * scale,
        }
        return patch, meta


def make_global_crop_v2(output_size: int = 512) -> ContentAwareCropV2:
    return ContentAwareCropV2(output_size=output_size, scale=(0.4, 1.0),
                              min_importance=0.70, max_attempts=20)

def make_regional_crop_v2(output_size: int = 224) -> ContentAwareCropV2:
    return ContentAwareCropV2(output_size=output_size, scale=(0.15, 0.4),
                              min_importance=0.70, max_attempts=20)

def make_local_crop_v2(output_size: int = 112) -> ContentAwareCropV2:
    return ContentAwareCropV2(output_size=output_size, scale=(0.08, 0.20),
                              min_importance=0.70, max_attempts=20)


# ---------------------------------------------------------------------------
# Kernel filter augmentations
# ---------------------------------------------------------------------------
def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    ks = max(3, int(2 * math.ceil(2 * sigma) + 1))
    if ks % 2 == 0:
        ks += 1
    return TF.gaussian_blur(x, kernel_size=ks, sigma=sigma)

def soft_kernel_augment(x: torch.Tensor) -> torch.Tensor:
    sigma = random.uniform(0.5, 1.5)
    return _gaussian_blur_2d(x, sigma)

def hard_kernel_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Filtre de reconstruction CT 'fort' : sharpening + bruit.
    """
    alpha = random.uniform(0.4, 1.0)
    sigma = random.uniform(0.8, 1.8)
    blurred  = _gaussian_blur_2d(x, sigma)
    sharpened = x + alpha * (x - blurred)
    device = x.device
    dtype = x.dtype
    edge_kernel = torch.tensor([[-1., -1., -1.],
                                [-1.,  8., -1.],
                                [-1., -1., -1.]], device=device, dtype=dtype)
    edge_kernel = edge_kernel.view(1, 1, 3, 3)
    
    edges = F.conv2d(sharpened, edge_kernel, padding=1)
    edge_strength = random.uniform(0.5, 1.2)     
    enhanced = sharpened + edge_strength * edges
    sharpened = torch.clamp(enhanced, 0.0, 1.0)
    noise_scale = random.uniform(0.01, 0.025)
    noise    = torch.randn_like(sharpened) * noise_scale
    return torch.clamp(sharpened + noise, 0.0, 1.0)

def apply_kernel_augment(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if random.random() >= p:
        return x
    if random.random() < 0.5:
        return soft_kernel_augment(x)
    return hard_kernel_augment(x)


# ---------------------------------------------------------------------------
# FlexiCT Phase 1 augmentations
# ---------------------------------------------------------------------------
def gaussian_noise(x: torch.Tensor, sigma_max: float = 0.05) -> torch.Tensor:
    sigma = random.uniform(0.0, sigma_max)
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)

def contrast_jitter(x: torch.Tensor, gamma_range: tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    gamma = random.uniform(*gamma_range)
    return torch.clamp(x ** gamma, 0.0, 1.0)

def simulated_lowres(x: torch.Tensor, scale_range: tuple[float, float] = (0.6, 1.0)) -> torch.Tensor:
    if random.random() < 0.25:
        scale  = random.uniform(*scale_range)
        h, w   = x.shape[-2], x.shape[-1]
        sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
        x = TF.resize(x, [sh, sw], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        x = TF.resize(x, [h, w],   interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return x

def intensity_scaling(x: torch.Tensor, scale_range: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    return torch.clamp(x * random.uniform(*scale_range), 0.0, 1.0)

def ct_intensity_augment(x: torch.Tensor) -> torch.Tensor:
    x = gaussian_noise(x)
    x = contrast_jitter(x)
    x = simulated_lowres(x)
    x = intensity_scaling(x)
    return x


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def zscore_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean()
    std  = x.std().clamp(min=eps)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Multi-crop transform
# ---------------------------------------------------------------------------
class AxialDINOvTransform:
    def __init__(
        self,
        n_global:      int   = 2,
        n_regional:    int   = 4,
        n_local:       int   = 4,
        global_size:   int   = 512,
        regional_size: int   = 224, 
        local_size:    int   = 112,
        hflip_prob:    float = 0.5,
        filter_prob:   float = 0.5,
    ):
        self.n_global      = n_global
        self.n_regional    = n_regional
        self.n_local       = n_local
        self.global_size   = global_size    # queried by dataset.py for mask generation
        self.regional_size = regional_size
        self.local_size    = local_size
        self.hflip_prob    = hflip_prob
        self.filter_prob   = filter_prob

        self.global_crop   = make_global_crop_v2(output_size=global_size)
        self.regional_crop = make_regional_crop_v2(output_size=regional_size)
        self.local_crop    = make_local_crop_v2(output_size=local_size)

    def _augment(self, hu_patch: torch.Tensor, window_key: str) -> torch.Tensor:
        if random.random() < self.hflip_prob:
            hu_patch = hu_patch.flip(-1)

        win = WINDOWS[window_key]
        x   = apply_window(hu_patch, **win)
        x   = x.unsqueeze(0)

        x = apply_kernel_augment(x, p=self.filter_prob)
        x = ct_intensity_augment(x)
        x = zscore_norm(x)
        return x

    def __call__(self, hu_slice: torch.Tensor) -> dict:
        global_crops, global_hu, global_meta = [], [], []
        
        # 1. Globaux (512)
        for _ in range(self.n_global):
            wk = sample_global_window()
            patch_hu, meta = self.global_crop(hu_slice)
            global_hu.append(patch_hu)
            global_meta.append(meta)
            crop_t = self._augment(patch_hu, wk)
            global_crops.append(crop_t)

        # 2. Régionaux (224)
        regional_crops = []
        for _ in range(self.n_regional):
            wk = sample_local_window()
            patch_hu, _ = self.regional_crop(hu_slice)
            crop_t = self._augment(patch_hu, wk)
            regional_crops.append(crop_t)

        # 3. Locaux (112)
        local_crops = []
        for _ in range(self.n_local):
            wk = sample_local_window()
            patch_hu, _ = self.local_crop(hu_slice)
            crop_t = self._augment(patch_hu, wk)
            local_crops.append(crop_t)

        return {
            "global_crops":   global_crops,
            "regional_crops": regional_crops,
            "local_crops":    local_crops,
            "global_hu":      global_hu,
            "global_meta":    global_meta,
        }