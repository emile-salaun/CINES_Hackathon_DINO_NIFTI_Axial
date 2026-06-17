from __future__ import annotations

import matplotlib
# Force Matplotlib à utiliser un backend sans interface graphique (idéal pour les serveurs)
matplotlib.use('Agg') 

import nibabel as nib
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from transforms import (
    make_global_crop_v2,
    make_regional_crop_v2,
    make_local_crop_v2,
    apply_window,
    hard_kernel_augment,
    WINDOWS,
    sample_global_window,
    sample_local_window
)

def load_single_middle_slice(nifti_path: str) -> torch.Tensor:
    """Charge le NIfTI et extrait la coupe axiale centrale."""
    print(f"Chargement de {nifti_path}...")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Extraction de la coupe du milieu sur l'axe Z
    z_mid = data.shape[2] // 2
    slice_np = data[:, :, z_mid]
    
    # Clamp des valeurs HU comme dans le pipeline de base
    slice_np = np.clip(slice_np, -1000.0, 1000.0)
    return torch.tensor(slice_np, dtype=torch.float32)

def plot_comparison(nifti_path: str, output_filename: str = "comparison_hard_vs_normal.png"):
    """Génère un comparatif Normal vs Hard Kernel aux 3 échelles pour une coupe."""
    try:
        hu_slice = load_single_middle_slice(nifti_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{nifti_path}' est introuvable.")
        return

    print("Génération des crops avec Content-Aware Cropping (70% contenu min)...")
    # Initialisation des générateurs de crops individuels pour contrôler le pipeline
    global_crop = make_global_crop_v2(output_size=512)
    regional_crop = make_regional_crop_v2(output_size=224)
    local_crop = make_local_crop_v2(output_size=112)

    # Extraction des patchs bruts en HU
    patch_global, _ = global_crop(hu_slice)
    patch_regional, _ = regional_crop(hu_slice)
    patch_local, _ = local_crop(hu_slice)

    # Échantillonnage dynamique des fenêtres pour montrer la diversité
    win_global = sample_global_window()
    win_regional = sample_local_window()
    win_local = sample_local_window()

    # Configuration de la grille : 3 lignes (Global, Régional, Local) x 2 colonnes (Normal, Hard Kernel)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 20))
    fig.suptitle("Comparatif Visuel : Normal vs Hard Kernel Augment (Laplacien + Sharpening)", fontsize=18, y=0.96)

    scales = [
        ("Échelle Globale (512x512)", patch_global, win_global, axes[0]),
        ("Échelle Régionale (224x224)", patch_regional, win_regional, axes[1]),
        ("Échelle Locale (112x112)", patch_local, win_local, axes[2])
    ]

    print("Application du fenêtrage et du Hard Kernel Augment...")
    for title, patch, win_key, ax_row in scales:
        # 1. Version Normale (Uniquement fenêtrée, plage [0, 1])
        img_normal = apply_window(patch, **WINDOWS[win_key])
        
        # 2. Version Hard Kernel (Fenêtrée + Unsharp Masking + Laplacien + Bruit)
        # hard_kernel_augment prend un tenseur de forme (1, H, W)
        img_tensor = img_normal.unsqueeze(0)
        img_hard = hard_kernel_augment(img_tensor).squeeze(0)

        # Affichage de la colonne de gauche : Normal
        # On force vmin=0 et vmax=1 pour que l'échelle de gris soit strictement identique à droite et à gauche
        ax_row[0].imshow(img_normal.numpy(), cmap="gray", vmin=0, vmax=1)
        ax_row[0].set_title(f"{title} - NORMAL\nFenêtre : {win_key.upper()}", fontsize=12, fontweight="bold")
        ax_row[0].axis("off")

        # Affichage de la colonne de droite : Hard Kernel
        ax_row[1].imshow(img_hard.numpy(), cmap="gray", vmin=0, vmax=1)
        ax_row[1].set_title(f"{title} - HARD KERNEL\nRehaussement de bords + Bruit", fontsize=12, fontweight="bold")
        ax_row[1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Sauvegarde du fichier à côté du script
    output_path = os.path.join(os.path.dirname(__file__) if os.path.dirname(__file__) else ".", output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Image comparative sauvegardée avec succès ici : {output_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    CHEMIN_NIFTI = "/data/sep24nifti/A10038136866/A10038136866.nii.gz"
    plot_comparison(CHEMIN_NIFTI)