from __future__ import annotations

import matplotlib
# Force Matplotlib à utiliser un backend sans interface graphique (idéal pour les serveurs)
matplotlib.use('Agg') 

import nibabel as nib
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from transforms import AxialDINOvTransform, apply_window, WINDOWS

def load_middle_slices(nifti_path: str, num_slices: int = 3) -> list[torch.Tensor]:
    """Charge le NIfTI et extrait les coupes centrales sur l'axe Z."""
    print(f"Chargement de {nifti_path}...")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    z_mid = data.shape[2] // 2
    
    slices = []
    start_z = z_mid - (num_slices // 2)
    
    for z in range(start_z, start_z + num_slices):
        slice_np = data[:, :, z]
        slice_np = np.clip(slice_np, -1000.0, 1000.0)
        slice_tensor = torch.tensor(slice_np, dtype=torch.float32)
        slices.append((z, slice_tensor))
        
    return slices

def plot_transformations(nifti_path: str, output_filename: str = "transformations_output.png"):
    """Applique les transformations et sauvegarde le résultat dans une image."""
    slices_data = load_middle_slices(nifti_path, num_slices=3)
    
    print("Application des transformations...")
    # Mise à jour avec les nouveaux paramètres par défaut : 2 globaux, 4 régionaux, 4 locaux
    transform = AxialDINOvTransform(n_global=2, n_regional=4, n_local=4, global_size=512, regional_size=224, local_size=112)
    
    # 1 Original + 2 Globaux + 4 Régionaux + 4 Locaux = 11 colonnes
    n_cols = 1 + 2 + 4 + 4 
    n_rows = len(slices_data)
    
    # Figure plus large pour accommoder les 11 colonnes
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(26, 4 * n_rows))
    fig.suptitle("Visualisation des Transformations CT (Globales, Régionales, Locales)", fontsize=18)
    
    for row_idx, (z_idx, hu_slice) in enumerate(slices_data):
        result = transform(hu_slice)
        global_crops = result["global_crops"]
        regional_crops = result["regional_crops"]
        local_crops = result["local_crops"]
        
        # --- Original (Colonne 0) ---
        ax_orig = axes[row_idx, 0]
        orig_vis = apply_window(hu_slice, **WINDOWS["abdo"]).numpy()
        ax_orig.imshow(orig_vis, cmap="gray")
        ax_orig.set_title(f"Original (Z={z_idx})\nFenêtre Abdo")
        ax_orig.axis("off")
        
        # --- Globaux (Colonnes 1 et 2) ---
        for i in range(2):
            ax_glob = axes[row_idx, 1 + i]
            img_glob = global_crops[i].squeeze().numpy()
            ax_glob.imshow(img_glob, cmap="gray")
            # Titre ajusté car la fenêtre globale est maintenant dynamique (50% abdo, 50% autres)
            ax_glob.set_title(f"Global {i+1}\n(512x512)")
            ax_glob.axis("off")
            
        # --- Régionaux (Colonnes 3 à 6) ---
        for i in range(4):
            ax_reg = axes[row_idx, 3 + i]
            img_reg = regional_crops[i].squeeze().numpy()
            ax_reg.imshow(img_reg, cmap="gray")
            ax_reg.set_title(f"Régional {i+1}\n(224x224)")
            ax_reg.axis("off")
            
        # --- Locaux (Colonnes 7 à 10) ---
        for i in range(4):
            ax_loc = axes[row_idx, 7 + i]
            img_loc = local_crops[i].squeeze().numpy()
            ax_loc.imshow(img_loc, cmap="gray")
            ax_loc.set_title(f"Local {i+1}\n(112x112)")
            ax_loc.axis("off")

    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Image générée et sauvegardée avec succès ici : {output_path}")
    
    plt.close(fig) # Libère la mémoire

if __name__ == "__main__":
    CHEMIN_NIFTI = "/data/sep24nifti/A10038136866/A10038136866.nii.gz"
    
    try:
        plot_transformations(CHEMIN_NIFTI)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{CHEMIN_NIFTI}' est introuvable.")