#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DINOv3 Phase 1 — axial NIfTI CT slices
# Jean-Zay  ·  N_NODES nœuds × 8 × A100-80G
#
# Soumission :
#   sbatch $WORK/nifti_dino_axial/submit_jz_multinode.slurm
#
# Pour changer le nombre de nœuds : modifier --nodes et N_NODES ci-dessous.
#
# Mise à l'échelle du LR (règle de la racine carrée) :
#   1 nœud  (effective batch  80) → base_lr = 2e-4  (config par défaut)
#   4 nœuds (effective batch 320) → base_lr = 4e-4  (×√4 = ×2)
#   8 nœuds (effective batch 640) → base_lr = 5.7e-4 (×√8 ≈ ×2.8)
#
# Reprise automatique après préemption :
#   train.py détecte le dernier ckpt_step*.pt dans OUTPUT_DIR.
#   Il suffit de re-soumettre le même script sans aucune modification.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=dino_axial_p1_mn
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1              # 1 torchrun launcher par nœud
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64              # 8 GPU × 8 workers DataLoader
#SBATCH --time=20:00:00
#SBATCH -A ssb@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3           # t3 = 20 h max ; dev = 2 h (accès rapide)
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --hint=nomultithread

set -eo pipefail

# ── Répertoires ───────────────────────────────────────────────────────────────
SCRIPT_DIR=$WORK/nifti_dino_axial
OUTPUT_DIR=$WORK/checkpoints/dino_axial_phase1_mn

mkdir -p "$OUTPUT_DIR"

# ── Modules Jean-Zay ─────────────────────────────────────────────────────────
module purge
module load arch/a100
module load pytorch-gpu/py3/2.4.0
source $WORK/venv_dino/bin/activate

# ── Environnement Python ─────────────────────────────────────────────────────
export PYTHONPATH="$WORK:$SCRIPT_DIR/models_pretrained/flexiCT/FlexiCT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=1  ← NE PAS décommenter en multi-nœud

# ── Rendezvous multi-nœud ────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Job $SLURM_JOB_ID — $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
echo "Nodes      : $SLURM_NNODES  ($SLURM_JOB_NODELIST)"
echo "Master     : $MASTER_ADDR:$MASTER_PORT"
echo "Output dir : $OUTPUT_DIR"
echo "Data dir   : $SCRATCH/ct_nifti"

# ── Lancement ────────────────────────────────────────────────────────────────
# srun lance 1 processus torchrun par nœud ; torchrun gère les 8 GPU locaux.
# --rdzv_backend=c10d est requis pour le multi-nœud (remplace --standalone).
srun --ntasks-per-node=1 torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$SCRIPT_DIR/train.py" \
    --config     "$SCRIPT_DIR/configs/phase1.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --nifti_dir  "$SCRATCH/ct_nifti" \
    --pretrained "$SCRIPT_DIR/models_pretrained/flexiCT/2D_final_model.pth"