#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DINOv3 Phase 1 — axial NIfTI CT slices
# Jean-Zay  ·  1 nœud × 8 × A100-80G
#
# Soumission :
#   sbatch $WORK/nifti_dino_axial/submit_jz.slurm
#
# Reprise automatique après préemption :
#   train.py détecte le dernier ckpt_step*.pt dans OUTPUT_DIR.
#   Il suffit de re-soumettre le même script sans aucune modification.
#
# Pour tester (2h max, accès prioritaire) :
#   Remplacer qos_gpu_a100-t3 par qos_gpu_a100-dev et --time par 02:00:00
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=dino_axial_p1
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64          # 8 GPU × 8 workers DataLoader
#SBATCH --time=00:30:00
#SBATCH -A ssb@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev       # t3 = 20 h max ; dev = 2 h (accès rapide)
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --hint=nomultithread

set -eo pipefail

# ── Répertoires ───────────────────────────────────────────────────────────────
SCRIPT_DIR=$WORK/nifti_dino_axial
OUTPUT_DIR=$WORK/checkpoints/dino_axial_phase1

mkdir -p "$OUTPUT_DIR"

# ── Modules Jean-Zay ─────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
source $WORK/venv_dino/bin/activate

# ── Environnement Python ─────────────────────────────────────────────────────
export PYTHONPATH="$WORK:$SCRIPT_DIR/models_pretrained/flexiCT/FlexiCT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

echo "Job $SLURM_JOB_ID — $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
echo "Output dir : $OUTPUT_DIR"
echo "Data dir   : $SCRATCH/ct_nifti"

# ── Lancement ────────────────────────────────────────────────────────────────
# --resume est détecté automatiquement depuis output_dir (dernier ckpt_step*.pt)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    "$SCRIPT_DIR/train.py" \
    --config     "$SCRIPT_DIR/configs/phase1.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --nifti_dir  "$SCRATCH/ct_nifti" \
    --pretrained "$SCRIPT_DIR/models_pretrained/flexiCT/2D_final_model.pth" \
