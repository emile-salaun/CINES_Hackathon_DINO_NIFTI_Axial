#!/bin/bash
#SBATCH --job-name=dino_smoke_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32          # 4 processus GPU x 8 workers DataLoader
#SBATCH --time=00:45:00
#SBATCH -A ssb@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev       # file courte (acces rapide), max 2h
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --hint=nomultithread

set -eo pipefail

SCRIPT_DIR=$WORK/nifti_dino_axial

module purge
module load arch/a100
module load pytorch-gpu/py3/2.4.0
source $WORK/venv_dino/bin/activate

export PYTHONPATH="$WORK:$SCRIPT_DIR/models_pretrained/flexiCT/FlexiCT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

echo "Job $SLURM_JOB_ID — $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"

OUTPUT_DIR=$WORK/checkpoints/smoke_test
mkdir -p "$OUTPUT_DIR"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    "$SCRIPT_DIR/smoke_test.py" \
    --nifti_dir   "$SCRATCH/ct_nifti" \
    --max_volumes 20 \
    --pretrained  "$SCRIPT_DIR/models_pretrained/flexiCT/2D_final_model.pth" \
    --output_dir  "$OUTPUT_DIR"