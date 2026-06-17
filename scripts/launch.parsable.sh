#!/bin/bash

#==================================================================
#  NIfTI DINO Axial — Phase 1 pre-training
#==================================================================

# -----------------------------------------------------------------
# Environment
# -----------------------------------------------------------------

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIR="/lus/scratch/BCINES/dci/salaun/hackathon-juin/gh/CINES_Hackathon_DINO_NIFTI_Axial/nifti_dino_axial"
export LOGS_DIR="$SCRIPT_DIR/logs"

# -----------------------------------------------------------------
# Args (optionnels)
# -----------------------------------------------------------------

NNODES="${1:-1}"
GPUS_PER_NODE="${2:-8}"
NODELIST_ARG=""
[[ -n "${3:-}" ]] && NODELIST_ARG="--nodelist=${3}"

export GPUS_PER_NODE  # lu par cluster.sh via --export=ALL

export LOGS_DIR="$SCRIPT_DIR/logs-comp/${NNODES}nodes_${GPUS_PER_NODE}gpus"
mkdir -p "$LOGS_DIR"

# -----------------------------------------------------------------
# Config summary
# -----------------------------------------------------------------

echo ""
echo "  Job        : nifti_dino"
echo "  Nodes      : $NNODES"
echo "  GPUs/node  : $GPUS_PER_NODE  (MI250)"
echo "  CPUs/task  : 64"
echo "  Config     : $DIR/configs/phase1.yaml"
echo "  Pretrained : $DIR/models_pretrained/flexiCT/2D_final_model.pth"
echo "  Output     : $DIR/checkpoints"
echo "  Logs       : $LOGS_DIR"
[[ -n "$NODELIST_ARG" ]] && echo "  Nodelist   : ${3}"
echo ""

# -----------------------------------------------------------------
# Job submit
# -----------------------------------------------------------------

JOB_ID=$(sbatch --parsable \
    --account=dci \
    --job-name=nifti_dino \
    --nodes="$NNODES" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --ntasks-per-node=1 \
    --cpus-per-task=64 \
    --time=08:00:00 \
    --exclusive \
    --constraint=MI250 \
    --output="$LOGS_DIR/run_%j.out" \
    --error="$LOGS_DIR/run_%j.out" \
    --export=ALL \
    $NODELIST_ARG \
    -- "$SCRIPT_DIR/run.sh")

sleep 5

NODESLIST=$(squeue -h -j "$JOB_ID" -o "%N")

echo "Job $JOB_ID submitted at $(date -R) on $NODESLIST"
echo ""