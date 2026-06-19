#!/bin/bash

#==================================================================
#  NIfTI DINO Axial — Phase 1 pre-training
#==================================================================

# -----------------------------------------------------------------
# Environment
# -----------------------------------------------------------------

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIR="${DIR:-/lus/scratch/BCINES/dci/salaun/hackathon-juin/gh/CINES_Hackathon_DINO_NIFTI_Axial/nifti_dino_axial}"

# -----------------------------------------------------------------
# Args (optionnels)
# -----------------------------------------------------------------

NNODES="${1:-1}"
GPUS_PER_NODE="${2:-8}"
CONSTRAINT="${3:-MI250}"
CPUS_PER_TASK="${4:-32}"
NODELIST_ARG=""
[[ -n "${5:-}" ]] && NODELIST_ARG="--nodelist=${5}"

if [[ "$CONSTRAINT" != "MI250" && "$CONSTRAINT" != "MI300" ]]; then
    echo "Error: constraint must be MI250 or MI300 (got '${CONSTRAINT}')"
    exit 1
fi

export GPUS_PER_NODE  # lu par cluster.sh via --export=ALL
export CONSTRAINT     # lu par run.sh (case BIND_STRATEGY)
export BIND_STRATEGY  # opt-in : 'mi300_srun4' active le pattern HPE multi-node

NTASKS_PER_NODE=1
case "${BIND_STRATEGY:-}" in
    mi300_srun4)
        NTASKS_PER_NODE="${GPUS_PER_NODE}"
        ;;
esac

export LOGS_DIR="$SCRIPT_DIR/logs/${CONSTRAINT}/${NNODES}nodes_${GPUS_PER_NODE}gpus"
mkdir -p "$LOGS_DIR"

# -----------------------------------------------------------------
# Config summary
# -----------------------------------------------------------------

echo ""
echo "  Job        : nifti_dino"
echo "  Nodes      : $NNODES"
echo "  GPUs/node  : $GPUS_PER_NODE  ($CONSTRAINT)"
echo "  CPUs/task  : $CPUS_PER_TASK"
echo "  Constraint : $CONSTRAINT"
echo "  Config     : $DIR/configs/phase1.yaml"
echo "  Pretrained : $DIR/models_pretrained/flexiCT/2D_final_model.pth"
echo "  Output     : $DIR/checkpoints"
echo "  Logs       : $LOGS_DIR"
[[ -n "$NODELIST_ARG" ]] && echo "  Nodelist   : ${5}"
echo ""

# -----------------------------------------------------------------
# Job submit
# -----------------------------------------------------------------

JOB_ID=$(sbatch --parsable \
    --account="${SBATCH_ACCOUNT:-dci}" \
    ${SBATCH_RESERVATION:+--reservation=$SBATCH_RESERVATION} \
    --job-name=nifti_dino \
    --nodes="$NNODES" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --ntasks-per-node="$NTASKS_PER_NODE" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time=10:00:00 \
    --exclusive \
    --constraint="$CONSTRAINT" \
    --output="$LOGS_DIR/run_%j.out" \
    --error="$LOGS_DIR/run_%j.out" \
    --export=ALL \
    $NODELIST_ARG \
    -- "$SCRIPT_DIR/run.sh")

sleep 5

NODESLIST=$(squeue -h -j "$JOB_ID" -o "%N")

echo "Job $JOB_ID submitted at $(date -R) on $NODESLIST"
echo ""