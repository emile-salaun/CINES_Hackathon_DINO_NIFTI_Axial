#!/bin/bash

#==================================================================
#  NIfTI DINO Axial — job body (sourced by sbatch via launch.sh)
#==================================================================

export DIR="$SCRATCH/hackathon-juin/gh/CINES_Hackathon_DINO_NIFTI_Axial/nifti_dino_axial"

mkdir -p ./logs

source /lus/work/CT3/cad17796/SHARED/.venv8/bin/activate.poverlay

export LD_LIBRARY_PATH=/lus/work/CT3/cad17796/SHARED/spack-install-hipblaslt-patch/linux-zen3/hipblaslt-develop-24ro/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/lus/work/CT3/cad17796/SHARED/spack-install-hipblaslt-patch/linux-zen3/hipblaslt-develop-24ro/lib64/libhipblaslt.so.1

export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

export OMP_NUM_THREADS="${CPUS_PER_TASK:-32}"
export HSA_FORCE_FINE_GRAIN_PCIE=1 
export HSA_XNACK=1


echo ""
echo "================================================================"
echo "  NIfTI DINO Axial — Job configuration"
echo "----------------------------------------------------------------"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Nodes        : ${SLURM_JOB_NUM_NODES}  (${SLURM_JOB_NODELIST})"
echo "  GPUs/node    : ${GPUS_PER_NODE}  (total: ${TOTAL_GPUS})"
echo "  CPUs/task    : ${CPUS_PER_TASK:-32}"
echo "  OMP threads  : ${OMP_NUM_THREADS}"
echo "  Constraint   : ${$CONSTRAINT":-N/A}"
echo "  Config       : ${DIR}/configs/phase1.yaml"
echo "  Output       : ${DIR}/checkpoints"
echo "================================================================"
echo ""


echo "Job started at $(date -R)"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

srun --ntasks-per-node=1 --gpus-per-task="${GPUS_PER_NODE}" \
    -- torchrun \
        --nnodes="${SLURM_JOB_NUM_NODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv-id="${SLURM_JOB_ID}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1):29400" \
        --max-restarts=0 \
        -- "$DIR/train.py" \
            --config "$DIR/configs/phase1.yaml" \
            --pretrained "$DIR/models_pretrained/flexiCT/2D_final_model.pth" \
            --nifti_dir /lus/work/CT3/cad17796/SHARED/merlinabdominalctdataset/merlin_data \
            --output_dir "$DIR/$JOB_ID" \
            --max_volumes 100 
            # --profile
            

echo "Job ended at $(date -R)"