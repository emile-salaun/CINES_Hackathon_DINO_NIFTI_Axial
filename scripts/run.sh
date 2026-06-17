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

export OMP_NUM_THREADS="${CPUS_PER_TASK:-64}"
export HSA_FORCE_FINE_GRAIN_PCIE=1 
export HSA_XNACK=1

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
            --output_dir "$DIR/checkpoints-mi300" \
            --max_volumes 100 \
            --profile
            

echo "Job ended at $(date -R)"