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

export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1              
export NCCL_CROSS_NIC=1                
export FI_CXI_ATS=0                    

export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4 

echo ""
echo "================================================================"
echo "  NIfTI DINO Axial — Job configuration"
echo "----------------------------------------------------------------"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Nodes        : ${SLURM_JOB_NUM_NODES}  (${SLURM_JOB_NODELIST})"
echo "  GPUs/node    : ${GPUS_PER_NODE}  (total: ${TOTAL_GPUS})"
echo "  CPUs/task    : ${CPUS_PER_TASK:-32}"
echo "  OMP threads  : ${OMP_NUM_THREADS}"
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
            --output_dir "$DIR/checkpoints-$SLURM_JOB_ID"
            

echo "Job ended at $(date -R)"
