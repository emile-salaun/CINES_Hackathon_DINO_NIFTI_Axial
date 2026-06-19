#!/bin/bash

#==================================================================
#  NIfTI DINO Axial — job body (sourced by sbatch via launch.sh)
#==================================================================

export DIR="${DIR:-$SCRATCH/hackathon-juin/gh/CINES_Hackathon_DINO_NIFTI_Axial/nifti_dino_axial}"

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

# BIND_STRATEGY (env var, optional) — switch entre patterns de launch.
#   (vide) | none   -> default DEV (1 SLURM task + torchrun, hsn0 single NIC)
#   mi300_srun4     -> HPE multi-node SLURM-native (4 tasks/node, 4 NICs + IB)
#                       Mesuré sur Adastra (FlexiCT ViT-base, batch=40, 200k steps):
#                         4 nodes x 4 APUs = 217.6 img/s (85% scaling lineaire)
#                       cf. scripts/srun_mi300_bind.sh pour le binding NUMA/NIC.
case "${BIND_STRATEGY:-none}" in
    none|"")
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
        ;;

    mi300_srun4)
        export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
        unset NCCL_IB_DISABLE
        export HSA_FORCE_FINE_GRAIN_PCIE=1
        export HSA_XNACK=1
        MIOPEN_BASE="${SCRATCHDIR:-$HOME}/.miopen/${SLURM_JOB_ID:-local}"
        export MIOPEN_USER_DB_PATH="${MIOPEN_BASE}/db"
        export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_BASE}/cache"
        mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

        SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "$DIR")/scripts}"
        BIND_WRAPPER="${SCRIPT_DIR}/srun_mi300_bind.sh"
        CFG_PATH="$DIR/configs/phase1${RUN_TAG:+_${RUN_TAG}}.yaml"
        OUT_DIR="$DIR/checkpoints/${CONSTRAINT,,}_${BIND_STRATEGY}${RUN_TAG:+_${RUN_TAG}}_${SLURM_JOB_NUM_NODES}n_${SLURM_JOB_ID}"
        echo "BIND_STRATEGY=mi300_srun4  wrapper=${BIND_WRAPPER}  config=${CFG_PATH}  output=${OUT_DIR}"
        srun --cpu-bind=none --mem-bind=none \
            -- "${BIND_WRAPPER}" "$DIR/train.py" \
                --config "${CFG_PATH}" \
                --pretrained "$DIR/models_pretrained/flexiCT/2D_final_model.pth" \
                --output_dir "${OUT_DIR}"
        ;;

    *)
        echo "ERROR: BIND_STRATEGY='${BIND_STRATEGY}' inconnu. Valeurs: none (defaut), mi300_srun4" >&2
        exit 1
        ;;
esac

echo "Job ended at $(date -R)"
