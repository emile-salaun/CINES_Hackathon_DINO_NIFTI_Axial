#!/bin/bash
#==============================================================================
# bench_mi300_b40_4gpu.slurm  —  STANDALONE reproduction
#
# Reproduit le benchmark mono-node MI300A originel (job 5075686, 2026-06-18) :
#     1 nœud MI300A, 4 APUs, batch=40, mi300_numa binding
#     → 0.40 it/s warm = 64 samples/s = ETA 10M images ~1.81 jours
#
# Fichier UNIQUE (pas de wrapper externe) : la binding NUMA per-rank est
# écrite à /tmp au démarrage du job pour que torchrun --no-python l'exec.
#
# Usage :
#     cd $HOME      # ⚠️ submit depuis Lustre, PAS /tmp (= local au nœud, log invisible)
#     sbatch /chemin/vers/bench_mi300_b40_4gpu.slurm
#
# Le log run_<JOBID>.out apparaît dans le dossier de soumission (cwd au moment
# du sbatch). Doit être un FS partagé (Lustre $HOME/$WORK/$SCRATCH) sinon le
# log n'est visible que sur le nœud compute.
#
# Pour un entraînement Phase 1 complet (plusieurs jours) : bump --time à 24h+
# et retirer --max_volumes 100 dans la commande train.py en bas.
#==============================================================================

#SBATCH --account=hpe
#SBATCH --reservation=hackaton
#SBATCH --job-name=mi300_b40_4gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --constraint=MI300
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.out
#SBATCH --export=ALL

set -eu

# ---- Paths du projet (à adapter si le repo est déplacé) ---------------------
# Default : repo clone dans $HOME. Override via env :
#   REPO_DIR=/chemin/vers/repo sbatch bench_mi300_b40_4gpu.slurm
DIR="${REPO_DIR:-$HOME/CINES_Hackathon_DINO_NIFTI_Axial/nifti_dino_axial}"
CONFIG="$DIR/configs/phase1_b40.yaml"
PRETRAINED="$DIR/models_pretrained/flexiCT/2D_final_model.pth"
OUTPUT_DIR="$DIR/checkpoints/bench_mi300_b40_4gpu_${SLURM_JOB_ID}"

# ---- venv poverlay partagée Adastra ----------------------------------------
source /lus/work/CT3/cad17796/SHARED/.venv8/bin/activate.poverlay

# ---- Patch hipBLASLt (requis MI300, cf README projet) ----------------------
HIPBLASLT_DIR=/lus/work/CT3/cad17796/SHARED/spack-install-hipblaslt-patch/linux-zen3/hipblaslt-develop-24ro
export LD_LIBRARY_PATH=${HIPBLASLT_DIR}/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=${HIPBLASLT_DIR}/lib64/libhipblaslt.so.1

# ---- MIOpen cache (writable per-job sur $SCRATCHDIR) -----------------------
MIOPEN_BASE="${SCRATCHDIR:-$HOME}/.miopen/${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="${MIOPEN_BASE}/db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_BASE}/cache"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

# ---- ROCm / HSA / NCCL Slingshot -------------------------------------------
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_XNACK=1

# ---- Wrapper de binding NUMA per-rank (inliné via heredoc) -----------------
# Mapping doc Adastra Adastra_MI300_4TasksWith24ThreadsAnd1GPU :
#   APU 0 → cores 0-23, APU 1 → 24-47, APU 2 → 48-71, APU 3 → 72-95
# torchrun --no-python exec ce wrapper par rank ; $LOCAL_RANK est set par
# torchrun. Chaque rank est épinglé sur les 24 cores de son APU NUMA-local.
WRAPPER="/tmp/bind_mi300_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" << 'BIND_EOF'
#!/bin/bash
NUMACTL=('0-23' '24-47' '48-71' '72-95')
CPU_SET="${NUMACTL[$((LOCAL_RANK % 4))]}"
export OMP_NUM_THREADS=24
exec numactl --localalloc --physcpubind="${CPU_SET}" -- python -u "$@"
BIND_EOF
chmod +x "$WRAPPER"

# ---- Launch : 1 SLURM task → torchrun spawn 4 ranks bindés per-APU ---------
echo "Job $SLURM_JOB_ID started at $(date -R) on $(hostname)"

srun --ntasks-per-node=1 --gpus-per-task=4 \
     --cpu-bind=none --mem-bind=none \
    -- torchrun \
        --nnodes=1 \
        --nproc_per_node=4 \
        --rdzv-id="$SLURM_JOB_ID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$(hostname):29400" \
        --max-restarts=0 \
        --no-python \
        -- "$WRAPPER" "$DIR/train.py" \
            --config "$CONFIG" \
            --pretrained "$PRETRAINED" \
            --output_dir "$OUTPUT_DIR" \
            --max_volumes 100

echo "Job $SLURM_JOB_ID ended at $(date -R)"
rm -f "$WRAPPER"
