#!/bin/bash
# srun_mi300_bind.sh — wrapper for SLURM-native multi-rank-per-node pattern on MI300A.
#
# Pattern: srun --ntasks-per-node=4 (1 task = 1 APU = 1 GPU = 1 NIC Slingshot)
# Usage  : srun --ntasks-per-node=4 -- ./srun_mi300_bind.sh python train.py args
#
# Each srun task is an INDEPENDENT DDP rank managed directly by SLURM:
# proper per-task cgroup, NIC affinity (Slingshot), accounting. No torchrun layer.
#
# This wrapper:
#   1. Translates SLURM_PROCID/SLURM_LOCALID/SLURM_NTASKS → RANK/LOCAL_RANK/
#      WORLD_SIZE that PyTorch's dist.init_process_group(backend="nccl") reads.
#   2. Sets MASTER_ADDR (1st node from $SLURM_JOB_NODELIST) + MASTER_PORT.
#   3. Pins this task to its NUMA-local cores (phys + SMT siblings) per the
#      Adastra MI300A topology (doc: Adastra_MI300_4TasksWith24ThreadsAnd1GPU).
#   4. execs python with the args passed to this wrapper.
#
# Measured perf Adastra (batch=40, FlexiCT ViT-base, 200k steps):
#   1 node  × 4 APUs : 0.40 it/s =  64.0 samples/s (mono-node baseline)
#   2 nodes × 4 APUs : 0.34 it/s = 108.8 samples/s (85% linear scaling)
#   4 nodes × 4 APUs : 0.34 it/s = 217.6 samples/s (plateau flat, 100% 2n→4n)
# → Inter-node scaling stable post-1er hop = NIC affinity Slingshot OK.

set -eu

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29400}"

# NUMA mapping per APU (phys cores + SMT siblings). 96 phys cores / 4 APUs = 24 each.
# SMT siblings : core N has sibling N+96 on MI300A (SMT2).
#   APU 0 → cores  0-23 +  96-119 (NIC 0)
#   APU 1 → cores 24-47 + 120-143 (NIC 1)
#   APU 2 → cores 48-71 + 144-167 (NIC 2)
#   APU 3 → cores 72-95 + 168-191 (NIC 3)
NUMACTL=('0-23,96-119' '24-47,120-143' '48-71,144-167' '72-95,168-191')
CPU_SET="${NUMACTL[$((LOCAL_RANK % 4))]}"
export OMP_NUM_THREADS=24

echo "[srun_mi300_bind] rank=${RANK}/${WORLD_SIZE}  local=${LOCAL_RANK}  cpu=${CPU_SET}  master=${MASTER_ADDR}:${MASTER_PORT}" >&2

exec numactl --localalloc --physcpubind="${CPU_SET}" -- python -u "$@"
