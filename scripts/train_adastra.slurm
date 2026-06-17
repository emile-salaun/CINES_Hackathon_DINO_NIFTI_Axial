#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=dci
#SBATCH --constraint=MI250
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test
#SBATCH --output=train_test_300.log
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --reservation=hackaton


source /lus/work/CT3/cad17796/SHARED/.venv8/bin/activate.poverlay
export VENV_DIR="/lus/work/CT3/cad17796/SHARED/.venv8"

PYTHON="${VENV_DIR}/bin/python"

PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PKG="${VENV_DIR}/lib/python${PYVER}/site-packages"

mkdir -p "$SITE_PKG"
FLEXICT_ABS="$(realpath ./models_pretrained/flexiCT/FlexiCT)"

echo "${FLEXICT_ABS}" > "${SITE_PKG}/flexict.pth"
echo "Added ${FLEXICT_ABS} → ${SITE_PKG}/flexict.pth"

module use /lus/scratch/BCINES/dci/amblard/Test_installation_rocm/Configuration.spack-user-develop/modules/tcl/linux-rhel9-zen3
module use /lus/scratch/BCINES/dci/amblard/Test_installation_rocm/Configuration.spack-user-develop/modules/tcl/linux-rhel9-x86_64
module load gcc/14.2.1/zen3/hipblaslt/develop-3bft 

module list 

OUTPUT_DIR=./checkpoints/smoke_test_train
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=4 train.py \
        --config configs/phase1.yaml \
        --output_dir $OUTPUT_DIR \
        --max_volumes 10
