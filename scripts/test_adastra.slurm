#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=dci
#SBATCH --constraint=MI300
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test
#SBATCH --output=t_test.log
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


OUTPUT_DIR=./checkpoints/smoke_test
mkdir -p "$OUTPUT_DIR"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    "./smoke_test.py" \
    --nifti_dir   "$WORK_cad17796_SHARED/merlinabdominalctdataset/merlin_data" \
    --max_volumes 20 \
    --pretrained  "./models_pretrained/flexiCT/2D_final_model.pth" \
    --output_dir  "$OUTPUT_DIR"