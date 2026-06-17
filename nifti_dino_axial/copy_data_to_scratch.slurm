#!/bin/bash
# Copie les données depuis $ALL_CCFRSTORE vers $SCRATCH sur un nœud prepost.
# Soumission : sbatch nifti_dino_axial/copy_data_to_scratch.slurm
#
# À faire UNE SEULE FOIS avant de lancer l'entraînement.
# Si interrompu, rsync reprend là où il s'est arrêté (--checksum).

#SBATCH --job-name=copy_ct_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=prepost
#SBATCH -A ssb@a100
#SBATCH --output=logs/copy_data_%j.out

set -eo pipefail

SRC=$ALL_CCFRSTORE/merlinabdominalctdataset/merlin_data/
DST=$SCRATCH/ct_nifti

mkdir -p "$DST" logs

echo "Copie $SRC → $DST"
echo "Début : $(date)"

# --checksum  : compare les checksums, pas juste mtime (plus fiable pour reprise)
# --no-whole-file : économise la RAM (utile pour gros fichiers)
rsync -av --progress "$SRC/" "$DST/"

echo "Fin : $(date)"
echo "Taille finale :"
du -sh "$DST"
