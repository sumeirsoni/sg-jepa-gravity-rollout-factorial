#!/usr/bin/env bash
#SBATCH --job-name=sgj-factorial-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SGJEPA_ROOT:?set SGJEPA_ROOT to the SG-JEPA checkout}"
: "${SGJEPA_PYTHON:?set SGJEPA_PYTHON to the paper environment}"
: "${SGJEPA_DATASET:?set SGJEPA_DATASET to the shared Square dataset path}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$SGJEPA_ROOT"
if [[ -e "$SGJEPA_DATASET" ]]; then
  echo "dataset path already exists: $SGJEPA_DATASET" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$SGJEPA_DATASET")"
"$SGJEPA_PYTHON" -m data_generation.generate \
  --recipe data_generation/recipes/main_text.yaml \
  --task square \
  --split all \
  --output "$SGJEPA_DATASET"
"$SGJEPA_PYTHON" scripts/verify_examples.py data \
  --path "$SGJEPA_DATASET" \
  --task square \
  --episodes 48000 \
  --train-episodes 8000 \
  --test-episodes 40000
