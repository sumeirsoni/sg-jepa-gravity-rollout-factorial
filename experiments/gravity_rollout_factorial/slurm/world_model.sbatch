#!/usr/bin/env bash
#SBATCH --job-name=sgj-factorial-wm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

: "${SGJEPA_ROOT:?set SGJEPA_ROOT to the SG-JEPA checkout}"
: "${SGJEPA_PYTHON:?set SGJEPA_PYTHON to the paper environment}"
: "${SGJEPA_DATASET:?set SGJEPA_DATASET to the shared Square dataset path}"
: "${SGJEPA_OUTPUT:?set SGJEPA_OUTPUT to the experiment output path}"

conditions=(correct_onestep correct_rollout constant_onestep constant_rollout)
train_seed=42
condition_index=$SLURM_ARRAY_TASK_ID
condition="${conditions[$condition_index]}"
seed="$train_seed"

case "$condition" in
  correct_onestep) config="configs/train/square_factorial_correct_onestep.yaml" ;;
  correct_rollout) config="configs/train/square_factorial_correct_rollout.yaml" ;;
  constant_onestep) config="configs/train/square_factorial_constant_onestep.yaml" ;;
  constant_rollout) config="configs/train/square_factorial_constant_rollout.yaml" ;;
  *) echo "unknown condition: $condition" >&2; exit 2 ;;
esac

output_dir="$SGJEPA_OUTPUT/world_models/$condition/seed_$seed"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$SGJEPA_ROOT"
mkdir -p "$output_dir"
"$SGJEPA_PYTHON" train.py \
  --config "$config" \
  --data "$SGJEPA_DATASET" \
  --output-dir "$output_dir" \
  --device cuda \
  --seed "$seed"
"$SGJEPA_PYTHON" scripts/verify_examples.py world-model \
  --output "$output_dir" \
  --config "$config" \
  --seed "$seed" \
  --completed-epochs 20
