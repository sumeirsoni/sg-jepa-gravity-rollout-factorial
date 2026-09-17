#!/usr/bin/env bash
#SBATCH --job-name=sgj-factorial-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19%8
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

: "${SGJEPA_ROOT:?set SGJEPA_ROOT to the SG-JEPA checkout}"
: "${SGJEPA_PYTHON:?set SGJEPA_PYTHON to the paper environment}"
: "${SGJEPA_DATASET:?set SGJEPA_DATASET to the shared Square dataset path}"
: "${SGJEPA_OUTPUT:?set SGJEPA_OUTPUT to the experiment output path}"

conditions=(correct_onestep correct_rollout constant_onestep constant_rollout)
seeds=(42 43 44 45 46)
train_seed=42
condition_index=$((SLURM_ARRAY_TASK_ID / 5))
seed_index=$((SLURM_ARRAY_TASK_ID % 5))
condition="${conditions[$condition_index]}"
seed="${seeds[$seed_index]}"

case "$condition" in
  correct_onestep) config="configs/train/square_factorial_correct_onestep.yaml" ;;
  correct_rollout) config="configs/train/square_factorial_correct_rollout.yaml" ;;
  constant_onestep) config="configs/train/square_factorial_constant_onestep.yaml" ;;
  constant_rollout) config="configs/train/square_factorial_constant_rollout.yaml" ;;
  *) echo "unknown condition: $condition" >&2; exit 2 ;;
esac

world_model_dir="$SGJEPA_OUTPUT/world_models/$condition/seed_$train_seed"
probe_dir="$SGJEPA_OUTPUT/probes/$condition/seed_$seed"
evaluation_dir="$SGJEPA_OUTPUT/evaluations/$condition/seed_$seed"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$SGJEPA_ROOT"
test -f "$world_model_dir/checkpoint.pt"
test -f "$probe_dir/probe_weights.pt"
test ! -e "$evaluation_dir"
"$SGJEPA_PYTHON" evaluate_prediction.py \
  --task square \
  --method "$condition-seed-$seed" \
  --config "$config" \
  --checkpoint "$world_model_dir/checkpoint.pt" \
  --probe "$probe_dir/probe_weights.pt" \
  --data "$SGJEPA_DATASET" \
  --normalization-stats "$world_model_dir/normalization_stats.json" \
  --episode-manifest data_generation/manifests/planar_test_gstrat200_seed42.json \
  --output-dir "$evaluation_dir" \
  --device cuda \
  --horizon 44 \
  --max-episodes 5000
"$SGJEPA_PYTHON" scripts/verify_examples.py report \
  --path "$evaluation_dir/evaluation.json"
