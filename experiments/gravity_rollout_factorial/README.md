# Gravity and rollout factorial

This experiment separates physical-parameter conditioning from autoregressive
rollout training in SG-JEPA.

## Cells

| Cell | Gravity input | World-model objective |
| --- | --- | --- |
| `correct_onestep` | True normalized episode gravity | One-step prediction |
| `constant_onestep` | Normalized training mean, `0` | One-step prediction |
| `correct_rollout` | True normalized episode gravity | Five-step autoregressive rollout |
| `constant_rollout` | Normalized training mean, `0` | Five-step autoregressive rollout |

Each cell trains with world-model seed `42`. Each checkpoint gets five probe
and evaluation seeds, `42` through `46`.

The rollout cells replace the one-step objective. They set
`prediction_weight: 0.0` and `rollout_weight: 1.0`.

## Entrypoints

Set `SGJEPA_ROOT`, `SGJEPA_PYTHON`, `SGJEPA_DATASET`, and `SGJEPA_OUTPUT` in
your environment. Then run the scripts in this order:

1. `slurm/data.sbatch` generates and verifies the shared Square dataset.
2. `slurm/world_model.sbatch` trains one world model for each cell.
3. `slurm/probe.sbatch` fits five probes for each cell.
4. `slurm/evaluate.sbatch` evaluates five seeds for each cell.

The scripts use generic environment variables. They do not assume a specific
cluster, account, user, or storage path.

Use `slurm/submit.sh` to provide the partition, QoS, and GPU resource required
by your site.
