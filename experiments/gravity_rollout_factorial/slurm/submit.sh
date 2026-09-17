#!/usr/bin/env bash
set -euo pipefail

: "${SGJEPA_PARTITION:?set the site-specific Slurm partition}"
: "${SGJEPA_QOS:?set the site-specific Slurm QoS}"
: "${SGJEPA_GRES:?set the site-specific GPU resource, for example gpu:1}"

sbatch \
  --partition="$SGJEPA_PARTITION" \
  --qos="$SGJEPA_QOS" \
  --gres="$SGJEPA_GRES" \
  "$@"
