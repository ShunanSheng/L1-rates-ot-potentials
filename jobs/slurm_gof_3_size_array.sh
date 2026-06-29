#!/bin/bash
#
# Stage 3: empirical size array.
#
# Default array covers:
#   3 null distributions x GOF_NUM_CHUNKS=10 chunks = 30 tasks.
#
# Run after calibration aggregation has produced critical_values.csv.
#
# Submit:
#   sbatch jobs/slurm_gof_3_size_array.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_size
#SBATCH --output=logs/ot_gof_size_%A_%a.out
#SBATCH --error=logs/ot_gof_size_%A_%a.err
#SBATCH -c 1
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=0-29

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

NULLS=(uniform_ball truncated_gaussian truncated_elliptical_t)
GOF_NUM_CHUNKS="${GOF_NUM_CHUNKS:-10}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

NULL_INDEX=$((TASK_ID / GOF_NUM_CHUNKS))
CHUNK_ID=$((TASK_ID % GOF_NUM_CHUNKS))

if [ "$NULL_INDEX" -ge "${#NULLS[@]}" ]; then
  echo "Task ${TASK_ID} exceeds configured null/chunk grid; exiting."
  exit 0
fi

export GOF_MODE="size"
export GOF_NULL="${NULLS[$NULL_INDEX]}"
export GOF_CHUNK_ID="$CHUNK_ID"
export GOF_NUM_CHUNKS

echo "Size stage: null=${GOF_NULL}, chunk=${GOF_CHUNK_ID}/${GOF_NUM_CHUNKS}"
bash jobs/slurm_gof.sh
