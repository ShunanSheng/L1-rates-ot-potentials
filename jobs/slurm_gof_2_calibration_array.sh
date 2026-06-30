#!/bin/bash
#
# Stage 2: calibration array.
#
# Default array covers:
#   3 null distributions x GOF_NUM_CHUNKS=10 chunks = 30 tasks.
#
# Submit:
#   sbatch jobs/slurm_gof_2_calibration_array.sh
#
# If changing GOF_NUM_CHUNKS, also override the Slurm array range:
#   sbatch --array=0-14 --export=ALL,GOF_NUM_CHUNKS=5 jobs/slurm_gof_2_calibration_array.sh
#
# To intentionally replace existing calibration chunks, set GOF_OVERWRITE=1:
#   sbatch --export=ALL,GOF_MAX_ITER=2000,GOF_OVERWRITE=1 jobs/slurm_gof_2_calibration_array.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_cal
#SBATCH --output=logs/ot_gof_cal_%A_%a.out
#SBATCH --error=logs/ot_gof_cal_%A_%a.err
#SBATCH -c 1
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=0-29

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

NULLS=(uniform_ball truncated_gaussian truncated_elliptical_t)
GOF_NUM_CHUNKS="${GOF_NUM_CHUNKS:-10}"
GOF_OVERWRITE="${GOF_OVERWRITE:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

NULL_INDEX=$((TASK_ID / GOF_NUM_CHUNKS))
CHUNK_ID=$((TASK_ID % GOF_NUM_CHUNKS))

if [ "$NULL_INDEX" -ge "${#NULLS[@]}" ]; then
  echo "Task ${TASK_ID} exceeds configured null/chunk grid; exiting."
  exit 0
fi

export GOF_MODE="calibration"
export GOF_NULL="${NULLS[$NULL_INDEX]}"
export GOF_CHUNK_ID="$CHUNK_ID"
export GOF_NUM_CHUNKS
export GOF_OVERWRITE

echo "Calibration stage: null=${GOF_NULL}, chunk=${GOF_CHUNK_ID}/${GOF_NUM_CHUNKS}, overwrite=${GOF_OVERWRITE}"
bash jobs/slurm_gof.sh
