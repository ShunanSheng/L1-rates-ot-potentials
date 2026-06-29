#!/bin/bash
#
# Stage 4: empirical power array.
#
# Default array covers:
#   3 nulls x 3 alternatives x 6 levels x GOF_NUM_CHUNKS=10 chunks = 540 tasks.
#
# Run after calibration aggregation has produced critical_values.csv.
#
# Submit:
#   sbatch jobs/slurm_gof_4_power_array.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_power
#SBATCH --output=logs/ot_gof_power_%A_%a.out
#SBATCH --error=logs/ot_gof_power_%A_%a.err
#SBATCH -c 1
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=0-539

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

NULLS=(uniform_ball truncated_gaussian truncated_elliptical_t)
ALTS=(location_shift scale mixture_contamination)
LOCATION_LEVELS=(0.05 0.10 0.15 0.20 0.25 0.30)
SCALE_LEVELS=(1.025 1.05 1.075 1.10 1.15 1.20)
MIXTURE_LEVELS=(0.01 0.02 0.05 0.10 0.15 0.20)

GOF_NUM_CHUNKS="${GOF_NUM_CHUNKS:-10}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

CHUNK_ID=$((TASK_ID % GOF_NUM_CHUNKS))
REST=$((TASK_ID / GOF_NUM_CHUNKS))
LEVEL_INDEX=$((REST % 6))
REST=$((REST / 6))
ALT_INDEX=$((REST % 3))
NULL_INDEX=$((REST / 3))

if [ "$NULL_INDEX" -ge "${#NULLS[@]}" ]; then
  echo "Task ${TASK_ID} exceeds configured null/alt/level/chunk grid; exiting."
  exit 0
fi

ALT="${ALTS[$ALT_INDEX]}"
if [ "$ALT" = "location_shift" ]; then
  LEVEL="${LOCATION_LEVELS[$LEVEL_INDEX]}"
elif [ "$ALT" = "scale" ]; then
  LEVEL="${SCALE_LEVELS[$LEVEL_INDEX]}"
else
  LEVEL="${MIXTURE_LEVELS[$LEVEL_INDEX]}"
fi

export GOF_MODE="power"
export GOF_NULL="${NULLS[$NULL_INDEX]}"
export GOF_ALT="$ALT"
export GOF_LEVEL="$LEVEL"
export GOF_CHUNK_ID="$CHUNK_ID"
export GOF_NUM_CHUNKS

echo "Power stage: null=${GOF_NULL}, alt=${GOF_ALT}, level=${GOF_LEVEL}, chunk=${GOF_CHUNK_ID}/${GOF_NUM_CHUNKS}"
bash jobs/slurm_gof.sh
