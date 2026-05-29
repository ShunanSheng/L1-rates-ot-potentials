#!/bin/bash
#
#SBATCH --account=stats
#SBATCH --job-name=ot_by_pair
#SBATCH --output=logs/ot_by_pair_%A_%a.out
#SBATCH --error=logs/ot_by_pair_%A_%a.err
#SBATCH -c 1
#SBATCH --time=0-08:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --array=0-19

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs results/raw results/summary results/figs

module load anaconda/3-2023.09

# Make BLAS/OpenMP libraries respect the single CPU requested above.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ot-exp
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Could not find conda or .venv. Create the ot-exp environment first." >&2
  exit 1
fi

python -m pip show otexp >/dev/null

PARAMS=(
  "2 256"
  "2 512"
  "2 1024"
  "2 2048"
  "2 4096"
  "3 256"
  "3 512"
  "3 1024"
  "3 2048"
  "3 4096"
  "4 256"
  "4 512"
  "4 1024"
  "4 2048"
  "4 4096"
  "10 256"
  "10 512"
  "10 1024"
  "10 2048"
  "10 4096"
)

read D N <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

echo "Running d=${D}, n=${N}, task=${SLURM_ARRAY_TASK_ID}"

python scripts/run_experiment.py \
  --d "$D" \
  --n_target "$N" \
  --B 100 \
  --n_source 4096 \
  --seed 2026 \
  --outdir results \
  --max_iter 180
