#!/bin/bash
#
#SBATCH --account=stats
#SBATCH --job-name=by_d
#SBATCH --output=logs/by_d%A_%a.out
#SBATCH --error=logs/by_d_%A_%a.err
#SBATCH -c 1
#SBATCH --time=0-04:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --array=0-5

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

D_LIST=(2 3 4 6 8 10)
D=${D_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running new_d dimension d=${D}, task=${SLURM_ARRAY_TASK_ID}"

python scripts/run_experiment.py \
  --d "$D" \
  --n_target 100 300 1000 3000 10000\
  --B 100 \
  --n_source 10000 \
  --seed 2026 \
  --outdir results \
  --max_iter 1000
