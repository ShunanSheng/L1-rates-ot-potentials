#!/bin/bash
#
#SBATCH --account=stats
#SBATCH --job-name=ot_test
#SBATCH --output=logs/ot_test_%j.out
#SBATCH --error=logs/ot_test_%j.err
#SBATCH -c 1
#SBATCH --time=0-01:00
#SBATCH --mem-per-cpu=5gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results_test/raw results_test/summary results_test/figs

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

python scripts/run_experiment.py \
  --d 2 \
  --n_target 10 30 100 300 1000 3000\
  --B 100 \
  --n_source 5000 \
  --seed 2026 \
  --outdir results_test \
  --max_iter 100

python scripts/aggregate.py \
  --raw_dir results_test/raw \
  --out_path results_test/summary/aggregated.csv

python scripts/plot_results.py \
  --summary_path results_test/summary/aggregated.csv \
  --figdir results_test/figs
