#!/bin/bash
#
#SBATCH --account=stats
#SBATCH --job-name=ot_post
#SBATCH --output=logs/ot_post_%j.out
#SBATCH --error=logs/ot_post_%j.err
#SBATCH -c 1
#SBATCH --time=0-01:00
#SBATCH --mem-per-cpu=5gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/summary results/figs

module load anaconda/3-2023.09

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

python scripts/aggregate.py \
  --raw_dir results/raw \
  --out_path results/summary/aggregated.csv

python scripts/plot_results.py \
  --summary_path results/summary/aggregated.csv \
  --raw_dir results/raw \
  --figdir results/figs

# python scripts/aggregate.py \
#   --raw_dir results_test/raw \
#   --out_path results_test/summary/aggregated.csv

# python scripts/plot_results_test.py \
#   --summary_path results_test/summary/aggregated.csv \
#   --raw_dir results_test/raw \
#   --figdir results_test/figs
