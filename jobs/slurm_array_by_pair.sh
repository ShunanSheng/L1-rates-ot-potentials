#!/bin/bash
#
# One array task runs one (dimension, n_target) pair.
#
#SBATCH --account=stats
#SBATCH --job-name=ot_pair
#SBATCH --output=logs/ot_pair_%A_%a.out
#SBATCH --error=logs/ot_pair_%A_%a.err
#SBATCH -c 1
#SBATCH --time=0-23:00
#SBATCH --mem-per-cpu=16gb
#SBATCH --array=0-29

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs results/raw results/metadata results/summary results/figs

module load anaconda/3-2023.09

# Make BLAS/OpenMP libraries respect the single CPU requested above.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /burg-archive/home/ss6574/.conda/envs/ot-exp
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Could not find conda or .venv. Create the ot-exp environment first." >&2
  exit 1
fi

PYTHON=/burg-archive/home/ss6574/.conda/envs/ot-exp/bin/python

D_LIST=(2 3 4 6 8 10)
N_LIST=(100 300 1000 3000 10000)
B=100
N_PER_D=${#N_LIST[@]}

TASK=${SLURM_ARRAY_TASK_ID}
N_INDEX=$((TASK % N_PER_D))
D_INDEX=$((TASK / N_PER_D))

D=${D_LIST[$D_INDEX]}
N=${N_LIST[$N_INDEX]}

echo "Python executable: $PYTHON"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -c "import otexp; print('otexp package:', otexp.__file__)"
echo "Running pair task=${TASK}: d=${D}, n=${N}, B=${B}"

"$PYTHON" scripts/run_experiment.py \
  --d "$D" \
  --n_target "$N" \
  --B "$B" \
  --n_source 10000 \
  --seed 2026 \
  --outdir results \
  --max_iter 1000
