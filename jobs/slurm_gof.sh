#!/bin/bash
#
# Goodness-of-fit Slurm job.
#
# Override any GOF_* value at submission time, for example:
#   sbatch --export=ALL,GOF_N=64,GOF_B=200,GOF_N_EVAL=200 jobs/slurm_gof.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof
#SBATCH --output=logs/ot_gof_%j.out
#SBATCH --error=logs/ot_gof_%j.err
#SBATCH -c 1
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=16gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs results_gof/raw results_gof/summary

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

PYTHON="${PYTHON:-$(command -v python)}"

GOF_D="${GOF_D:-3}"
GOF_N="${GOF_N:-1000}"
GOF_B="${GOF_B:-1000}"
GOF_N_EVAL="${GOF_N_EVAL:-100}"
GOF_N_SOURCE="${GOF_N_SOURCE:-5000}"
GOF_SEED="${GOF_SEED:-2026}"
GOF_MAX_ITER="${GOF_MAX_ITER:-200}"
GOF_CHUNK_SIZE="${GOF_CHUNK_SIZE:-2500}"
GOF_OUTDIR="${GOF_OUTDIR:-results_gof}"

echo "Python executable: $PYTHON"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -m pip show otexp >/dev/null
"$PYTHON" -c "import otexp; print('otexp package:', otexp.__file__)"

echo "Running GOF: d=${GOF_D}, n=${GOF_N}, B=${GOF_B}, n_eval=${GOF_N_EVAL}, n_source=${GOF_N_SOURCE}"

"$PYTHON" scripts/run_gof.py \
  --d "$GOF_D" \
  --n "$GOF_N" \
  --B "$GOF_B" \
  --n_eval "$GOF_N_EVAL" \
  --n_source "$GOF_N_SOURCE" \
  --seed "$GOF_SEED" \
  --max_iter "$GOF_MAX_ITER" \
  --chunk_size "$GOF_CHUNK_SIZE" \
  --location_thetas 0.005 0.01 0.02 0.05 0.1 \
  --scale_thetas 0.005 0.01 0.02 0.05 0.1 \
  --mixture_thetas 0.005 0.01 0.02 0.05 0.1 \
  --outdir "$GOF_OUTDIR"
