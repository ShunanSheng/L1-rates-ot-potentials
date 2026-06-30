#!/bin/bash
#
# Goodness-of-fit Slurm job.
#
# Override any GOF_* value at submission time, for example:
#   sbatch --export=ALL,GOF_MODE=power,GOF_NULL=uniform_ball,GOF_ALT=location_shift,GOF_LEVEL=0.05,GOF_CHUNK_ID=0,GOF_NUM_CHUNKS=10 jobs/slurm_gof.sh
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

mkdir -p logs results_gof/references results_gof/raw results_gof/summary results_gof/figs

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
GOF_MODE="${GOF_MODE:-calibration}"
GOF_NULL="${GOF_NULL:-uniform_ball}"
GOF_ALT="${GOF_ALT:-location_shift}"
GOF_LEVEL="${GOF_LEVEL:-}"
GOF_B_CAL="${GOF_B_CAL:-1000}" # Monte Carlo runs for estimating the critical value
GOF_N_SIZE="${GOF_N_SIZE:-1000}" 
GOF_N_ALT="${GOF_N_ALT:-1000}"
GOF_N_SOLVE="${GOF_N_SOLVE:-5000}"
GOF_N_EVAL_SOURCE="${GOF_N_EVAL_SOURCE:-5000}"
GOF_SEED="${GOF_SEED:-2026}"
GOF_MAX_ITER="${GOF_MAX_ITER:-2000}"
GOF_CHUNK_SIZE="${GOF_CHUNK_SIZE:-2500}"
GOF_CHUNK_ID="${GOF_CHUNK_ID:-}"
GOF_NUM_CHUNKS="${GOF_NUM_CHUNKS:-}"
GOF_OUTDIR="${GOF_OUTDIR:-results_gof}"

echo "Python executable: $PYTHON"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -m pip show otexp >/dev/null
"$PYTHON" -c "import otexp; print('otexp package:', otexp.__file__)"

echo "Running GOF: mode=${GOF_MODE}, null=${GOF_NULL}, alt=${GOF_ALT}, level=${GOF_LEVEL}, d=${GOF_D}, n=${GOF_N}"

EXTRA_ARGS=()
if [ -n "$GOF_LEVEL" ]; then
  EXTRA_ARGS+=(--level "$GOF_LEVEL")
fi
if [ -n "$GOF_CHUNK_ID" ]; then
  EXTRA_ARGS+=(--chunk_id "$GOF_CHUNK_ID")
fi
if [ -n "$GOF_NUM_CHUNKS" ]; then
  EXTRA_ARGS+=(--num_chunks "$GOF_NUM_CHUNKS")
fi

"$PYTHON" scripts/run_gof.py \
  --mode "$GOF_MODE" \
  --null "$GOF_NULL" \
  --alt "$GOF_ALT" \
  --d "$GOF_D" \
  --n "$GOF_N" \
  --B_cal "$GOF_B_CAL" \
  --N_size "$GOF_N_SIZE" \
  --N_alt "$GOF_N_ALT" \
  --n_solve "$GOF_N_SOLVE" \
  --n_eval_source "$GOF_N_EVAL_SOURCE" \
  --seed "$GOF_SEED" \
  --max_iter "$GOF_MAX_ITER" \
  --chunk_size "$GOF_CHUNK_SIZE" \
  --outdir "$GOF_OUTDIR" \
  "${EXTRA_ARGS[@]}"
