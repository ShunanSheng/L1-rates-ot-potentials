#!/bin/bash
#
# Complete GOF smoke-test Slurm job.
#
# This job tests the GOF code path end-to-end with tiny settings and writes to
# a disposable smoke output directory.
#
# Submit:
#   sbatch jobs/slurm_test_gof.sh
#
# Optional overrides:
#   sbatch --export=ALL,GOF_TEST_OUTDIR=results_gof_smoke jobs/slurm_test_gof.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_smoke
#SBATCH --output=logs/ot_gof_smoke_%j.out
#SBATCH --error=logs/ot_gof_smoke_%j.err
#SBATCH -c 1
#SBATCH --time=0-02:00
#SBATCH --mem-per-cpu=8gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

GOF_TEST_OUTDIR="${GOF_TEST_OUTDIR:-results_gof_smoke}"
PYTHON="${PYTHON:-}"

mkdir -p logs "$GOF_TEST_OUTDIR"

module load anaconda/3-2023.09

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "ot-exp"; then
    conda activate ot-exp
  else
    conda activate /burg-archive/home/ss6574/.conda/envs/ot-exp
  fi
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Could not find conda or .venv. Create the ot-exp environment first." >&2
  exit 1
fi

if [ -z "$PYTHON" ]; then
  PYTHON="$(command -v python)"
fi

echo "Python executable: $PYTHON"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -m pip show otexp >/dev/null
"$PYTHON" -c "import numpy, pandas, scipy, matplotlib; import otexp; print('otexp package:', otexp.__file__)"

echo
echo "== Tiny GOF end-to-end run =="
"$PYTHON" scripts/run_gof.py \
  --mode all \
  --null uniform_ball \
  --d 3 \
  --n 4 \
  --B_cal 2 \
  --N_size 2 \
  --N_alt 1 \
  --n_solve 16 \
  --n_eval_source 16 \
  --seed 2033 \
  --max_iter 5 \
  --chunk_size 8 \
  --outdir "$GOF_TEST_OUTDIR" \
  --overwrite

echo
echo "== Aggregate GOF smoke outputs =="
"$PYTHON" scripts/aggregate_gof.py \
  --outdir "$GOF_TEST_OUTDIR" \
  --alpha 0.05

echo
echo "== Plot GOF smoke outputs =="
"$PYTHON" scripts/plot_gof.py \
  --outdir "$GOF_TEST_OUTDIR"

echo
echo "== Smoke output files =="
find "$GOF_TEST_OUTDIR" -type f | sort

echo
echo "GOF smoke Slurm test completed successfully."
