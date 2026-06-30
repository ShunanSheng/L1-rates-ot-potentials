#!/bin/bash
#
# GOF max-iteration pilot Slurm job.
#
# This job runs a small calibration-only GOF pilot over several max_iter values
# and reports the OT solver success rate. Use it to choose GOF_MAX_ITER before
# launching the full staged GOF experiment.
#
# Submit:
#   sbatch jobs/slurm_test_gof.sh
#
# Optional overrides:
#   sbatch --export=ALL,GOF_TEST_MAX_ITERS="500 1000 2000",GOF_TEST_B_CAL=30 jobs/slurm_test_gof.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_maxiter
#SBATCH --output=logs/ot_gof_maxiter_%j.out
#SBATCH --error=logs/ot_gof_maxiter_%j.err
#SBATCH -c 1
#SBATCH --time=1-00:00
#SBATCH --mem-per-cpu=16gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

GOF_TEST_OUTDIR="${GOF_TEST_OUTDIR:-results_gof_maxiter_test}"
GOF_TEST_MAX_ITERS="${GOF_TEST_MAX_ITERS:-500 1000 2000}"
GOF_TEST_NULL="${GOF_TEST_NULL:-truncated_gaussian}"
GOF_TEST_D="${GOF_TEST_D:-3}"
GOF_TEST_N="${GOF_TEST_N:-1000}"
GOF_TEST_B_CAL="${GOF_TEST_B_CAL:-20}"
GOF_TEST_N_SOLVE="${GOF_TEST_N_SOLVE:-5000}"
GOF_TEST_N_EVAL_SOURCE="${GOF_TEST_N_EVAL_SOURCE:-5000}"
GOF_TEST_SEED="${GOF_TEST_SEED:-2026}"
GOF_TEST_CHUNK_SIZE="${GOF_TEST_CHUNK_SIZE:-2500}"
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
echo "== GOF max_iter pilot configuration =="
echo "outdir=${GOF_TEST_OUTDIR}"
echo "max_iters=${GOF_TEST_MAX_ITERS}"
echo "null=${GOF_TEST_NULL}"
echo "n=${GOF_TEST_N}, B_cal=${GOF_TEST_B_CAL}"
echo "n_solve=${GOF_TEST_N_SOLVE}, n_eval_source=${GOF_TEST_N_EVAL_SOURCE}"
echo "chunk_size=${GOF_TEST_CHUNK_SIZE}, seed=${GOF_TEST_SEED}"

for MAX_ITER in $GOF_TEST_MAX_ITERS; do
  RUN_OUTDIR="${GOF_TEST_OUTDIR}/max_iter=${MAX_ITER}"

  echo
  echo "== Calibration pilot: max_iter=${MAX_ITER} =="
  "$PYTHON" scripts/run_gof.py \
    --mode calibration \
    --null "$GOF_TEST_NULL" \
    --d "$GOF_TEST_D" \
    --n "$GOF_TEST_N" \
    --B_cal "$GOF_TEST_B_CAL" \
    --N_size 1 \
    --N_alt 1 \
    --n_solve "$GOF_TEST_N_SOLVE" \
    --n_eval_source "$GOF_TEST_N_EVAL_SOURCE" \
    --seed "$GOF_TEST_SEED" \
    --max_iter "$MAX_ITER" \
    --chunk_size "$GOF_TEST_CHUNK_SIZE" \
    --outdir "$RUN_OUTDIR" \
    --overwrite

  echo
  echo "== Aggregate pilot output: max_iter=${MAX_ITER} =="
  "$PYTHON" scripts/aggregate_gof.py \
    --outdir "$RUN_OUTDIR" \
    --alpha 0.05

  echo
  echo "== OT success diagnostics: max_iter=${MAX_ITER} =="
  "$PYTHON" -c "import pandas as pd, pathlib; root=pathlib.Path('$RUN_OUTDIR'); files=sorted((root/'raw'/'calibration').glob('**/chunk=*.csv')); df=pd.concat([pd.read_csv(p) for p in files], ignore_index=True); ot=df[df['statistic'].isin(['potential','w2'])].drop_duplicates(['null_name','scenario','alt_type','level','replicate','seed']); print('unique_ot_solves', len(ot)); print('success_rate', float(ot['success'].mean())); print('median_nit', float(ot['nit'].median())); print('max_nit', int(ot['nit'].max())); print('median_grad_inf', float(ot['grad_inf'].median())); print('max_grad_inf', float(ot['grad_inf'].max())); print(ot.groupby(['success','status','message']).size().to_string())"
done

echo
echo "== Combined max_iter summary =="
"$PYTHON" -c "
import pathlib
import pandas as pd

base = pathlib.Path('$GOF_TEST_OUTDIR')
rows = []
for run in sorted(base.glob('max_iter=*')):
    files = sorted((run / 'raw' / 'calibration').glob('**/chunk=*.csv'))
    if not files:
        continue
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    ot = df[df['statistic'].isin(['potential', 'w2'])].drop_duplicates([
        'null_name', 'scenario', 'alt_type', 'level', 'replicate', 'seed'
    ])
    rows.append({
        'max_iter': int(run.name.split('=')[1]),
        'unique_ot_solves': len(ot),
        'success_rate': float(ot['success'].mean()),
        'median_nit': float(ot['nit'].median()),
        'max_nit': int(ot['nit'].max()),
        'median_grad_inf': float(ot['grad_inf'].median()),
        'max_grad_inf': float(ot['grad_inf'].max()),
        'median_runtime_seconds': float(ot['runtime_seconds'].median()),
        'max_runtime_seconds': float(ot['runtime_seconds'].max()),
    })

summary = pd.DataFrame(rows).sort_values('max_iter')
path = base / 'max_iter_summary.csv'
summary.to_csv(path, index=False)
print(summary.to_string(index=False))
print('saved', path)
"

echo
echo "GOF max_iter pilot completed successfully."
