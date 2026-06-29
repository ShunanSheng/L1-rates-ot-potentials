#!/bin/bash
#
# Goodness-of-fit postprocessing Slurm job.
#
# Override at submission time, for example:
#   sbatch --export=ALL,GOF_POST_MODE=aggregate_only jobs/slurm_postprocessing_gof.sh
#   sbatch --export=ALL,GOF_POST_MODE=final jobs/slurm_postprocessing_gof.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_post
#SBATCH --output=logs/ot_gof_post_%j.out
#SBATCH --error=logs/ot_gof_post_%j.err
#SBATCH -c 1
#SBATCH --time=0-02:00
#SBATCH --mem-per-cpu=8gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

GOF_OUTDIR="${GOF_OUTDIR:-results_gof}"
GOF_ALPHA="${GOF_ALPHA:-0.05}"
GOF_POST_MODE="${GOF_POST_MODE:-auto}"

mkdir -p logs "${GOF_OUTDIR}/summary" "${GOF_OUTDIR}/figs"

module load anaconda/3-2023.09

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

echo "Python executable: $PYTHON"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -m pip show otexp >/dev/null
"$PYTHON" -c "import otexp; print('otexp package:', otexp.__file__)"

echo "Aggregating GOF outputs from ${GOF_OUTDIR} with alpha=${GOF_ALPHA}"
"$PYTHON" scripts/aggregate_gof.py \
  --outdir "$GOF_OUTDIR" \
  --alpha "$GOF_ALPHA"

POWER_SUMMARY="${GOF_OUTDIR}/summary/power_summary.csv"

if [ "$GOF_POST_MODE" = "aggregate_only" ]; then
  echo "GOF_POST_MODE=aggregate_only: skipping GOF plotting."
  exit 0
fi

if [ "$GOF_POST_MODE" = "final" ]; then
  echo "GOF_POST_MODE=final: plotting GOF power figures into ${GOF_OUTDIR}/figs"
  "$PYTHON" scripts/plot_gof.py \
    --outdir "$GOF_OUTDIR"
  exit 0
fi

if [ "$GOF_POST_MODE" = "auto" ]; then
  if [ -s "$POWER_SUMMARY" ] && "$PYTHON" -c "import pandas as pd, sys; df = pd.read_csv(sys.argv[1]); sys.exit(0 if not df.empty else 1)" "$POWER_SUMMARY"; then
    echo "Power summary found. Plotting GOF power figures into ${GOF_OUTDIR}/figs"
    "$PYTHON" scripts/plot_gof.py \
      --outdir "$GOF_OUTDIR"
  else
    echo "No nonempty power summary found at ${POWER_SUMMARY}; aggregation finished, skipping plots."
  fi
  exit 0
fi

echo "Unknown GOF_POST_MODE=${GOF_POST_MODE}. Use auto, aggregate_only, or final." >&2
exit 2
