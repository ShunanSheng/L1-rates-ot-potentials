#!/bin/bash
#
# Merge raw_shards into the usual raw and summary outputs.
#
#SBATCH --account=stats
#SBATCH --job-name=ot_merge
#SBATCH --output=logs/ot_merge_%j.out
#SBATCH --error=logs/ot_merge_%j.err
#SBATCH -c 1
#SBATCH --time=0-02:00
#SBATCH --mem-per-cpu=8gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs results/raw results/summary

module load anaconda/3-2023.09

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

"$PYTHON" scripts/aggregate_shards.py --outdir results --overwrite
