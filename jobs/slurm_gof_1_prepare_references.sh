#!/bin/bash
#
# Stage 1: prepare fixed GOF reference clouds for all null distributions.
#
# Submit:
#   sbatch jobs/slurm_gof_1_prepare_references.sh
#
# Optional overrides:
#   sbatch --export=ALL,GOF_OUTDIR=results_gof,GOF_N_SOLVE=10000,GOF_N_EVAL_SOURCE=10000 jobs/slurm_gof_1_prepare_references.sh
#
#SBATCH --account=stats
#SBATCH --job-name=ot_gof_refs
#SBATCH --output=logs/ot_gof_refs_%j.out
#SBATCH --error=logs/ot_gof_refs_%j.err
#SBATCH -c 1
#SBATCH --time=0-04:00
#SBATCH --mem-per-cpu=16gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export GOF_MODE="prepare_references"
export GOF_NULL="${GOF_NULL:-all}"

bash jobs/slurm_gof.sh
