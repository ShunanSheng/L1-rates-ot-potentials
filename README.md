# L1 Rates for OT Potentials

This repository contains numerical experiments for empirical semi-discrete optimal transport potentials on the unit ball.

The main rate experiment estimates

```text
E[ inf_a || phi_hat_n - phi - a ||_{L1(mu)} ]
```

where `mu = Unif(B_d)` and `phi(x) = 0.5 ||x||^2`. The project also includes a Monte Carlo goodness-of-fit experiment based on the same fitted potential error statistic.

## Setup

Create the environment from the project root:

```bash
conda env create -f environment.yml
conda activate ot-exp
python -m pip install -e .
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate ot-exp
python -m pip install -e .
```

For notebook use, install the notebook extras if available:

```bash
python -m pip install -e ".[notebook]"
```

## Semi-Discrete OT Computation

For each target sample `Y_1, ..., Y_n`, the code fits weights `h_i` for the empirical potential

```text
phi_hat(x) = max_i { x . Y_i - 0.5 ||Y_i||^2 + h_i }.
```

The weights solve the discretized semi-discrete dual objective over a source cloud `X_solve` sampled from `mu`:

```text
M^{-1} sum_m max_i { X_solve_m . Y_i - 0.5 ||Y_i||^2 + h_i }
- n^{-1} sum_i h_i.
```

This is implemented with L-BFGS-B in `src/otexp/core.py`:

```text
dual_obj_grad -> solve_weights -> evaluate_phi_hat
```

The current implementation uses two independent source clouds of size `n_source`:

```text
X_solve: used to compute the semi-discrete OT weights
X_eval:  used only to evaluate the L1(mu) loss/statistic
```

This avoids evaluating the fitted potential on the same cloud used to fit the weights. Output filenames for these runs include `_split_eval`.

## Rate Experiment

Run one dimension from the command line:

```bash
python scripts/run_experiment.py \
  --d 2 \
  --n_target 100 300 1000 3000 10000 \
  --B 100 \
  --n_source 10000 \
  --seed 2026 \
  --outdir results \
  --max_iter 1000
```

Run a sequential grid:

```bash
python scripts/run_grid.py \
  --d_list 2 3 4 6 8 10 \
  --n_target 100 300 1000 3000 10000 \
  --B 100 \
  --n_source 10000 \
  --seed 2026 \
  --outdir results \
  --max_iter 1000
```

Outputs are written to:

```text
results/raw/trials_d=<d>_n=<n>_B=<B>_source=<n_source>_seed=<seed>_split_eval.csv
results/metadata/
results/summary/
```

Existing raw files are resumed by default. Use `--overwrite` to recompute a matching output file from scratch.

## Goodness-of-Fit Test

The GOF experiment tests

```text
H0: nu = mu0, where mu0 = Unif(B_d).
```

For each dataset `Y`, the statistic is

```text
T_n = sqrt(n) * mean_{X_eval} | (phi_hat(X_eval) - phi(X_eval))
                              - median(phi_hat(X_eval) - phi(X_eval)) |.
```

The workflow is:

1. Draw `B` null datasets to calibrate the empirical critical value.
2. Draw `n_eval` fresh null datasets to estimate empirical size.
3. Draw `n_eval` datasets for each alternative/theta to estimate empirical power.

The alternatives are:

```text
location_shift:        Y = X + theta e1
scale:                 Y = (1 + theta) X
mixture_contamination: (1 - theta) mu0 + theta Law(X + mixture_shift e1)
```

Run GOF directly:

```bash
python scripts/run_gof.py \
  --d 3 \
  --n 3000 \
  --B 1000 \
  --n_eval 100 \
  --n_source 5000 \
  --seed 2026 \
  --max_iter 200 \
  --chunk_size 2500 \
  --location_thetas 0.02 0.05 0.1 \
  --scale_thetas 0.02 0.05 0.1 \
  --mixture_thetas 0.02 0.05 0.1 \
  --outdir results_gof
```

GOF outputs are written to:

```text
results_gof/summary/
results_gof/raw/
```

The summary CSV includes:

```text
estimate: empirical rejection probability
mc_se: Monte Carlo standard error of estimate over n_eval
critical_value: empirical null critical value from B calibration runs
success_rate: fraction of solver runs reporting success
error_rate: fraction of runs with caught exceptions
source_clouds: independent_solve_and_eval
```

## Slurm Workflow

All jobs are CPU-only.

Create logs and submit a rate smoke test:

```bash
mkdir -p logs
sbatch jobs/slurm_test.sh
```

Submit the production rate array, one task per `(d,n)` pair:

```bash
sbatch jobs/slurm_array_by_pair.sh
```

Monitor jobs and logs:

```bash
squeue -u "$USER"
tail -f logs/ot_pair_<ARRAY_JOBID>_<TASKID>.out
tail -f logs/ot_pair_<ARRAY_JOBID>_<TASKID>.err
```

After the rate array finishes, aggregate and plot:

```bash
sbatch jobs/slurm_postprocess.sh
```

The same postprocess commands can be run interactively:

```bash
python scripts/aggregate.py \
  --raw_dir results/raw \
  --out_path results/summary/aggregated.csv

python scripts/plot_results.py \
  --summary_path results/summary/aggregated.csv \
  --raw_dir results/raw \
  --figdir results/figs
```

Submit the GOF job:

```bash
sbatch jobs/slurm_gof.sh
```

Override GOF settings at submission time:

```bash
sbatch --export=ALL,GOF_D=3,GOF_N=3000,GOF_B=1000,GOF_N_EVAL=100 jobs/slurm_gof.sh
```

Check completed job runtime:

```bash
sacct -j <JOBID> --format=JobID,JobName%20,State,Elapsed,Start,End,MaxRSS,ExitCode
```

## Repository Layout

```text
src/otexp/          reusable sampling, solver, rate, and experiment code
scripts/           command-line rate, GOF, aggregation, and plotting scripts
jobs/              Slurm submission scripts
notebooks/         local exploratory notebooks
results/           rate experiment outputs
results_gof/       GOF experiment outputs
logs/              Slurm stdout/stderr
```

## Notes

- `n_source` is the size of each source cloud. The code now draws one cloud for solving and one independent cloud for evaluation.
- `B` is the Monte Carlo repetition count for the rate experiment, and the null calibration count for GOF.
- `n_eval` is used only in GOF; it is the number of fresh evaluation datasets per scenario.
- If empirical GOF power is `1.0` for all alternatives, decrease the theta grid.
- If solver `success_rate` is low, increase `max_iter` or loosen/tune optimization tolerances.
