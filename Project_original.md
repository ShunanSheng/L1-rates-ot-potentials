# Project: OT Potential Rate and Goodness-of-Fit Experiments

## Goal

Maintain a reproducible Python workflow for semi-discrete optimal transport
experiments on the unit ball. The project now supports two related numerical
studies:

1. A rate experiment estimating

   ```text
   E[ inf_a || phi_hat_n - phi - a ||_{L1(mu)} ]
   ```

   where `mu = Unif(B_d)` and `phi(x) = 0.5 ||x||^2`.

2. A Monte Carlo goodness-of-fit test for

   ```text
   H0: nu = mu0, where mu0 = Unif(B_d).
   ```

   The GOF statistic reuses the fitted OT potential error from the rate
   experiment and calibrates a critical value by null simulation.

The shared semi-discrete OT computation is implemented in `src/otexp/core.py`.

---

## Current Production Settings

### Rate Experiment

The original production grid is:

```text
d_list = [2, 3, 4, 10]
n_target = [256, 512, 1024, 2048, 4096]
n_source = 4096
B = 100
seed = 2026
```

The theoretical comparison rate is:

```text
beta(n,d) = n^{-1/2},                                  d = 1,2,3
beta(n,d) = n^{-1/2} (log n)^{5/2},                    d = 4
beta(n,d) = n^{-2/d} (log n)^{(d+2)/4},                d >= 5
```

### Goodness-of-Fit Experiment

The current cluster-scale GOF run is:

```text
d = 3
n = 3000
B = 1000
n_eval = 100
n_source = 5000
seed = 2026
alpha = 0.05
location_thetas = [0.02, 0.05, 0.1]
scale_thetas = [0.02, 0.05, 0.1]
mixture_thetas = [0.02, 0.05, 0.1]
mixture_shift = 0.5
max_iter = 200
chunk_size = 2500
```

The GOF workflow:

1. Draw `B` null samples to calibrate an empirical critical value.
2. Draw `n_eval` fresh null samples to estimate empirical size.
3. Draw `n_eval` samples for each alternative and theta to estimate empirical
   power.

The alternatives are:

```text
location_shift:        Y = X + theta e1
scale:                 Y = (1 + theta) X
mixture_contamination: (1 - theta) mu0 + theta Law(X + mixture_shift e1)
```

---

## Repository Structure

```text
ot_experiment/
|-- pyproject.toml
|-- environment.yml
|-- README.md
|-- Project.md
|-- src/
|   `-- otexp/
|       |-- __init__.py
|       |-- sampling.py
|       |-- core.py
|       |-- experiment.py
|       |-- rates.py
|       `-- io.py
|-- scripts/
|   |-- run_experiment.py
|   |-- run_grid.py
|   |-- run_gof.py
|   |-- aggregate.py
|   `-- plot_results.py
|-- notebooks/
|   |-- local_test.ipynb
|   `-- gof_small_sample_trial.ipynb
|-- jobs/
|   |-- slurm_test.sh
|   |-- slurm_array_by_pair.sh
|   |-- slurm_array_by_d.sh
|   |-- slurm_gof.sh
|   `-- slurm_postprocess.sh
|-- results/
|   |-- raw/
|   |-- metadata/
|   |-- summary/
|   `-- figs/
|-- results_gof/
|   |-- raw/
|   `-- summary/
`-- logs/
```

Purpose of each part:

```text
src/otexp/        reusable scientific code
scripts/          command-line rate, GOF, aggregation, and plotting scripts
notebooks/        local smoke, medium-scale, and GOF exploratory tests
jobs/             Slurm submission scripts
results/          rate experiment outputs
results_gof/      GOF calibration, simulation, and summary outputs
logs/             Slurm stdout/stderr logs
```

---

## Environment

Recommended setup from the project root:

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

For notebooks:

```bash
python -m pip install -e ".[notebook]"
```

Core dependencies:

```text
numpy
pandas
scipy
matplotlib
```

---

## Semi-Discrete OT Solver

For each target sample `Y_1, ..., Y_n`, the code fits weights `h_i` for

```text
phi_hat(x) = max_i { x . Y_i - 0.5 ||Y_i||^2 + h_i }.
```

The weights minimize the discretized semi-discrete dual objective on a source
cloud `X_solve`:

```text
M^{-1} sum_m max_i { X_solve_m . Y_i - 0.5 ||Y_i||^2 + h_i }
- n^{-1} sum_i h_i.
```

The project now uses two independent source clouds:

```text
X_solve: used to compute semi-discrete OT weights
X_eval:  used only to evaluate the L1(mu) loss or GOF statistic
```

This split avoids evaluating the fitted potential on the same cloud used to
fit the weights. Output filenames for these runs include `_split_eval`.

---

## Documented Code Changes

### `src/otexp/core.py`

`core.py` contains the shared solver primitives used by both experiments.

Current responsibilities:

```text
dual_obj_grad(theta, X, Y, chunk_size=2048)
solve_weights(X, Y, max_iter=1000, chunk_size=2048, gtol=1e-5, ftol=1e-10)
evaluate_phi_hat(X, Y, h, chunk_size=2048)
one_trial(n, d, X_solve, rng=None, max_iter=180, chunk_size=2048, X_eval=None)
```

Important changes:

```text
- The solve and evaluation source clouds are separated through X_solve and X_eval.
- one_trial defaults X_eval to X_solve for backward compatibility.
- solve_weights exposes chunk_size, gtol, and ftol so GOF runs can tune the optimizer.
- dual_obj_grad and evaluate_phi_hat compute score matrices in chunks to control memory.
- one_trial returns solver diagnostics: success, status, message, nit, nfev, njev, fun, grad_inf.
- The L1 loss is centered by subtracting the median of phi_hat - phi, which is the optimal additive shift for L1.
```

### `src/otexp/experiment.py`

`experiment.py` is the orchestration layer.

Rate experiment functions:

```text
run_one_n(...)
run_ball_experiment(...)
```

GOF experiment function:

```text
run_gof_experiment(...)
```

Important changes:

```text
- run_ball_experiment draws independent QMC source clouds for solving and evaluation.
- run_one_n writes split-evaluation raw CSVs and metadata JSON files.
- run_one_n can resume existing raw CSVs unless overwrite=True.
- run_gof_experiment reuses solve_weights and evaluate_phi_hat directly.
- run_gof_experiment computes T = sqrt(n) * centered_l1.
- GOF calibration, fresh null evaluation, and alternative evaluation are separated.
- GOF summaries report empirical size/power, Monte Carlo standard error, critical value, solver success rate, and error rate.
```

### `src/otexp/sampling.py`

Sampling utilities define the null model and source clouds.

Current responsibilities:

```text
sample_unit_ball_iid(M, d, rng)
sample_unit_ball_qmc(M, d, seed=123)
sample_mu(M, d, rng=None, seed=123, use_qmc=True)
sample_nu(n, d, rng)
phi_true(X)
```

Important notes:

```text
- sample_mu supports Sobol/QMC source clouds for stable integration over mu0.
- sample_nu draws IID target samples under the null.
- GOF alternatives are generated in experiment.py by transforming IID null samples.
```

### `src/otexp/io.py`

I/O utilities now provide atomic CSV saving for partial and final outputs:

```text
ensure_dir(path)
save_csv_atomic(df, path)
save_json(obj, path)
```

### `scripts/run_gof.py`

`run_gof.py` is the command-line entry point for the goodness-of-fit experiment.

It accepts:

```text
--d
--n
--B
--n_eval
--alpha
--location_thetas
--scale_thetas
--mixture_thetas
--mixture_shift
--n_source
--seed
--use_qmc_source / --no-use_qmc_source
--max_iter
--chunk_size
--gtol
--ftol
--outdir
```

It writes:

```text
results_gof/summary/summary_gof_*.csv
results_gof/raw/calibration_null_gof_*.csv
results_gof/raw/simulation_gof_*.csv
```

### `jobs/slurm_gof.sh`

`slurm_gof.sh` runs the GOF experiment as a CPU-only Slurm job and supports
environment-variable overrides:

```text
GOF_D
GOF_N
GOF_B
GOF_N_EVAL
GOF_N_SOURCE
GOF_SEED
GOF_MAX_ITER
GOF_CHUNK_SIZE
GOF_OUTDIR
```

Example override:

```bash
sbatch --export=ALL,GOF_D=3,GOF_N=3000,GOF_B=1000,GOF_N_EVAL=100 jobs/slurm_gof.sh
```

### `scripts/aggregate.py` and `scripts/plot_results.py`

The rate postprocessing scripts understand `_split_eval` filenames.

```text
aggregate.py prefers trials_*_split_eval.csv when present.
plot_results.py can plot raw split-evaluation trials and saves the combined PDF.
```

---

## Running the Rate Experiment

One dimension:

```bash
python scripts/run_experiment.py \
  --d 2 \
  --n_target 256 512 1024 2048 4096 \
  --B 100 \
  --n_source 4096 \
  --seed 2026 \
  --outdir results \
  --max_iter 180 \
  --chunk_size 2048
```

Sequential grid:

```bash
python scripts/run_grid.py \
  --d_list 2 3 4 10 \
  --n_target 256 512 1024 2048 4096 \
  --B 100 \
  --n_source 4096 \
  --seed 2026 \
  --outdir results \
  --max_iter 180 \
  --chunk_size 2048
```

Rate raw outputs:

```text
results/raw/trials_d=<d>_n=<n>_B=<B>_source=<n_source>_seed=<seed>_split_eval.csv
results/metadata/trials_d=<d>_n=<n>_B=<B>_source=<n_source>_seed=<seed>_split_eval.json
```

Rate summaries:

```text
results/summary/summary_d=<d>_B=<B>_source=<n_source>_seed=<seed>_split_eval.csv
results/summary/aggregated.csv
```

Raw trial columns include:

```text
d
n
b
seed
loss
success
status
message
nit
nfev
njev
fun
grad_inf
beta
runtime_seconds
error_message
```

---

## Running the Goodness-of-Fit Experiment

Small local smoke test:

```bash
python scripts/run_gof.py \
  --d 2 \
  --n 32 \
  --B 20 \
  --n_eval 10 \
  --n_source 512 \
  --seed 2026 \
  --max_iter 40 \
  --chunk_size 512 \
  --location_thetas 0.1 \
  --scale_thetas 0.1 \
  --mixture_thetas 0.1 \
  --outdir results_gof_smoke
```

Cluster-scale GOF run:

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

GOF output files:

```text
results_gof/summary/summary_gof_d=<d>_n=<n>_B=<B>_eval=<n_eval>_source=<n_source>_seed=<seed>_split_eval_*.csv
results_gof/raw/calibration_null_gof_d=<d>_n=<n>_B=<B>_eval=<n_eval>_source=<n_source>_seed=<seed>_split_eval_*.csv
results_gof/raw/simulation_gof_d=<d>_n=<n>_B=<B>_eval=<n_eval>_source=<n_source>_seed=<seed>_split_eval_*.csv
```

GOF summary columns include:

```text
quantity
scenario
alternative
theta
estimate
mc_se
n_eval
alpha
critical_value
calibration_B
mean_T
median_T
success_rate
error_rate
source_clouds
source_solve_seed
source_eval_seed
```

Interpretation:

```text
quantity = empirical size for the fresh null scenario
quantity = empirical power for alternatives
estimate = fraction of fresh samples with T > critical_value
mc_se = sqrt(estimate * (1 - estimate) / n_eval)
critical_value = empirical 1 - alpha null quantile from B calibration samples
```

---

## Notebook Workflow

Use notebooks for local statistical and numerical checks before cluster runs.

Rate smoke test:

```text
notebooks/local_test.ipynb
```

Suggested parameters:

```python
d = 2
n_target = (16, 32)
B = 2
n_source = 512
seed = 2026
max_iter = 30
chunk_size = 512
save_results = False
```

GOF smoke or small-sample exploration:

```text
notebooks/gof_small_sample_trial.ipynb
```

Suggested parameters:

```python
d = 2
n = 32
B = 20
n_eval = 10
n_source = 512
seed = 2026
max_iter = 40
chunk_size = 512
```

Acceptance criteria:

```text
No import errors.
Loss/statistic values are finite or failures are recorded in error_message.
Solver success rate is not catastrophically low.
Runtime and memory are reasonable before submitting Slurm jobs.
```

---

## Slurm Workflow

All jobs are CPU-only. Do not request GPUs unless the solver is rewritten to
use GPU libraries.

Rate smoke test:

```bash
mkdir -p logs
sbatch jobs/slurm_test.sh
```

Rate production array:

```bash
sbatch jobs/slurm_array_by_pair.sh
```

Rate postprocess:

```bash
sbatch jobs/slurm_postprocess.sh
```

Goodness-of-fit job:

```bash
sbatch jobs/slurm_gof.sh
```

Override GOF settings at submission time:

```bash
sbatch --export=ALL,GOF_D=3,GOF_N=3000,GOF_B=1000,GOF_N_EVAL=100 jobs/slurm_gof.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/ot_gof_<JOBID>.out
tail -f logs/ot_gof_<JOBID>.err
```

Check completed job accounting:

```bash
sacct -j <JOBID> --format=JobID,JobName%20,State,Elapsed,Start,End,MaxRSS,ExitCode
```

---

## Postprocessing

Aggregate rate results:

```bash
python scripts/aggregate.py \
  --raw_dir results/raw \
  --out_path results/summary/aggregated.csv
```

Plot rate results:

```bash
python scripts/plot_results.py \
  --summary_path results/summary/aggregated.csv \
  --raw_dir results/raw \
  --figdir results/figs
```

Current plotting output:

```text
results/figs/loglog_ot_potential_unit_ball_grid_reference_added_eval_split.pdf
```

GOF output is already summarized by `scripts/run_gof.py`; no separate GOF
postprocessing script is currently required.

---

## Runtime and Memory Notes

The expensive operation is forming chunked score matrices of shape
`chunk_size x n`.

For `chunk_size = 2048` and `n = 4096`:

```text
2048 * 4096 ~= 8.4 million entries
8 bytes per float ~= 67 MB for the score matrix before overhead
```

Guidance:

```text
- Keep chunk_size near 2048 or 2500 for large runs unless memory pressure appears.
- Reduce chunk_size if jobs fail with memory errors.
- Increase max_iter or tune gtol/ftol if solver success_rate is low.
- For GOF, reduce theta values if empirical power is 1.0 for all alternatives.
```

---

## Result-File Policy

Save raw results, not just averages.

Rate files can resume partial trial CSVs. Use `--overwrite` on rate scripts only
when recomputation is intended.

GOF files include theta grids in the filename so different alternative grids do
not silently collide.

Raw and summary outputs should preserve:

```text
solver diagnostics
random seeds
source cloud seeds
runtime information
error_message for caught exceptions
```

---

## Suggested Execution Order

```text
1. Install or update the environment.
2. Run the rate notebook smoke test.
3. Run the GOF notebook or CLI smoke test.
4. Run one Slurm rate smoke job.
5. Run one Slurm GOF smoke or reduced-size job if cluster timing is uncertain.
6. Submit the production rate array.
7. Submit the production GOF job.
8. Aggregate and plot rate results.
9. Inspect GOF empirical size, power, critical values, success rates, and error rates.
```

Do not debug solver behavior inside a large Slurm array. Use the notebook, a
tiny CLI run, or one reduced Slurm job first.

---

## Acceptance Checklist

```text
[ ] `python -m pip install -e ".[notebook]"` works locally.
[ ] `python -m pip install -e .` works on the cluster.
[ ] `python scripts/run_experiment.py --help` works.
[ ] `python scripts/run_gof.py --help` works.
[ ] Rate notebook smoke test runs.
[ ] GOF notebook or CLI smoke test runs.
[ ] `src/otexp/core.py` supports split solve/evaluation clouds.
[ ] `run_ball_experiment` writes `_split_eval` rate outputs.
[ ] `run_gof_experiment` returns summary, calibration null, and simulation dataframes.
[ ] GOF summary reports empirical size and empirical power by scenario and theta.
[ ] Slurm scripts are CPU-only.
[ ] Raw outputs include solver success and gradient diagnostics.
[ ] Rate aggregation and plotting understand `_split_eval` raw files.
```

---

## Future Extensions

```text
1. Add a dedicated GOF postprocessing and plotting script.
2. Parallelize over GOF calibration and evaluation replicates.
3. Add Git commit hash and package versions to metadata for every run.
4. Add retry logic for solver failures.
5. Save fitted rate slopes in a separate summary file.
6. Add alternatives beyond location, scale, and mixture contamination.
7. Add source measures beyond Unif(B_d).
```
