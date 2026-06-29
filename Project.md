# Project: OT Potential Rate and Goodness-of-Fit Experiments

## Goal

Maintain a reproducible Python workflow for semi-discrete optimal transport
experiments. The project supports two related numerical studies:

1. A rate experiment estimating

   ```text
   E[ inf_a || phi_hat_n - phi - a ||_{L1(mu)} ]
   ```

   where `mu = Unif(B_d)` and `phi(x) = 0.5 ||x||^2`.

2. A Monte Carlo goodness-of-fit experiment for simple null hypotheses

   ```text
   H0: P = P0
   ```

   in dimension `d = 3`, where `P0` is one of three fully specified continuous
   distributions: uniform on the unit ball, truncated Gaussian, or truncated
   elliptical Student. The proposed test uses the fitted semi-discrete OT
   potential. The benchmarks are squared Wasserstein distance and MMD.

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

The updated GOF experiment is a simple-null experiment in dimension `d = 3`.
All tests are calibrated separately under each null distribution.

```text
d = 3
n = 500
alpha = 0.05
B_cal = 500
N_size = 500
N_alt = 500
n_solve = 10000
n_eval_source = 10000
seed = 2026
max_iter = 200
chunk_size = 2500
```

Here:

```text
B_cal        = number of null samples for critical-value calibration
N_size       = number of fresh null samples for empirical size estimation
N_alt        = number of samples for each alternative level
n_solve      = source-cloud size used to solve semi-discrete OT weights
n_eval_source= independent source-cloud size used to evaluate potential, W2, and MMD
```

The three null distributions are:

```text
uniform_ball:
    P0 = Unif(B_3(0, 1))

truncated_gaussian:
    P0 = Law(Z | ||Z|| <= 2),  Z ~ N_3(0, I_3)

truncated_elliptical_t:
    P0 = Law(Y | Y^T Sigma^{-1} Y <= 4),
    Y ~ t_{nu,3}(0, Sigma),
    nu = 5,
    Sigma = diag(1, 1.5^2, 0.7^2)
```

For every null distribution `P0`, let `X0 ~ P0` and `e1 = (1,0,0)^T`. The
three alternative families are:

```text
location_shift:
    Y = X0 + delta * e1
    delta_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

scale:
    Y = s * X0
    scale_levels = [1.025, 1.05, 1.075, 1.10, 1.15, 1.20]

mixture_contamination:
    Y ~ (1 - epsilon) P0 + epsilon Q0
    Q0 = Law(0.5 * e1 + 0.25 * X0)
    epsilon_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
```

The test statistics are:

```text
potential:
    T_pot = sqrt(n) * inf_a || phi_hat_n - 0.5 ||.||^2 - a ||_{L1(P0)}

w2:
    T_w2 = W_2^2(P_hat_n, P0)

mmd:
    T_mmd = MMD_k^2(P_hat_n, P0)
```

The GOF workflow is:

1. Prepare and save null-specific reference objects: OT solve cloud, OT
   evaluation cloud, MMD bandwidth, and MMD reference-reference term. MMD uses
   the same `X_eval` cloud as W2 and the proposed potential statistic.
2. Draw `B_cal` null samples to calibrate empirical critical values for
   `potential`, `w2`, and `mmd` under each null.
3. Draw `N_size` fresh null samples to estimate empirical size under each null.
4. Draw `N_alt` samples for each null, alternative type, and nonzero level to
   estimate empirical power.
5. Aggregate raw replicate files into size, power, and paired power-difference
   summaries.
6. Produce one `1 x 3` power figure per null. The three columns are location
   shift, scale change, and mixture contamination.

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
|       |-- gof.py
|       |-- mmd.py
|       `-- io.py
|-- scripts/
|   |-- run_experiment.py
|   |-- run_grid.py
|   |-- run_gof.py
|   |-- aggregate.py
|   |-- aggregate_gof.py
|   |-- plot_results.py
|   `-- plot_gof.py
|-- notebooks/
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
|   |-- references/
|   |-- raw/
|   |-- summary/
|   `-- figs/
`-- logs/
```

Purpose of each part:

```text
src/otexp/        reusable scientific code
scripts/          command-line rate, GOF, aggregation, and plotting scripts
notebooks/        local smoke, medium-scale, and GOF exploratory tests
jobs/             Slurm submission scripts
results/          rate experiment outputs
results_gof/      GOF references, raw simulation outputs, summaries, and figures
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

The project uses two independent source clouds:

```text
X_solve: used to compute semi-discrete OT weights
X_eval:  used to evaluate the L1(P0) potential statistic, W2 statistic, and MMD
```

For GOF, `X_solve` and `X_eval` must be sampled from the null distribution
currently being tested. Thus the source clouds are null-specific and saved under
`results_gof/references/n=<n>_eval=<n_eval_source>_solve=<n_solve>_seed=<seed>/null=<null_name>/`.

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

Required GOF behavior:

```text
- solve_weights must work with any source cloud sampled from a supported null.
- evaluate_phi_hat must return phi_hat values on X_eval.
- W2 should be computed from the same hard cell assignment used by phi_hat.
- The potential statistic must center phi_hat - phi_true by its empirical median.
- Solver diagnostics must be stored for every replicate.
```

### `src/otexp/sampling.py`

Sampling utilities must support all GOF nulls and alternatives.

Required responsibilities:

```text
sample_unit_ball_iid(M, d, rng)
sample_unit_ball_qmc(M, d, seed=123)
sample_truncated_gaussian(M, d, radius, rng)
sample_truncated_elliptical_t(M, d, radius, nu, Sigma, rng)
sample_null(null_name, M, rng, use_qmc=False, seed=None)
sample_alternative(null_name, alt_type, level, M, rng)
sample_mu(M, d, rng=None, seed=123, use_qmc=True)
sample_nu(n, d, rng)
phi_true(X)
```

Notes:

```text
- sample_null must return arrays of shape (M, 3) for GOF.
- sample_alternative must implement location_shift, scale, and mixture_contamination.
- For mixture_contamination, draw Bernoulli(epsilon) indicators independently.
- If indicator is 0, sample from P0.
- If indicator is 1, return 0.5 * e1 + 0.25 * X0 with X0 sampled from P0.
- phi_true(X) remains 0.5 * ||X||^2 for the identity null transport.
```

### `src/otexp/gof.py`

Add a dedicated GOF module. It should contain null definitions, alternative
grids, statistic computation, calibration, power simulation, and aggregation
helpers.

Required responsibilities:

```text
get_null_config(null_name)
make_gof_references(null_name, config, rng)
compute_potential_and_w2(X, references, solver_config)
compute_gof_statistics(X, null_name, references, solver_config)
run_gof_calibration(null_name, config, chunk_id=None, num_chunks=None)
run_gof_size(null_name, config, chunk_id=None, num_chunks=None)
run_gof_power(null_name, alt_type, level, config, chunk_id=None, num_chunks=None)
```

The returned statistic dictionary must include:

```text
potential
w2
mmd
```

### `src/otexp/mmd.py`

Add a small MMD utility module.

Required responsibilities:

```text
median_bandwidth(Y_ref, rng, max_points=2000)
gaussian_multiscale_kernel_sum(X, Y, sigma0, scales=(0.5, 1.0, 2.0), chunk_size=2048)
precompute_mmd_reference(Y_ref, sigma0, chunk_size=2048)
compute_mmd_unbiased(X, Y_ref, sigma0, ref_ref_term, chunk_size=2048)
```

The kernel is:

```text
k(x,y) = (1/3) sum_{b in {1/2,1,2}} exp(-||x-y||^2 / (2 (b sigma0)^2)).
```

Use the unbiased MMD estimator:

```text
MMD^2 = n^{-1}(n-1)^{-1} sum_{i != j} k(X_i, X_j)
      + m^{-1}(m-1)^{-1} sum_{a != b} k(Y_a, Y_b)
      - 2 (nm)^{-1} sum_{i,a} k(X_i, Y_a).
```

### `src/otexp/experiment.py`

`experiment.py` is the orchestration layer.

Rate experiment functions:

```text
run_one_n(...)
run_ball_experiment(...)
```

GOF experiment functions should now delegate to `src/otexp/gof.py`.

Important GOF changes:

```text
- GOF is no longer restricted to mu0 = Unif(B_d).
- GOF runs over null_name in {uniform_ball, truncated_gaussian, truncated_elliptical_t}.
- GOF computes three statistics: potential, w2, and mmd.
- GOF calibration, size estimation, and power estimation are separated.
- GOF raw files should contain one row per replicate per statistic.
- GOF summaries should report empirical size, empirical power, Monte Carlo SE, critical value, solver success rate, and error rate.
```

### `src/otexp/io.py`

I/O utilities should provide atomic saving for all raw and summary outputs:

```text
ensure_dir(path)
save_csv_atomic(df, path)
save_json(obj, path)
load_json(path)
file_exists_and_nonempty(path)
```

### `scripts/run_gof.py`

`run_gof.py` is the command-line entry point for the GOF experiment. It should
support modes.

Required arguments:

```text
--mode {prepare_references,calibration,size,power,all}
--null {uniform_ball,truncated_gaussian,truncated_elliptical_t,all}
--alt {location_shift,scale,mixture_contamination}
--level
--d
--n
--B_cal
--N_size
--N_alt
--alpha
--n_solve
--n_eval_source
--seed
--use_qmc_source / --no-use_qmc_source
--max_iter
--chunk_size
--gtol
--ftol
--chunk_id
--num_chunks
--outdir
--overwrite
```

`--mode all` should run references, calibration, size, and power sequentially for
small local tests only. Cluster runs should use separate modes and chunks.

### `scripts/aggregate_gof.py`

Aggregate raw GOF files and compute critical values, empirical size, empirical
power, and paired power differences.

Outputs:

```text
results_gof/summary/critical_values.csv
results_gof/summary/size_summary.csv
results_gof/summary/power_summary.csv
results_gof/summary/paired_power_differences.csv
results_gof/summary/runtime_summary.csv
```

### `scripts/plot_gof.py`

Create one `1 x 3` power figure per null distribution. Each figure has columns:

```text
1. Location shift
2. Scale change
3. Mixture contamination
```

Each panel plots empirical rejection probability against perturbation level with
three curves:

```text
Potential
W2^2
MMD
```

Save both PDF and PNG:

```text
results_gof/figs/power_uniform_ball_1x3.pdf
results_gof/figs/power_uniform_ball_1x3.png
results_gof/figs/power_truncated_gaussian_1x3.pdf
results_gof/figs/power_truncated_gaussian_1x3.png
results_gof/figs/power_truncated_elliptical_t_1x3.pdf
results_gof/figs/power_truncated_elliptical_t_1x3.png
```

### `jobs/slurm_gof.sh`

`slurm_gof.sh` runs GOF jobs as CPU-only Slurm jobs and supports environment
variable overrides.

Required overrides:

```text
GOF_MODE
GOF_NULL
GOF_ALT
GOF_LEVEL
GOF_D
GOF_N
GOF_B_CAL
GOF_N_SIZE
GOF_N_ALT
GOF_N_SOLVE
GOF_N_EVAL_SOURCE
GOF_SEED
GOF_MAX_ITER
GOF_CHUNK_SIZE
GOF_CHUNK_ID
GOF_NUM_CHUNKS
GOF_OUTDIR
```

Example:

```bash
sbatch --export=ALL,GOF_MODE=power,GOF_NULL=uniform_ball,GOF_ALT=location_shift,GOF_LEVEL=0.05,GOF_CHUNK_ID=0,GOF_NUM_CHUNKS=10 jobs/slurm_gof.sh
```

### `scripts/aggregate.py` and `scripts/plot_results.py`

The rate postprocessing scripts remain unchanged and continue to handle
`_split_eval` rate filenames.

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

### Small local smoke test

```bash
python scripts/run_gof.py \
  --mode all \
  --null uniform_ball \
  --d 3 \
  --n 32 \
  --B_cal 5 \
  --N_size 5 \
  --N_alt 5 \
  --n_solve 512 \
  --n_eval_source 512 \
  --seed 2026 \
  --max_iter 40 \
  --chunk_size 512 \
  --outdir results_gof_smoke \
  --overwrite
```

### Prepare references for production

```bash
python scripts/run_gof.py \
  --mode prepare_references \
  --null all \
  --d 3 \
  --n 500 \
  --n_solve 10000 \
  --n_eval_source 10000 \
  --seed 2026 \
  --outdir results_gof
```

### Calibration

```bash
python scripts/run_gof.py \
  --mode calibration \
  --null uniform_ball \
  --d 3 \
  --n 500 \
  --B_cal 500 \
  --n_solve 10000 \
  --n_eval_source 10000 \
  --seed 2026 \
  --max_iter 200 \
  --chunk_size 2500 \
  --chunk_id 0 \
  --num_chunks 10 \
  --outdir results_gof
```

Repeat calibration for:

```text
uniform_ball
truncated_gaussian
truncated_elliptical_t
```

### Fresh null size estimation

```bash
python scripts/run_gof.py \
  --mode size \
  --null uniform_ball \
  --d 3 \
  --n 500 \
  --N_size 500 \
  --seed 2026 \
  --max_iter 200 \
  --chunk_size 2500 \
  --chunk_id 0 \
  --num_chunks 10 \
  --outdir results_gof
```

### Power simulation

```bash
python scripts/run_gof.py \
  --mode power \
  --null uniform_ball \
  --alt location_shift \
  --level 0.05 \
  --d 3 \
  --n 500 \
  --N_alt 500 \
  --seed 2026 \
  --max_iter 200 \
  --chunk_size 2500 \
  --chunk_id 0 \
  --num_chunks 10 \
  --outdir results_gof
```

Repeat power jobs over:

```text
null_name in [uniform_ball, truncated_gaussian, truncated_elliptical_t]
alt in [location_shift, scale, mixture_contamination]
level in the corresponding six nonzero levels
chunk_id in 0, ..., num_chunks - 1
```

### GOF raw output files

```text
results_gof/raw/calibration/n=<n>_eval=<n_eval_source>_solve=<n_solve>_seed=<seed>/null=<null_name>/chunk=<chunk_id>.csv
results_gof/raw/size/n=<n>_eval=<n_eval_source>_solve=<n_solve>_seed=<seed>/null=<null_name>/chunk=<chunk_id>.csv
results_gof/raw/power/n=<n>_eval=<n_eval_source>_solve=<n_solve>_seed=<seed>/null=<null_name>/alt=<alt_type>/level=<level>/chunk=<chunk_id>.csv
```

Each raw file should contain one row per replicate per statistic:

```text
null_name
n
n_eval_source
n_solve
base_seed
scenario
alt_type
level
replicate
seed
statistic
value
critical_value
reject
runtime_seconds
success
status
message
nit
nfev
njev
fun
grad_inf
error_message
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
nulls = ["uniform_ball"]
d = 3
n = 32
B_cal = 5
N_size = 5
N_alt = 5
n_solve = 512
n_eval_source = 512
seed = 2026
max_iter = 40
chunk_size = 512
```

Acceptance criteria:

```text
No import errors.
All three null samplers return arrays of shape (M, 3).
All three statistics are finite or failures are recorded in error_message.
Critical values are computed for potential, w2, and mmd.
Solver success rate is not catastrophically low.
Runtime and memory are reasonable before submitting Slurm jobs.
```

---

## Slurm Workflow

All jobs are CPU-only. Do not request GPUs unless the solver is rewritten to use
GPU libraries.

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

Goodness-of-fit staged workflow:

```bash
# 1. Prepare fixed X_solve/X_eval references for all nulls.
sbatch jobs/slurm_gof_1_prepare_references.sh

# 2. Run calibration chunks for all nulls.
sbatch jobs/slurm_gof_2_calibration_array.sh

# 3. Aggregate calibration chunks to produce critical_values.csv.
sbatch --export=ALL,GOF_POST_MODE=aggregate_only jobs/slurm_postprocessing_gof.sh

# 4. Run empirical size chunks for all nulls.
sbatch jobs/slurm_gof_3_size_array.sh

# 5. Run empirical power chunks for all null/alternative/level combinations.
sbatch jobs/slurm_gof_4_power_array.sh

# 6. Final aggregation and plots.
sbatch --export=ALL,GOF_POST_MODE=final jobs/slurm_postprocessing_gof.sh
```

The staged array scripts default to `GOF_NUM_CHUNKS=10`. If changing the number
of chunks, also change the Slurm array range. For example, calibration with 5
chunks needs `3 nulls * 5 chunks = 15` tasks:

```bash
sbatch --array=0-14 --export=ALL,GOF_NUM_CHUNKS=5 jobs/slurm_gof_2_calibration_array.sh
```

The lower-level worker script is still available for one-off runs:

```bash
sbatch --export=ALL,GOF_MODE=power,GOF_NULL=uniform_ball,GOF_ALT=location_shift,GOF_LEVEL=0.05,GOF_CHUNK_ID=0,GOF_NUM_CHUNKS=10 jobs/slurm_gof.sh
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

Current rate plotting output:

```text
results/figs/loglog_ot_potential_unit_ball_grid_reference_added_eval_split.pdf
```

Aggregate GOF results:

```bash
python scripts/aggregate_gof.py \
  --outdir results_gof \
  --alpha 0.05
```

Plot GOF results:

```bash
python scripts/plot_gof.py \
  --outdir results_gof
```

GOF summary outputs:

```text
results_gof/summary/critical_values.csv
results_gof/summary/size_summary.csv
results_gof/summary/power_summary.csv
results_gof/summary/paired_power_differences.csv
results_gof/summary/runtime_summary.csv
```

GOF figure outputs:

```text
results_gof/figs/power_uniform_ball_1x3_n=<n>_n_eval_source=<n_eval_source>_n_solve=<n_solve>_base_seed=<seed>.pdf
results_gof/figs/power_uniform_ball_1x3_n=<n>_n_eval_source=<n_eval_source>_n_solve=<n_solve>_base_seed=<seed>.png
results_gof/figs/power_truncated_gaussian_1x3_n=<n>_n_eval_source=<n_eval_source>_n_solve=<n_solve>_base_seed=<seed>.pdf
results_gof/figs/power_truncated_gaussian_1x3_n=<n>_n_eval_source=<n_eval_source>_n_solve=<n_solve>_base_seed=<seed>.png
results_gof/figs/power_truncated_elliptical_t_1x3_n=<n>_n_eval_source=<n_eval_source>_n_solve=<n_solve>_base_seed=<seed>.pdf
results_gof/figs/power_truncated_elliptical_t_1x3_n=<n>_n_eval_source=<n_eval_source>_n_solve=<n_solve>_base_seed=<seed>.png
```

The GOF plots use the same serif statistical style as the rate plots. Curves
show empirical rejection probabilities with 95% Monte Carlo error bars, a
horizontal nominal-alpha reference line, and legend labels reporting empirical
size under the null.

GOF summary columns should include:

```text
null_name
alt_type
level
statistic
estimate
mc_se
num_replicates
alpha
critical_value
success_rate
error_rate
mean_value
median_value
n
n_eval_source
n_solve
base_seed
```

Paired power-difference columns should include:

```text
null_name
alt_type
level
left_statistic
right_statistic
estimate
mc_se
num_pairs
n
n_eval_source
n_solve
base_seed
```

where `estimate` is

```text
mean(1{left_statistic rejects} - 1{right_statistic rejects}).
```

---

## Runtime and Memory Notes

The expensive OT operation is forming chunked score matrices of shape
`chunk_size x n`.

For `chunk_size = 2500` and `n = 500`:

```text
2500 * 500 = 1.25 million entries
8 bytes per float ~= 10 MB for the score matrix before overhead
```

MMD computations can be expensive because the shared `X_eval` reference size is
`n_eval_source = 10000`. Compute MMD kernel sums in chunks and precompute the
`X_eval` reference-reference term once per null.

Guidance:

```text
- Keep chunk_size near 2048 or 2500 for large OT runs unless memory pressure appears.
- Reduce chunk_size if jobs fail with memory errors.
- Increase max_iter or tune gtol/ftol if solver success_rate is low.
- Store and reuse OT output for both potential and W2 statistics.
- Store and reuse shared reference samples, MMD bandwidths, and reference-reference terms.
- If all powers are 1.0, reduce perturbation levels in a follow-up run.
```

---

## Result-File Policy

Save raw results, not just averages.

Rate files can resume partial trial CSVs. Use `--overwrite` on rate scripts only
when recomputation is intended.

GOF files must include the null name, scenario, alternative type, level, and
chunk identifier in the path so distinct runs do not silently collide.

Raw and summary outputs should preserve:

```text
solver diagnostics
random seeds
reference object seeds
runtime information
critical values
rejection indicators
error_message for caught exceptions
```

Every long-running script should be restartable. If a chunk file already exists
and passes a basic integrity check, skip recomputation unless `--overwrite` is
supplied.

---

## Suggested Execution Order

```text
1. Install or update the environment.
2. Run the rate notebook smoke test.
3. Run the GOF notebook or CLI smoke test for one null and one alternative level.
4. Run one Slurm rate smoke job.
5. Run one Slurm GOF smoke or reduced-size job if cluster timing is uncertain.
6. Prepare GOF reference objects for all three nulls.
7. Submit GOF calibration jobs for all three nulls.
8. Aggregate calibration outputs and inspect critical values.
9. Submit GOF fresh-null size jobs for all three nulls.
10. Submit GOF power jobs over all nulls, alternatives, levels, and chunks.
11. Aggregate GOF size and power results.
12. Plot the three 1 x 3 GOF power figures.
13. Submit the production rate array if rate results still need updating.
14. Aggregate and plot rate results.
15. Inspect empirical size, power, paired power differences, critical values,
    success rates, error rates, and runtime summaries.
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
[ ] `src/otexp/sampling.py` implements all three GOF null distributions.
[ ] `sample_alternative` implements location_shift, scale, and mixture_contamination.
[ ] `src/otexp/mmd.py` computes the multiscale Gaussian MMD statistic in chunks.
[ ] `run_gof.py --mode prepare_references` writes null-specific reference objects.
[ ] GOF calibration writes one row per replicate per statistic.
[ ] GOF power writes one row per replicate per statistic.
[ ] `aggregate_gof.py` writes critical_values.csv, size_summary.csv, power_summary.csv, and paired_power_differences.csv.
[ ] `plot_gof.py` writes one 1 x 3 figure per null distribution.
[ ] GOF summary reports empirical size and empirical power by null, alternative, level, and statistic.
[ ] Slurm scripts are CPU-only.
[ ] Raw outputs include solver success and gradient diagnostics.
[ ] Rate aggregation and plotting understand `_split_eval` raw files.
```

---

## Future Extensions

```text
1. Add larger final runs with B_cal = 1000 and N_alt = 1000 after pilot stability.
2. Parallelize GOF calibration and evaluation replicates more aggressively.
3. Add Git commit hash and package versions to metadata for every run.
4. Add retry logic for solver failures.
5. Save fitted rate slopes in a separate summary file.
6. Add additional benchmarks only after the W2 and MMD comparison is stable.
7. Add source measures beyond the three current simple nulls.
8. Add optional sensitivity plots over solver tolerance and source-cloud size.
```
