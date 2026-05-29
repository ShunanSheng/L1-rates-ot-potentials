# Codex Task: Set Up and Run OT Potential Numerical Experiment

## Goal

Set up a reproducible Python workflow for the numerical experiment estimating

\[
\mathbb{E}\left[\inf_a \|\widehat\varphi_n - \varphi - a\|_{L^1(\mu)}\right]
\]

for the semi-discrete optimal transport potential experiment on the unit ball.

The production experiment should test:

```text
d_list = [2, 3, 4, 10]
n_target = [256, 512, 1024, 2048, 4096]
n_source = 4096
B = 100
seed = 2026
```

The theoretical comparison rate is

```text
beta(n,d) = n^{-1/2},                                  d = 1,2,3
beta(n,d) = n^{-1/2} (log n)^{5/2},                    d = 4
beta(n,d) = n^{-2/d} (log n)^{(d+2)/4},                d >= 5
```

The workflow should support:

1. local smoke tests through Jupyter notebooks;
2. local medium-scale tests through Jupyter notebooks;
3. command-line runs for Slurm cluster jobs;
4. aggregation of raw Monte Carlo outputs;
5. log-log plotting of empirical loss versus fitted theoretical rate.

---

## Preferred Research Workflow

Use a hybrid workflow appropriate for statistical/numerical research:

```text
Conda environment
+ modular Python project
+ local Jupyter notebook smoke/medium tests
+ Slurm job arrays for large runs
+ raw CSV/JSON result files
+ aggregate/plot after jobs finish
```

Local testing should be notebook-first. The command-line scripts should be kept
for cluster testing and Slurm production runs.

Use Conda/Mamba if available, especially on a cluster. Plain `venv` is also acceptable for this project because the dependencies are simple.

Recommended environment:

```bash
conda create -n ot-exp python=3.11
conda activate ot-exp
python -m pip install -e ".[notebook]"
```

If using `venv` instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
```

Do not mix multiple environment systems within the same project unless necessary.

---

## Repository Structure

Create or maintain the following structure:

```text
ot_experiment/
├── pyproject.toml
├── README.md
├── src/
│   └── otexp/
│       ├── __init__.py
│       ├── sampling.py
│       ├── core.py
│       ├── rates.py
│       └── io.py
├── scripts/
│   ├── run_experiment.py
│   ├── run_grid.py
│   ├── aggregate.py
│   └── plot_results.py
├── notebooks/
│   └── local_test.ipynb
├── jobs/
│   ├── slurm_test.sh
│   └── slurm_array_by_pair.sh
├── results/
│   ├── raw/
│   ├── summary/
│   └── figs/
└── logs/
```

The purpose of each part:

```text
src/otexp/        reusable scientific code
scripts/          command-line experiment scripts
notebooks/        local smoke and medium-scale interactive tests
jobs/             cluster submission scripts
results/raw/      raw output for each (d,n) or run
results/summary/  aggregated summaries
results/figs/     log-log plots
logs/             Slurm stdout/stderr logs
```

---

## Core Python Requirements

The project should install the following dependencies:

```text
numpy
pandas
scipy
matplotlib
```

A minimal `pyproject.toml` should include:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "otexp"
version = "0.1.0"
description = "Numerical experiments for OT potential convergence rates"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
]

[project.optional-dependencies]
notebook = [
    "ipykernel",
    "jupyterlab",
]

[tool.setuptools.packages.find]
where = ["src"]
```

---

## Scientific Code Organization

### `src/otexp/rates.py`

Implement:

```python
def beta_rate(n, d):
    """
    beta(n,d), up to multiplicative constants:
        n^{-1/2},                         d=1,2,3
        n^{-1/2} log(n)^{5/2},             d=4
        n^{-2/d} log(n)^{(d+2)/4},         d>=5
    """
```

### `src/otexp/sampling.py`

Implement sampling from the unit ball:

```python
def sample_unit_ball_iid(M, d, rng):
    """IID samples from Unif(B_d)."""


def sample_unit_ball_qmc(M, d, seed=123):
    """Sobol/QMC source cloud approximately uniform on B_d."""


def sample_mu(M, d, rng=None, seed=123, use_qmc=True):
    """Source measure mu = Unif(B_d)."""


def sample_nu(n, d, rng):
    """Target measure nu = mu = Unif(B_d)."""


def phi_true(X):
    """True potential phi(x) = 0.5 ||x||^2."""
```

### `src/otexp/core.py`

Implement the semi-discrete OT dual solver:

```python
def dual_obj_grad(theta, X, Y, chunk_size=2048):
    """Dual objective and gradient for semi-discrete OT."""


def solve_weights(X, Y, max_iter=180):
    """Solve for semi-discrete dual weights."""


def evaluate_phi_hat(X, Y, h, chunk_size=2048):
    """Evaluate the empirical potential on source cloud X."""


def one_trial(n, d, X_source, rng, max_iter=180):
    """
    One Monte Carlo repetition.

    Returns a dictionary with at least:
        loss
        success
        grad_inf
    """
```

### `src/otexp/io.py`

Implement robust saving utilities:

```python
def ensure_dir(path):
    """Create directory if needed and return Path object."""


def save_results(df, out_path):
    """Save CSV atomically if possible."""


def save_metadata(metadata, out_path):
    """Save metadata as JSON."""
```

---

## Experiment Scripts

### `scripts/run_experiment.py`

This should run one experiment for a single dimension `d` and possibly multiple `n_target` values.

It should accept command-line arguments:

```text
--d
--n_target
--B
--n_source
--seed
--outdir
--max_iter
--use_qmc_source / --no-use_qmc_source
```

Example cluster smoke-test command:

```bash
python scripts/run_experiment.py \
  --d 2 \
  --n_target 16 32 \
  --B 2 \
  --n_source 512 \
  --seed 2026 \
  --outdir results_smoke \
  --max_iter 30
```

Expected output:

```text
results_smoke/raw/trials_d=2_n=16_B=2_source=512_seed=2026.csv
results_smoke/raw/trials_d=2_n=32_B=2_source=512_seed=2026.csv
results_smoke/metadata/trials_d=2_n=16_B=2_source=512_seed=2026.json
results_smoke/metadata/trials_d=2_n=32_B=2_source=512_seed=2026.json
```

Each raw CSV should include one row per Monte Carlo repetition, with columns:

```text
d
n
b
seed
loss
success
grad_inf
beta
runtime_seconds
error_message
```

### `scripts/run_grid.py`

This should run a full grid from the command line for cluster testing or production.

Arguments:

```text
--d_list
--n_target
--B
--n_source
--seed
--outdir
--max_iter
```

Production cluster command:

```bash
python scripts/run_grid.py \
  --d_list 2 3 4 10 \
  --n_target 256 512 1024 2048 4096 \
  --B 100 \
  --n_source 4096 \
  --seed 2026 \
  --outdir results \
  --max_iter 180
```

For ordinary local exploration, use `notebooks/local_test.ipynb`. For the full production grid, prefer Slurm arrays.

### `scripts/aggregate.py`

Aggregate all raw CSV files.

Command:

```bash
python scripts/aggregate.py \
  --raw_dir results/raw \
  --out_path results/summary/aggregated.csv
```

The aggregated CSV should include:

```text
d
n
mean_loss
std_loss
se_loss
num_trials
success_rate
median_grad_inf
beta
mean_loss_over_beta
```

### `scripts/plot_results.py`

Plot log-log comparison for each dimension.

Command:

```bash
python scripts/plot_results.py \
  --summary_path results/summary/aggregated.csv \
  --figdir results/figs
```

Expected figures:

```text
results/figs/loglog_d=2.png
results/figs/loglog_d=3.png
results/figs/loglog_d=4.png
results/figs/loglog_d=10.png
```

Each plot should show:

1. empirical mean loss with 95% error bars;
2. best fitted multiple `C * beta(n,d)`;
3. log-log axes.

---

## Local Notebook Validation Workflow

### Step 1: Install project

Using Conda:

```bash
conda activate ot-exp
python -m pip install -e ".[notebook]"
```

Or using `venv`:

```bash
source .venv/bin/activate
python -m pip install -e ".[notebook]"
```

### Step 2: Run notebook smoke test

A smoke test is a tiny run whose only purpose is to check the full pipeline end-to-end. It is not meant to produce scientific results.

Start Jupyter from the project root:

```bash
jupyter lab notebooks/local_test.ipynb
```

Use these notebook parameters:

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

This checks:

```text
sampling works
OT solver runs
losses are finite
summary dataframe is displayed
plotting works inline
```

Acceptance criteria:

```text
No import errors.
No missing-directory errors.
Summary dataframe is displayed.
Plot is displayed inline.
Loss values are finite or failures are explicitly recorded in error_message.
```

### Step 3: Run medium local notebook test

Run one realistic setting with small `B` by editing the notebook parameter cell:

```python
d = 2
n_target = (256,)
B = 3
n_source = 4096
seed = 2026
max_iter = 80
chunk_size = 2048
save_results = False
```

This checks realistic memory/runtime behavior before cluster submission.

Acceptance criteria:

```text
Runtime is reasonable.
Memory does not blow up.
Solver success rate is not catastrophically low.
CSV and JSON metadata are saved.
```

---

## Slurm Cluster Workflow

The cluster job should not request GPU unless the code explicitly uses GPU acceleration. This project currently uses NumPy/SciPy and should run as a CPU job.

Do not include:

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000
module load cuda11.1/toolkit
```

unless the code is rewritten to use GPU libraries.

---

## Slurm Test Script

Create `jobs/slurm_test.sh`:

```bash
#!/bin/bash
#
#SBATCH --account=stats
#SBATCH --job-name=ot_test
#SBATCH --output=logs/ot_test_%j.out
#SBATCH --error=logs/ot_test_%j.err
#SBATCH -c 1
#SBATCH --time=0-01:00
#SBATCH --mem-per-cpu=5gb

set -euo pipefail

cd /path/to/ot_experiment

mkdir -p logs
mkdir -p results/raw
mkdir -p results/summary
mkdir -p results/figs

# Option A: Conda
source ~/.bashrc
conda activate ot-exp

# Option B: venv instead of Conda
# source .venv/bin/activate

python scripts/run_experiment.py \
  --d 2 \
  --n_target 256 \
  --B 5 \
  --n_source 4096 \
  --seed 2026 \
  --outdir results \
  --max_iter 80
```

Submit:

```bash
sbatch jobs/slurm_test.sh
```

Check logs:

```bash
ls logs
cat logs/ot_test_*.out
cat logs/ot_test_*.err
```

---

## Slurm Production Array Script

Create `jobs/slurm_array_by_pair.sh`:

```bash
#!/bin/bash
#
#SBATCH --account=stats
#SBATCH --job-name=ot_array
#SBATCH --output=logs/ot_%A_%a.out
#SBATCH --error=logs/ot_%A_%a.err
#SBATCH -c 1
#SBATCH --time=0-06:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --array=0-19

set -euo pipefail

cd /path/to/ot_experiment

mkdir -p logs
mkdir -p results/raw
mkdir -p results/summary
mkdir -p results/figs

# Option A: Conda
source ~/.bashrc
conda activate ot-exp

# Option B: venv instead of Conda
# source .venv/bin/activate

PARAMS=(
  "2 256"
  "2 512"
  "2 1024"
  "2 2048"
  "2 4096"
  "3 256"
  "3 512"
  "3 1024"
  "3 2048"
  "3 4096"
  "4 256"
  "4 512"
  "4 1024"
  "4 2048"
  "4 4096"
  "10 256"
  "10 512"
  "10 1024"
  "10 2048"
  "10 4096"
)

read D N <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

python scripts/run_experiment.py \
  --d "$D" \
  --n_target "$N" \
  --B 100 \
  --n_source 4096 \
  --seed 2026 \
  --outdir results \
  --max_iter 180
```

Submit:

```bash
sbatch jobs/slurm_array_by_pair.sh
```

Monitor:

```bash
squeue -u $USER
```

After jobs finish:

```bash
python scripts/aggregate.py \
  --raw_dir results/raw \
  --out_path results/summary/aggregated.csv

python scripts/plot_results.py \
  --summary_path results/summary/aggregated.csv \
  --figdir results/figs
```

---

## Result-File Policy

Save raw results, not just averages.

Each raw CSV should correspond to one setting or one controlled block:

```text
results/raw/d=2_n=256_B=100_seed=2026.csv
results/raw/d=2_n=512_B=100_seed=2026.csv
...
```

Each raw JSON metadata file should include:

```text
d
n
B
n_source
seed
use_qmc_source
max_iter
runtime_seconds
python_version
numpy_version
scipy_version
pandas_version
```

Avoid overwriting silently. If a file already exists, either:

1. skip it;
2. overwrite only if `--overwrite` is passed;
3. write to a timestamped file.

Preferred behavior: default to not overwriting existing results unless `--overwrite` is provided.

---

## Runtime and Memory Notes

The expensive part is the semi-discrete OT dual evaluation. For source cloud size `M = n_source` and target size `n`, score matrices are computed in chunks.

For production:

```text
n_source = 4096
n up to 4096
```

The score matrix for one chunk of size 2048 and `n=4096` has about:

```text
2048 * 4096 ≈ 8.4 million entries
```

At 8 bytes per float, this is about 67 MB for the score matrix, before overhead. This is manageable, but runtime may be substantial for `B=100` and large `n`.

Keep `chunk_size=2048` or reduce it if memory errors occur.

---

## Suggested Execution Order

Run in this order:

```text
1. Install environment.
2. Run notebook smoke test.
3. Check notebook-displayed summary and plot outputs.
4. Run medium local notebook test.
5. Upload project to cluster if needed.
6. Run one Slurm test job.
7. Submit full Slurm job array.
8. Aggregate production results.
9. Generate production figures.
10. Inspect success rates and gradient diagnostics.
```

Do not debug `one_trial` inside a large Slurm array. Debug locally in the notebook or with one tiny Slurm job first.

---

## Acceptance Checklist for Codex

The task is complete if all of the following hold:

```text
[ ] Project installs locally with `python -m pip install -e ".[notebook]"`.
[ ] Project installs on the cluster with `python -m pip install -e .`.
[ ] `python scripts/run_experiment.py --help` works.
[ ] Notebook smoke test runs successfully.
[ ] Notebook smoke test displays a summary dataframe.
[ ] Notebook smoke test displays an inline plot.
[ ] Notebook medium test with d=2, n=256, B=3, n_source=4096 runs.
[ ] Slurm test script is present and uses CPU resources only.
[ ] Slurm production array covers exactly 20 pairs: 4 dimensions times 5 sample sizes.
[ ] Production settings are d=2,3,4,10; n=256,512,1024,2048,4096; B=100; n_source=4096.
[ ] Raw outputs include solver success and gradient diagnostics.
[ ] Aggregated output includes mean loss, standard error, success rate, beta, and mean_loss_over_beta.
[ ] Plotting script creates one log-log comparison plot per dimension.
```

---

## Notes for Future Extensions

Potential improvements after the first successful production run:

```text
1. Parallelize over Monte Carlo repetitions B instead of only over (d,n).
2. Add `--chunk_size` as a command-line argument.
3. Add retry logic for failed solver runs.
4. Add timing per trial.
5. Save fitted slopes in a separate summary file.
6. Add support for other source measures beyond Unif(B_d).
7. Add Git commit hash to metadata if repository is under Git.
```
