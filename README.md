# OT potential experiment template

This project packages the unit-ball empirical OT-potential experiment into reusable Python modules, a local Jupyter notebook workflow, and Slurm job files.
This ZIP also includes `CODEX_INSTRUCTIONS.md`, a more detailed task specification intended for Codex or another coding agent.


The production grid requested here is

```text
d in {2, 3, 4, 10}
n_target = (256, 512, 1024, 2048, 4096)
n_source = 4096
B = 100
```

The code computes an approximation of

```text
E[ inf_a || phi_hat_n - phi - a ||_{L1(mu)} ]
```

for `mu = Unif(B_d)` and `phi(x) = 0.5 ||x||^2`.

## 1. Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"
```

If you use conda instead:

```bash
conda create -n ot-exp python=3.11 -y
conda activate ot-exp
pip install -e ".[notebook]"
```

For cluster jobs where Jupyter is not needed, `pip install -e .` is enough.

## 2. Local notebook smoke test

Use the notebook for local testing and exploration:

```bash
jupyter lab notebooks/local_test.ipynb
```

The notebook runs a tiny version first. This checks imports, solver calls, and plotting without using the command-line scripts or saving results automatically.

The default notebook smoke-test parameters are:

```text
d = 2
n_target = (16, 32)
B = 2
n_source = 512
seed = 2026
max_iter = 30
```

Check that the notebook directly displays the summary dataframe and log-log plot.

## 3. Local medium notebook test

After the smoke test works, edit the parameter cell in the notebook to run one production-size `n` with a small number of Monte Carlo repetitions:

```python
d = 2
n_target = (256,)
B = 3
n_source = 4096
max_iter = 80
```

This is the best way to estimate runtime and memory before using the cluster.

## 4. Command-line scripts

The command-line scripts are intended for cluster testing and production runs. This runs the full requested grid sequentially; usually it should be run on a cluster, not a laptop.

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

## 5. Ginsburg cluster run with Slurm

The Ginsburg examples use `sbatch <script>` for batch submission, and the `stats` account matches the template you provided. This project is CPU-based, so the Slurm files here do not request a GPU. The GPU directives in the DDPM template, such as `--gres=gpu:1`, `--constraint=rtx8000`, and `module load cuda11.1/toolkit`, are only needed for CUDA/GPU code.

### 5.1 Copy the project to Ginsburg

From your laptop, copy or sync this project to your Ginsburg home directory or, preferably for larger outputs, the `stats` scratch area. Columbia recommends using the transfer host for `scp`:

```bash
scp -r /path/to/ot_experiment <UNI>@motion.rcs.columbia.edu:/burg/stats/users/<UNI>/ot_experiment
```

Then log in and enter the project:

```bash
ssh <UNI>@ginsburg.rcs.columbia.edu
# Short form also works:
# ssh <UNI>@burg.rcs.columbia.edu
cd /burg/stats/users/<UNI>/ot_experiment
```

If you keep the code under `/burg/home/<UNI>`, write large result directories under `/burg/stats/...` instead of filling the 50 GB home quota.

### 5.2 Create the Python environment once

On Ginsburg:

```bash
module load anaconda/3-2023.09
conda env create -f environment.yml
conda activate ot-exp
python -m pip install -e .
python -c "import otexp; print('otexp import OK')"
```

If the conda environment already exists, update it instead:

```bash
module load anaconda/3-2023.09
conda env update -f environment.yml --prune
conda activate ot-exp
python -m pip install -e .
```

### 5.3 Run a one-hour smoke test first

```bash
mkdir -p logs
sbatch jobs/slurm_test.sh
```

Check the queue and logs:

```bash
squeue -u "$USER"
ls logs
tail -n 80 logs/ot_test_<JOBID>.out
tail -n 80 logs/ot_test_<JOBID>.err
```

The smoke test writes to `results_test/`. Confirm that these exist before launching the production array:

```bash
ls results_test/raw results_test/summary results_test/figs
```

### 5.4 Submit the production grid

There are two Slurm templates.

#### Option A: one array task per dimension

Each task runs all five `n` values for one dimension.

```bash
sbatch jobs/slurm_array_by_d.sh
```

#### Option B: one array task per `(d,n)` pair

This is easier to restart and is usually safer.

```bash
sbatch jobs/slurm_array_by_pair.sh
```

Monitor it with:

```bash
squeue -u "$USER"
tail -n 60 logs/ot_pair_<ARRAY_JOBID>_<TASKID>.out
```

To rerun only one failed `(d,n)` pair, resubmit the matching array index. For example, task `0` is `(d=2,n=100)` and task `29` is `(d=10,n=10000)`:

```bash
sbatch --array=0 jobs/slurm_array_by_pair.sh
```

### 5.5 Aggregate and plot after the array finishes

Submit the post-processing job:

```bash
sbatch jobs/slurm_postprocess.sh
```

Or run the same commands interactively from the project root:

After all jobs finish:

```bash
python scripts/aggregate.py \
  --raw_dir results/raw \
  --out_path results/summary/aggregated.csv

python scripts/plot_results.py \
  --summary_path results/summary/aggregated.csv \
  --figdir results/figs
```

The final plot is written to:

```text
results/figs/loglog_ot_potential_unit_ball_grid.png
```

## 6. Notes

- The code is CPU-based. Do not request GPU unless you rewrite the solver to use CUDA/PyTorch/JAX.
- The `n=4096`, `B=100` cases can be expensive. Test one small job first.
- The scripts skip existing raw files by default. Add `--overwrite` if you want to rerun them.
- Raw trial-level outputs are saved separately for each `(d,n)`.
