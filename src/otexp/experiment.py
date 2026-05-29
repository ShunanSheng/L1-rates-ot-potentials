from pathlib import Path
import time

import numpy as np
import pandas as pd

from .sampling import sample_mu
from .core import one_trial
from .rates import beta_rate
from .io import ensure_dir, save_csv_atomic, save_json


def _raw_filename(d, n, B, n_source, seed):
    return f"trials_d={d}_n={n}_B={B}_source={n_source}_seed={seed}.csv"


def _summary_filename(d, B, n_source, seed):
    return f"summary_d={d}_B={B}_source={n_source}_seed={seed}.csv"


def run_one_n(
    *,
    d,
    n,
    B,
    X_source,
    seed,
    outdir,
    n_source,
    max_iter=180,
    chunk_size=2048,
    overwrite=False,
    save_results=True,
):
    """
    Run B trials for one pair (d,n) and return a summary row.

    When save_results=True, trial-level CSV and metadata JSON files are saved.
    When save_results=False, all trial results stay in memory.
    """
    if save_results:
        outdir = Path(outdir)
        raw_dir = ensure_dir(outdir / "raw")
        meta_dir = ensure_dir(outdir / "metadata")
        raw_path = raw_dir / _raw_filename(d, n, B, n_source, seed)
        meta_path = meta_dir / raw_path.with_suffix(".json").name

    if save_results and raw_path.exists() and not overwrite:
        raw_df = pd.read_csv(raw_path)
    else:
        rows = []
        start_time = time.time()

        for b in range(B):
            trial_seed = seed + 1_000_000 * d + 10_000 * n + b
            trial_rng = np.random.default_rng(trial_seed)
            trial_start = time.time()

            try:
                out = one_trial(
                    n=n,
                    d=d,
                    X_source=X_source,
                    rng=trial_rng,
                    max_iter=max_iter,
                    chunk_size=chunk_size,
                )
                rows.append({
                    "d": d,
                    "n": n,
                    "b": b,
                    "seed": trial_seed,
                    "loss": out["loss"],
                    "success": out["success"],
                    "status": out["status"],
                    "message": out["message"],
                    "nit": out["nit"],
                    "fun": out["fun"],
                    "grad_inf": out["grad_inf"],
                    "beta": float(beta_rate(n, d)),
                    "runtime_seconds": time.time() - trial_start,
                    "error_message": "",
                })
            except Exception as exc:
                rows.append({
                    "d": d,
                    "n": n,
                    "b": b,
                    "seed": trial_seed,
                    "loss": np.nan,
                    "success": False,
                    "status": -999,
                    "message": "exception",
                    "nit": np.nan,
                    "fun": np.nan,
                    "grad_inf": np.nan,
                    "beta": float(beta_rate(n, d)),
                    "runtime_seconds": time.time() - trial_start,
                    "error_message": repr(exc),
                })

            # Save partial progress every 5 trials, and at the end.
            if save_results and ((b + 1) % 5 == 0 or (b + 1) == B):
                save_csv_atomic(pd.DataFrame(rows), raw_path)

        raw_df = pd.DataFrame(rows)
        if save_results:
            save_csv_atomic(raw_df, raw_path)
            save_json({
                "d": d,
                "n": n,
                "B": B,
                "n_source": n_source,
                "seed": seed,
                "max_iter": max_iter,
                "chunk_size": chunk_size,
                "raw_path": str(raw_path),
                "runtime_seconds_total": time.time() - start_time,
            }, meta_path)

    losses = raw_df["loss"].dropna().to_numpy()
    beta = float(beta_rate(n, d))

    if len(losses) >= 2:
        std_loss = float(np.std(losses, ddof=1))
        se_loss = float(std_loss / np.sqrt(len(losses)))
    elif len(losses) == 1:
        std_loss = 0.0
        se_loss = 0.0
    else:
        std_loss = np.nan
        se_loss = np.nan

    mean_loss = float(np.mean(losses)) if len(losses) else np.nan

    return {
        "d": d,
        "n": n,
        "B": B,
        # "B_completed": int(len(raw_df)),
        # "B_success_loss": int(len(losses)),
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "se_loss": se_loss,
        "beta": beta,
        "mean_loss_over_beta": mean_loss / beta if np.isfinite(mean_loss) else np.nan,
        # "success_rate": float(raw_df["success"].mean()) if len(raw_df) else np.nan,
        # "median_grad_inf": float(raw_df["grad_inf"].median()) if len(raw_df) else np.nan,
        # "median_runtime_seconds": float(raw_df["runtime_seconds"].median()) if len(raw_df) else np.nan,
        # "raw_file": str(raw_path),
    }


def run_ball_experiment(
    *,
    d=2,
    n_target=(256, 512, 1024, 2048, 4096),
    B=100,
    n_source=4096,
    seed=2026,
    use_qmc_source=True,
    outdir="results",
    max_iter=180,
    chunk_size=2048,
    overwrite=False,
    save_results=True,
):
    """
    Run the unit-ball experiment for one dimension d and several n values.
    """
    if save_results:
        outdir = Path(outdir)
        ensure_dir(outdir / "raw")
        ensure_dir(outdir / "summary")

    rng = np.random.default_rng(seed)
    X_source = sample_mu(
        n_source,
        d,
        rng=rng,
        seed=seed + 1000 * d,
        use_qmc=use_qmc_source,
    )

    rows = []
    for n in n_target:
        # print(f"[run] d={d}, n={n}, B={B}, n_source={n_source}", flush=True)
        row = run_one_n(
            d=d,
            n=int(n),
            B=B,
            X_source=X_source,
            seed=seed,
            outdir=outdir,
            n_source=n_source,
            max_iter=max_iter,
            chunk_size=chunk_size,
            overwrite=overwrite,
            save_results=save_results,
        )
        rows.append(row)
        if save_results:
            summary_df = pd.DataFrame(rows)
            save_csv_atomic(summary_df, outdir / "summary" / _summary_filename(d, B, n_source, seed))
        # print(summary_df.tail(1).to_string(index=False), flush=True)

    df = pd.DataFrame(rows)

    valid = df["mean_loss"].notna() & (df["mean_loss"] > 0)
    if valid.sum() >= 2:
        slope_loss, _ = np.polyfit(np.log(df.loc[valid, "n"]), np.log(df.loc[valid, "mean_loss"]), deg=1)
        slope_beta, _ = np.polyfit(np.log(df.loc[valid, "n"]), np.log(df.loc[valid, "beta"]), deg=1)
    else:
        slope_loss = np.nan
        slope_beta = np.nan

    return df, float(slope_loss), float(slope_beta)
