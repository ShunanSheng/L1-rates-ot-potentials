from pathlib import Path
import time

import numpy as np
import pandas as pd

from .sampling import phi_true, sample_mu
from .core import evaluate_phi_hat, one_trial, solve_weights
from .rates import beta_rate
from .io import ensure_dir, save_csv_atomic, save_json


def _raw_filename(d, n, B, n_source, seed):
    return f"trials_d={d}_n={n}_B={B}_source={n_source}_seed={seed}.csv"


def _raw_shard_filename(d, n, B, n_source, seed, b_start, b_stop):
    return (
        f"trials_d={d}_n={n}_B={B}_source={n_source}_seed={seed}"
        f"_b={b_start}-{b_stop}.csv"
    )


def _summary_filename(d, B, n_source, seed):
    return f"summary_d={d}_B={B}_source={n_source}_seed={seed}.csv"


def _summary_shard_filename(d, n, B, n_source, seed, b_start, b_stop):
    return (
        f"summary_d={d}_n={n}_B={B}_source={n_source}_seed={seed}"
        f"_b={b_start}-{b_stop}.csv"
    )


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
    b_start=0,
    b_stop=None,
    overwrite=False,
    save_results=True,
):
    """
    Run B trials for one pair (d,n) and return a summary row.

    When save_results=True, trial-level CSV and metadata JSON files are saved.
    When save_results=False, all trial results stay in memory.
    """
    if b_stop is None:
        b_stop = B
    if not (0 <= b_start < b_stop <= B):
        raise ValueError(f"Invalid trial range [{b_start}, {b_stop}) for B={B}")

    is_shard = b_start != 0 or b_stop != B

    if save_results:
        outdir = Path(outdir)
        raw_dir = ensure_dir(outdir / ("raw_shards" if is_shard else "raw"))
        meta_dir = ensure_dir(outdir / ("metadata_shards" if is_shard else "metadata"))
        if is_shard:
            raw_path = raw_dir / _raw_shard_filename(d, n, B, n_source, seed, b_start, b_stop)
        else:
            raw_path = raw_dir / _raw_filename(d, n, B, n_source, seed)
        meta_path = meta_dir / raw_path.with_suffix(".json").name

    rows = []
    completed_b = set()
    start_time = time.time()

    if save_results and raw_path.exists() and not overwrite:
        raw_df_existing = pd.read_csv(raw_path)
        if "b" in raw_df_existing.columns:
            raw_df_existing = raw_df_existing.drop_duplicates("b", keep="last")
            raw_df_existing = raw_df_existing[raw_df_existing["b"].between(b_start, b_stop - 1)]
            raw_df_existing = raw_df_existing.sort_values("b")
            rows = raw_df_existing.to_dict("records")
            completed_b = set(raw_df_existing["b"].astype(int))
        else:
            rows = raw_df_existing.to_dict("records")

    if len(completed_b) >= (b_stop - b_start) and not overwrite:
        raw_df = pd.DataFrame(rows)
    else:
        if completed_b:
            print(
                f"[resume] d={d}, n={n}, b=[{b_start}, {b_stop}): "
                f"found {len(completed_b)}/{b_stop - b_start} completed trials",
                flush=True,
            )
        for b in range(b_start, b_stop):
            if b in completed_b:
                continue

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
                    "nfev": out["nfev"],
                    "njev": out["njev"],
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
                    "nfev": np.nan,
                    "njev": np.nan,
                    "fun": np.nan,
                    "grad_inf": np.nan,
                    "beta": float(beta_rate(n, d)),
                    "runtime_seconds": time.time() - trial_start,
                    "error_message": repr(exc),
                })

            print(
                f"[trial] d={d}, n={n}, b={b + 1}/{B}, "
                f"runtime={rows[-1]['runtime_seconds']:.2f}s, "
                f"success={rows[-1]['success']}, nit={rows[-1]['nit']}, "
                f"nfev={rows[-1]['nfev']}",
                flush=True,
            )

            # Save partial progress every 5 trials, and at the end.
            if save_results and ((b + 1) % 5 == 0 or (b + 1) == b_stop):
                partial_df = pd.DataFrame(rows).drop_duplicates("b", keep="last").sort_values("b")
                save_csv_atomic(partial_df, raw_path)

        raw_df = pd.DataFrame(rows).drop_duplicates("b", keep="last").sort_values("b")
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
                "b_start": b_start,
                "b_stop": b_stop,
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
    b_start=0,
    b_stop=None,
    overwrite=False,
    save_results=True,
):
    """
    Run the unit-ball experiment for one dimension d and several n values.
    """
    if b_stop is None:
        b_stop = B
    is_shard = b_start != 0 or b_stop != B

    if save_results:
        outdir = Path(outdir)
        ensure_dir(outdir / ("raw_shards" if is_shard else "raw"))
        ensure_dir(outdir / ("summary_shards" if is_shard else "summary"))

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
            b_start=b_start,
            b_stop=b_stop,
            overwrite=overwrite,
            save_results=save_results,
        )
        rows.append(row)
        if save_results:
            summary_df = pd.DataFrame(rows)
            if is_shard:
                summary_path = outdir / "summary_shards" / _summary_shard_filename(
                    d, int(n), B, n_source, seed, b_start, b_stop
                )
            else:
                summary_path = outdir / "summary" / _summary_filename(d, B, n_source, seed)
            save_csv_atomic(summary_df, summary_path)
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


def run_gof_experiment(
    *,
    d=2,
    n=32,
    B=100,
    n_eval=100,
    alpha=0.05,
    n_source=4096,
    seed=2026,
    use_qmc_source=True,
    max_iter=180,
    chunk_size=2048,
    gtol=1e-6,
    ftol=1e-10,
):
    """
    Run a Monte Carlo goodness-of-fit experiment on the unit ball.

    The null is H0: nu = mu0, where mu0 is Unif(B_d). The statistic is

        T_n = sqrt(n) || phi_hat_n - ||.||^2/2
              - med(phi_hat_n - ||.||^2/2) ||_{L1(mu0)}.

    Parameters
    ----------
    B:
        Number of Monte Carlo replicates used to calibrate the critical value.
    n_eval:
        Number of fresh null and alternative samples used to estimate type I
        error and power after calibration.
    Returns
    -------
    summary_df, null_df, sim_df:
        summary_df contains the estimated type I error and power. null_df
        contains the calibration replicates. sim_df contains the fresh null and
        alternative evaluation replicates.
    """
    rng = np.random.default_rng(seed)
    X_source = sample_mu(
        n_source,
        d,
        rng=rng,
        seed=seed + 1000 * d,
        use_qmc=use_qmc_source,
    )
    phi_source = phi_true(X_source)

    def statistic(Y):
        h, res = solve_weights(
            X_source,
            Y,
            max_iter=max_iter,
            chunk_size=chunk_size,
            gtol=gtol,
            ftol=ftol,
        )
        phi_hat = evaluate_phi_hat(X_source, Y, h, chunk_size=chunk_size)
        diff = phi_hat - phi_source
        centered_l1 = float(np.mean(np.abs(diff - np.median(diff))))
        return {
            "T": float(np.sqrt(Y.shape[0]) * centered_l1),
            "centered_l1": centered_l1,
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nit": int(res.nit),
            "fun": float(res.fun),
            "grad_inf": float(np.max(np.abs(res.jac))),
        }

    null_rows = []
    for b in range(B):
        trial_seed = seed + b + 1
        trial_rng = np.random.default_rng(trial_seed)
        start_time = time.time()
        try:
            Y = sample_mu(n, d, rng=trial_rng, use_qmc=False)
            out = statistic(Y)
            null_rows.append({
                "scenario": "calibration_null",
                "replicate": b,
                "seed": trial_seed,
                **out,
                "runtime_seconds": time.time() - start_time,
                "error_message": "",
            })
        except Exception as exc:
            null_rows.append(_gof_error_row("calibration_null", b, trial_seed, start_time, exc))

    null_df = pd.DataFrame(null_rows)
    null_T = null_df.loc[null_df["T"].notna(), "T"].to_numpy()
    critical_value = _empirical_critical_value(null_T, alpha)

    sim_rows = []
    for r in range(n_eval):
        null_seed = seed + 300_000 + r
        null_rng = np.random.default_rng(null_seed)
        start_time = time.time()
        try:
            Y = sample_mu(n, d, rng=null_rng, use_qmc=False)
            out = statistic(Y)
            sim_rows.append({
                "scenario": "null",
                "replicate": r,
                "seed": null_seed,
                "reject": bool(out["T"] > critical_value),
                **out,
                "critical_value": critical_value,
                "runtime_seconds": time.time() - start_time,
                "error_message": "",
            })
        except Exception as exc:
            sim_rows.append(_gof_error_row("null", r, null_seed, start_time, exc, critical_value=critical_value))

        alt_seed = seed + 400_000 + r
        alt_rng = np.random.default_rng(alt_seed)
        start_time = time.time()
        try:
            Y = _sample_center_biased_ball(n, d, alt_rng)
            out = statistic(Y)
            sim_rows.append({
                "scenario": "alternative",
                "replicate": r,
                "seed": alt_seed,
                "reject": bool(out["T"] > critical_value),
                **out,
                "critical_value": critical_value,
                "runtime_seconds": time.time() - start_time,
                "error_message": "",
            })
        except Exception as exc:
            sim_rows.append(_gof_error_row("alternative", r, alt_seed, start_time, exc, critical_value=critical_value))

    sim_df = pd.DataFrame(sim_rows)
    summary_rows = []
    for scenario, group in sim_df.groupby("scenario", sort=False):
        reject = group["reject"].dropna().astype(bool)
        estimate = float(reject.mean()) if len(reject) else np.nan
        summary_rows.append({
            "quantity": "type I error" if scenario == "null" else "power",
            "scenario": scenario,
            "estimate": estimate,
            "mc_se": float(np.sqrt(estimate * (1.0 - estimate) / len(reject))) if len(reject) else np.nan,
            "n_eval": int(len(reject)),
            "alpha": alpha,
            "critical_value": critical_value,
            "calibration_B": int(len(null_T)),
            "mean_T": float(group["T"].mean()),
            "median_T": float(group["T"].median()),
            "success_rate": float(group["success"].mean()),
            "error_rate": float(group["error_message"].astype(bool).mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, null_df, sim_df


def _empirical_critical_value(values, alpha):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return np.nan
    idx = int(np.ceil((1.0 - alpha) * len(values))) - 1
    return float(values[np.clip(idx, 0, len(values) - 1)])


def _sample_center_biased_ball(n, d, rng):
    Z = rng.normal(size=(n, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    radii = rng.random(n) ** (2.0 / d)
    return radii[:, None] * Z


def _gof_error_row(scenario, replicate, seed, start_time, exc, critical_value=np.nan):
    return {
        "scenario": scenario,
        "replicate": replicate,
        "seed": seed,
        "reject": np.nan,
        "T": np.nan,
        "centered_l1": np.nan,
        "success": False,
        "status": -999,
        "message": "exception",
        "nit": np.nan,
        "fun": np.nan,
        "grad_inf": np.nan,
        "critical_value": critical_value,
        "runtime_seconds": time.time() - start_time,
        "error_message": repr(exc),
    }
