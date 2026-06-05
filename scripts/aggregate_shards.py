#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from otexp.io import ensure_dir, save_csv_atomic
from otexp.rates import beta_rate


SHARD_RE = re.compile(
    r"^trials_d=(?P<d>\d+)_n=(?P<n>\d+)_B=(?P<B>\d+)_source=(?P<n_source>\d+)"
    r"_seed=(?P<seed>\d+)_b=(?P<b_start>\d+)-(?P<b_stop>\d+)\.csv$"
)


def summarize(raw_df, d, n, B):
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
    row = {
        "d": d,
        "n": n,
        "B": B,
        "B_completed": int(raw_df["b"].nunique()) if "b" in raw_df.columns else len(raw_df),
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "se_loss": se_loss,
        "beta": beta,
        "mean_loss_over_beta": mean_loss / beta if np.isfinite(mean_loss) else np.nan,
    }
    if "success" in raw_df.columns:
        row["success_rate"] = float(raw_df["success"].mean())
    if "grad_inf" in raw_df.columns:
        row["median_grad_inf"] = float(raw_df["grad_inf"].median())
    if "runtime_seconds" in raw_df.columns:
        row["median_runtime_seconds"] = float(raw_df["runtime_seconds"].median())
    if "nfev" in raw_df.columns:
        row["median_nfev"] = float(raw_df["nfev"].median())
    return row


def main():
    parser = argparse.ArgumentParser(description="Merge trial shard CSVs into raw and summary outputs.")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    shard_dir = outdir / "raw_shards"
    raw_dir = ensure_dir(outdir / "raw")
    summary_dir = ensure_dir(outdir / "summary")

    groups = {}
    for path in sorted(shard_dir.glob("*.csv")):
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        info = {k: int(v) for k, v in match.groupdict().items()}
        key = (info["d"], info["n"], info["B"], info["n_source"], info["seed"])
        groups.setdefault(key, []).append(path)

    if not groups:
        raise SystemExit(f"No shard CSVs found in {shard_dir}")

    summary_rows_by_run = {}
    for key, paths in sorted(groups.items()):
        d, n, B, n_source, seed = key
        raw_path = raw_dir / f"trials_d={d}_n={n}_B={B}_source={n_source}_seed={seed}.csv"
        if raw_path.exists() and not args.overwrite:
            raw_df = pd.read_csv(raw_path)
        else:
            parts = [pd.read_csv(path) for path in paths]
            raw_df = pd.concat(parts, ignore_index=True)
            raw_df = raw_df.drop_duplicates("b", keep="last").sort_values("b")
            save_csv_atomic(raw_df, raw_path)

        row = summarize(raw_df, d, n, B)
        row["raw_file"] = str(raw_path)
        summary_rows_by_run.setdefault((d, B, n_source, seed), []).append(row)

    for (d, B, n_source, seed), rows in sorted(summary_rows_by_run.items()):
        summary_df = pd.DataFrame(rows).sort_values("n")
        summary_path = summary_dir / f"summary_d={d}_B={B}_source={n_source}_seed={seed}.csv"
        save_csv_atomic(summary_df, summary_path)
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
