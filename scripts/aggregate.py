#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from otexp.io import ensure_dir, save_csv_atomic


RAW_FILENAME_RE = re.compile(
    r"^trials_d=(?P<d>\d+)_n=(?P<n>\d+)_B=(?P<B>\d+)_source=(?P<n_source>\d+)_seed=(?P<seed>\d+)(?:_split_eval)?\.csv$"
)


def _parse_raw_filename(path: Path):
    match = RAW_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized raw CSV filename: {path.name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def _read_raw_with_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = _parse_raw_filename(path)
    if "d" not in df.columns:
        df["d"] = meta["d"]
    if "n" not in df.columns:
        df["n"] = meta["n"]
    if "B" not in df.columns:
        df["B"] = meta["B"]
    return df


def aggregate(raw_dir, out_path):
    raw_dir = Path(raw_dir)
    split_files = sorted(raw_dir.glob("trials_*_split_eval.csv"))
    files = split_files if split_files else sorted(raw_dir.glob("trials_*.csv"))

    if not files:
        raise FileNotFoundError(f"No trial CSV files found in {raw_dir}")

    if split_files:
        print(f"Using {len(split_files)} split-evaluation trial CSV files from {raw_dir}")

    raw = pd.concat([_read_raw_with_metadata(f) for f in files], ignore_index=True)

    summary = (
        raw.groupby(["d", "n"], as_index=False)
        .agg(
            mean_loss=("loss", "mean"),
            std_loss=("loss", "std"),
            B=("loss", "size"),
            # B_success_loss=("loss", "count"),
            # success_rate=("success", "mean"),
            # median_grad_inf=("grad_inf", "median"),
            # median_runtime_seconds=("runtime_seconds", "median"),
            beta=("beta", "mean"),
        )
    )
    summary["se_loss"] = summary["std_loss"] / np.sqrt(summary["B"].clip(lower=1))
    summary["mean_loss_over_beta"] = summary["mean_loss"] / summary["beta"]
    summary = summary.sort_values(["d", "n"])

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    save_csv_atomic(summary, out_path)
    print(f"Saved aggregated summary to {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="results/raw")
    parser.add_argument("--out_path", type=str, default="results/summary/aggregated.csv")
    args = parser.parse_args()
    aggregate(args.raw_dir, args.out_path)
