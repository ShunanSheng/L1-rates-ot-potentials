#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from otexp.io import ensure_dir
from otexp.rates import beta_rate


RAW_FILENAME_RE = re.compile(
    r"^trials_d=(?P<d>\d+)_n=(?P<n>\d+)_B=(?P<B>\d+)_source=(?P<n_source>\d+)_seed=(?P<seed>\d+)\.csv$"
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
    return df


def _load_raw_dir(raw_dir: Path):
    files = sorted(raw_dir.glob("trials_*.csv"))
    if not files:
        return None
    return pd.concat([_read_raw_with_metadata(f) for f in files], ignore_index=True)

def _log_beta_slope(n, d):
    if d in (1, 2, 3, 4):
        return -0.5
    return -2.0 / d

def plot_results(summary_path, figdir, raw_dir=None, ncols=3):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    summary_path = Path(summary_path)
    figdir = ensure_dir(figdir)

    raw = None
    if raw_dir is not None:
        raw = _load_raw_dir(Path(raw_dir))

    if raw is None:
        df = pd.read_csv(summary_path)
        d_values = sorted(df["d"].unique())
    else:
        df = None
        d_values = sorted(raw["d"].unique())

    if not d_values:
        raise ValueError("No data found to plot.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(d_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, d in enumerate(d_values):
        ax = axes[idx // ncols][idx % ncols]

        if raw is not None:
            sub_raw = raw[raw["d"] == d]
            sub_raw = sub_raw[sub_raw["loss"].notna()]
            ax.scatter(
                sub_raw["n"],
                sub_raw["loss"],
                s=16,
                c="0.6",
                alpha=0.6,
                marker="*",
                label="_nolegend_",
            )

            stats_by_n = (
                sub_raw.groupby("n", as_index=False)["loss"]
                .agg(mean="mean", std="std", count="size")
                .sort_values("n")
            )
            n = stats_by_n["n"].to_numpy()
            mean_loss = stats_by_n["mean"].to_numpy()
            se_loss = stats_by_n["std"].to_numpy() / np.sqrt(stats_by_n["count"].clip(lower=1))
        else:
            sub = df[df["d"] == d].sort_values("n")
            n = sub["n"].to_numpy()
            mean_loss = sub["mean_loss"].to_numpy()
            se_loss = sub["se_loss"].to_numpy()

        ax.plot(n, mean_loss, "o", color="black", label=r"$\mathrm{Average\ value\ for\ each\ } n$")
        if se_loss is not None:
            band = 1.96 * se_loss
            ax.errorbar(
                n,
                mean_loss,
                yerr=band,
                fmt="none",
                ecolor="0.35",
                elinewidth=1.6,
                capsize=5,
                capthick=1.6,
                alpha=0.9,
                zorder=2,
                label="_nolegend_",
            )

        valid = np.isfinite(mean_loss) & (mean_loss > 0) & np.isfinite(n) & (n > 0)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(np.log(n[valid]), np.log(mean_loss[valid]), 1)
            n_line = np.logspace(np.log10(n[valid].min()), np.log10(n[valid].max()), 100)
            fit_y = np.exp(intercept) * n_line**slope
            ax.plot(n_line, fit_y, color="blue", label=rf"$\mathrm{{Best\ linear\ fit}},\ \mathrm{{slope}}={slope:.2f}$")

            log_beta_slope = _log_beta_slope(n, int(d))
            ax.plot(
                [],
                [],
                color="red",
                linestyle="--",
                label=f"Reference slope = {log_beta_slope:.2f}",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$d = {int(d)}$")
        ax.legend(fontsize=8)

    total = nrows * ncols
    for idx in range(len(d_values), total):
        fig.delaxes(axes[idx // ncols][idx % ncols])

    fig.supxlabel(r"$n$", fontsize=20)
    fig.supylabel(r"$\mathrm{loss}$", fontsize=20)
    fig.tight_layout()

    out_path = Path(figdir) / "loglog_ot_potential_unit_ball_grid_reference_added.pdf"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=str, default="results/summary/aggregated.csv")
    parser.add_argument("--raw_dir", type=str, default="results/raw")
    parser.add_argument("--figdir", type=str, default="results/figs")
    parser.add_argument("--ncols", type=int, default=3)
    args = parser.parse_args()
    plot_results(args.summary_path, args.figdir, raw_dir=args.raw_dir, ncols=args.ncols)
