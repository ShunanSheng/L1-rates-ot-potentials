#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otexp.gof import ALT_LEVELS  # noqa: E402
from otexp.io import ensure_dir  # noqa: E402
from otexp.sampling import GOF_NULLS  # noqa: E402


STAT_LABELS = {
    "potential": "Potential",
    "w2": "W2^2",
    "mmd": "MMD",
}
ALT_TITLES = {
    "location_shift": "Location shift",
    "scale": "Scale change",
    "mixture_contamination": "Mixture contamination",
}


def plot_gof(outdir="results_gof"):
    outdir = Path(outdir)
    summary_path = outdir / "summary" / "power_summary.csv"
    figdir = ensure_dir(outdir / "figs")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing GOF power summary: {summary_path}")

    power = pd.read_csv(summary_path)
    if power.empty:
        raise ValueError(f"GOF power summary is empty: {summary_path}")

    for null_name in GOF_NULLS:
        sub_null = power[power["null_name"] == null_name]
        if sub_null.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
        for ax, alt_type in zip(axes, ALT_LEVELS):
            sub_alt = sub_null[sub_null["alt_type"] == alt_type]
            for statistic, label in STAT_LABELS.items():
                sub = sub_alt[sub_alt["statistic"] == statistic].sort_values("level")
                if sub.empty:
                    continue
                ax.plot(
                    sub["level"],
                    sub["estimate"],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                )
                if "mc_se" in sub:
                    lower = (sub["estimate"] - 1.96 * sub["mc_se"]).clip(lower=0.0)
                    upper = (sub["estimate"] + 1.96 * sub["mc_se"]).clip(upper=1.0)
                    ax.fill_between(sub["level"], lower, upper, alpha=0.12)

            ax.set_title(ALT_TITLES[alt_type])
            ax.set_xlabel("Perturbation level")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel("Empirical rejection probability")
        axes[-1].legend(loc="lower right")
        fig.suptitle(null_name)
        fig.tight_layout()

        pdf_path = figdir / f"power_{null_name}_1x3.pdf"
        png_path = figdir / f"power_{null_name}_1x3.png"
        fig.savefig(pdf_path, dpi=200)
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GOF empirical power summaries.")
    parser.add_argument("--outdir", type=str, default="results_gof")
    args = parser.parse_args()
    plot_gof(args.outdir)


if __name__ == "__main__":
    main()
