#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otexp.gof import ALT_LEVELS  # noqa: E402
from otexp.io import ensure_dir  # noqa: E402
from otexp.sampling import GOF_NULLS  # noqa: E402


STAT_LABELS = {
    "potential": r"$\mathrm{Potential}$",
    "w2": r"$W_2^2$",
    "mmd": r"$\mathrm{MMD}$",
}
STAT_STYLE = {
    "potential": {"color": "black", "marker": "o"},
    "w2": {"color": "blue", "marker": "s"},
    "mmd": {"color": "red", "marker": "^"},
}
ALT_TITLES = {
    "location_shift": r"$\mathrm{Location\ shift}$",
    "scale": r"$\mathrm{Scale\ change}$",
    "mixture_contamination": r"$\mathrm{Mixture\ contamination}$",
}
NULL_TITLES = {
    "uniform_ball": "Uniform over ball",
    "truncated_gaussian": "Truncated Gaussian",
    "truncated_elliptical_t": "Truncated Elliptic",
}
SPEC_COLS = ["n", "n_eval_source", "n_solve", "base_seed"]


def _set_plot_style():
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
        "legend.fontsize": 11,
    })


def _present_spec_cols(df):
    return [col for col in SPEC_COLS if col in df.columns]


def _spec_suffix(keys, spec_cols):
    if not spec_cols:
        return ""
    key_map = dict(zip(spec_cols, keys if isinstance(keys, tuple) else (keys,)))
    return "_".join(f"{col}={int(key_map[col])}" for col in spec_cols)


def _spec_title(keys, spec_cols):
    if not spec_cols:
        return ""
    key_map = dict(zip(spec_cols, keys if isinstance(keys, tuple) else (keys,)))
    return (
        rf"$n={int(key_map['n'])}$, "
        rf"$m_\mathrm{{eval}}={int(key_map['n_eval_source'])}$, "
        rf"$m_\mathrm{{solve}}={int(key_map['n_solve'])}$"
    )


def _size_lookup(size):
    if size is None or size.empty:
        return {}
    lookup = {}
    spec_cols = _present_spec_cols(size)
    for _, row in size.iterrows():
        spec_key = tuple(row[col] for col in spec_cols)
        lookup[(spec_key, row["null_name"], row["statistic"])] = row["estimate"]
    return lookup


def plot_gof(outdir="results_gof"):
    _set_plot_style()
    outdir = Path(outdir)
    summary_path = outdir / "summary" / "power_summary.csv"
    size_path = outdir / "summary" / "size_summary.csv"
    figdir = ensure_dir(outdir / "figs")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing GOF power summary: {summary_path}")

    power = pd.read_csv(summary_path)
    if power.empty:
        raise ValueError(f"GOF power summary is empty: {summary_path}")

    size = pd.read_csv(size_path) if size_path.exists() else pd.DataFrame()
    size_lookup = _size_lookup(size)
    spec_cols = _present_spec_cols(power)
    grouped_power = power.groupby(spec_cols, dropna=False) if spec_cols else [((), power)]

    for spec_keys, spec_power in grouped_power:
        spec_key_tuple = spec_keys if isinstance(spec_keys, tuple) else (spec_keys,)
        spec_suffix = _spec_suffix(spec_key_tuple, spec_cols)
        # spec_title = _spec_title(spec_key_tuple, spec_cols)

        for null_name in GOF_NULLS:
            sub_null = spec_power[spec_power["null_name"] == null_name]
            if sub_null.empty:
                continue

            alpha = None
            if "alpha" in sub_null.columns and sub_null["alpha"].notna().any():
                alpha = float(sub_null["alpha"].dropna().iloc[0])
            elif size is not None and not size.empty and "alpha" in size.columns and size["alpha"].notna().any():
                alpha = float(size["alpha"].dropna().iloc[0])
            if alpha is None:
                alpha = 0.05

            fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), sharey=True)
            for ax, alt_type in zip(axes, ALT_LEVELS):
                sub_alt = sub_null[sub_null["alt_type"] == alt_type]
                ax.axhline(
                    alpha,
                    color="0.35",
                    linestyle="--",
                    linewidth=1.3,
                    label=rf"$\alpha={alpha:.2f}$",
                )
                for statistic, label in STAT_LABELS.items():
                    sub = sub_alt[sub_alt["statistic"] == statistic].sort_values("level")
                    if sub.empty:
                        continue

                    size_estimate = size_lookup.get((spec_key_tuple, null_name, statistic))
                    if size_estimate is not None and np.isfinite(size_estimate):
                        label = rf"{label}, $\widehat{{\alpha}}_0={size_estimate:.2f}$"

                    style = STAT_STYLE[statistic]
                    yerr = None
                    if "mc_se" in sub:
                        yerr = 1.96 * sub["mc_se"].to_numpy()
                    ax.errorbar(
                        sub["level"],
                        sub["estimate"],
                        yerr=yerr,
                        color=style["color"],
                        marker=style["marker"],
                        markersize=5,
                        linewidth=1.8,
                        elinewidth=1.4,
                        capsize=4,
                        capthick=1.2,
                        label=label,
                    )

                # ax.set_title(ALT_TITLES[alt_type])
                # ax.set_xlabel(r"$\mathrm{Perturbation\ level}$")
                ax.set_xlabel(ALT_TITLES[alt_type])
                ax.set_ylim(-0.02, 1.02)
                ax.set_yticks(np.linspace(0.0, 1.0, 6))
                ax.grid(True, color="0.88", linewidth=0.8)

            axes[0].set_ylabel(r"$\mathrm{Empirical\ rejection\ probability}$")
            handles, labels = axes[-1].get_legend_handles_labels()
            axes[-1].legend(
                handles,
                labels,
                loc="lower right",
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.4,
                handletextpad=0.45,
                labelspacing=0.35,
            )
            # title = NULL_TITLES.get(null_name, null_name.replace("_", " ").title())
            # fig.suptitle(title, fontsize=18)
            fig.tight_layout()

            suffix = f"_{spec_suffix}" if spec_suffix else ""
            pdf_path = figdir / f"power_{null_name}_1x3{suffix}.pdf"
            png_path = figdir / f"power_{null_name}_1x3{suffix}.png"
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
