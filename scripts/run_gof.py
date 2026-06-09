#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _float_tuple(values):
    return tuple(float(value) for value in values)


def _format_number_for_filename(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _theta_label(prefix, values):
    return f"{prefix}=" + "-".join(_format_number_for_filename(value) for value in values)


def main():
    parser = argparse.ArgumentParser(
        description="Run a goodness-of-fit experiment for the unit ball."
    )
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--location_thetas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--scale_thetas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--mixture_thetas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--mixture_shift", type=float, default=0.5)
    parser.add_argument("--n_source", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use_qmc_source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--outdir", type=str, default="results_gof")
    args = parser.parse_args()

    from otexp.experiment import run_gof_experiment
    from otexp.io import ensure_dir, save_csv_atomic

    location_thetas = _float_tuple(args.location_thetas)
    scale_thetas = _float_tuple(args.scale_thetas)
    mixture_thetas = _float_tuple(args.mixture_thetas)

    summary_df, calibration_null_df, sim_df = run_gof_experiment(
        d=args.d,
        n=args.n,
        B=args.B,
        n_eval=args.n_eval,
        alpha=args.alpha,
        location_thetas=location_thetas,
        scale_thetas=scale_thetas,
        mixture_thetas=mixture_thetas,
        mixture_shift=args.mixture_shift,
        n_source=args.n_source,
        seed=args.seed,
        use_qmc_source=args.use_qmc_source,
        max_iter=args.max_iter,
        chunk_size=args.chunk_size,
        gtol=args.gtol,
        ftol=args.ftol,
    )

    outdir = Path(args.outdir)
    summary_dir = ensure_dir(outdir / "summary")
    raw_dir = ensure_dir(outdir / "raw")
    theta_stem = "_".join([
        _theta_label("loc", location_thetas),
        _theta_label("scale", scale_thetas),
        _theta_label("mix", mixture_thetas),
        f"mixshift={_format_number_for_filename(args.mixture_shift)}",
    ])
    stem = (
        f"gof_d={args.d}_n={args.n}_B={args.B}_eval={args.n_eval}"
        f"_source={args.n_source}_seed={args.seed}_{theta_stem}"
    )

    summary_path = summary_dir / f"summary_{stem}.csv"
    calibration_path = raw_dir / f"calibration_null_{stem}.csv"
    simulation_path = raw_dir / f"simulation_{stem}.csv"

    save_csv_atomic(summary_df, summary_path)
    save_csv_atomic(calibration_null_df, calibration_path)
    save_csv_atomic(sim_df, simulation_path)

    print("\nGoodness-of-fit summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved summary to {summary_path}")
    print(f"Saved calibration null replicates to {calibration_path}")
    print(f"Saved simulation replicates to {simulation_path}")


if __name__ == "__main__":
    main()
