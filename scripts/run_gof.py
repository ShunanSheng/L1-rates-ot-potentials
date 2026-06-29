#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GOF_NULLS = ("uniform_ball", "truncated_gaussian", "truncated_elliptical_t")
ALT_LEVELS = {
    "location_shift": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    "scale": (1.025, 1.05, 1.075, 1.10, 1.15, 1.20),
    "mixture_contamination": (0.01, 0.02, 0.05, 0.10, 0.15, 0.20),
}


def _nulls_from_arg(value):
    return GOF_NULLS if value == "all" else (value,)


def _levels_for_power(alt, level):
    if level is not None:
        return (float(level),)
    return ALT_LEVELS[alt]


def main():
    parser = argparse.ArgumentParser(description="Run GOF experiments for simple nulls.")
    parser.add_argument("--mode", choices=["prepare_references", "calibration", "size", "power", "all"], required=True)
    parser.add_argument("--null", choices=[*GOF_NULLS, "all"], required=True)
    parser.add_argument("--alt", choices=list(ALT_LEVELS), default="location_shift")
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--B_cal", type=int, default=500)
    parser.add_argument("--N_size", type=int, default=500)
    parser.add_argument("--N_alt", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_solve", type=int, default=10000)
    parser.add_argument("--n_eval_source", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use_qmc_source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--chunk_size", type=int, default=2500)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--chunk_id", type=int, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="results_gof")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.d != 3:
        raise ValueError("The GOF experiment currently supports d=3 only.")
    if args.mode == "power" and args.null == "all":
        raise ValueError("--mode power requires a single --null value.")
    if (args.chunk_id is None) != (args.num_chunks is None):
        raise ValueError("--chunk_id and --num_chunks must be supplied together.")

    from otexp.gof import (
        default_gof_config,
        prepare_gof_references,
        run_gof_all,
        run_gof_calibration,
        run_gof_power,
        run_gof_size,
    )

    config = default_gof_config(
        d=args.d,
        n=args.n,
        B_cal=args.B_cal,
        N_size=args.N_size,
        N_alt=args.N_alt,
        alpha=args.alpha,
        n_solve=args.n_solve,
        n_eval_source=args.n_eval_source,
        seed=args.seed,
        use_qmc_source=args.use_qmc_source,
        max_iter=args.max_iter,
        chunk_size=args.chunk_size,
        gtol=args.gtol,
        ftol=args.ftol,
        outdir=args.outdir,
        overwrite=args.overwrite,
    )

    for null_name in _nulls_from_arg(args.null):
        print(f"[gof] mode={args.mode}, null={null_name}", flush=True)
        if args.mode == "prepare_references":
            prepare_gof_references(null_name, config)
        elif args.mode == "calibration":
            df = run_gof_calibration(null_name, config, chunk_id=args.chunk_id, num_chunks=args.num_chunks)
            print(df.groupby("statistic")["value"].describe().to_string())
        elif args.mode == "size":
            df = run_gof_size(null_name, config, chunk_id=args.chunk_id, num_chunks=args.num_chunks)
            print(df.groupby("statistic")["reject"].mean().to_string())
        elif args.mode == "power":
            for level in _levels_for_power(args.alt, args.level):
                print(f"[gof] power alt={args.alt}, level={level}", flush=True)
                df = run_gof_power(
                    null_name,
                    args.alt,
                    level,
                    config,
                    chunk_id=args.chunk_id,
                    num_chunks=args.num_chunks,
                )
                print(df.groupby("statistic")["reject"].mean().to_string())
        elif args.mode == "all":
            run_gof_all(null_name, config)


if __name__ == "__main__":
    main()
