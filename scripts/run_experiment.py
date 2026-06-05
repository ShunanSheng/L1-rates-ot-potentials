#!/usr/bin/env python
import argparse

from otexp.experiment import run_ball_experiment


def main():
    parser = argparse.ArgumentParser(description="Run unit-ball empirical OT-potential experiment for one d.")
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n_target", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--n_source", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--use_qmc_source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--b_start", type=int, default=0)
    parser.add_argument("--b_stop", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

   
    df, slope_loss, slope_beta = run_ball_experiment(
        d=args.d,
        n_target=tuple(args.n_target),
        B=args.B,
        n_source=args.n_source,
        seed=args.seed,
        use_qmc_source=args.use_qmc_source,
        outdir=args.outdir,
        max_iter=args.max_iter,
        chunk_size=args.chunk_size,
        b_start=args.b_start,
        b_stop=args.b_stop,
        overwrite=args.overwrite,
    )

    print("\n Summary:")
    print(df.to_string(index=False))
    print()
    print(f"Empirical log-log slope for mean loss: {slope_loss:.3f}")
    print(f"Log-log slope of beta(n,{args.d}): {slope_beta:.3f}")


if __name__ == "__main__":
    main()
