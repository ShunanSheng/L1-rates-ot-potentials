#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the full requested grid sequentially.")
    parser.add_argument("--d_list", type=int, nargs="+", default=[2, 3, 4, 10])
    parser.add_argument("--n_target", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--n_source", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--use_qmc_source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from otexp.experiment import run_ball_experiment

    for d in args.d_list:
        print(f"\n========== Starting d={d} ==========")
        run_ball_experiment(
            d=d,
            n_target=tuple(args.n_target),
            B=args.B,
            n_source=args.n_source,
            seed=args.seed,
            use_qmc_source=args.use_qmc_source,
            outdir=args.outdir,
            max_iter=args.max_iter,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
