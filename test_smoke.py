#!/usr/bin/env python
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otexp.gof import (
    compute_gof_statistics,
    default_gof_config,
    prepare_gof_references,
)
from otexp.sampling import sample_null


def main():
    with tempfile.TemporaryDirectory(prefix="otexp_smoke_", dir="/private/tmp") as tmpdir:
        config = default_gof_config(
            n=8,
            n_solve=32,
            n_eval_source=32,
            seed=2026,
            max_iter=20,
            chunk_size=16,
            outdir=tmpdir,
            overwrite=True,
            use_qmc_source=True,
        )

        references = prepare_gof_references("uniform_ball", config)
        spec = (
            f"n={config['n']}"
            f"_eval={config['n_eval_source']}"
            f"_solve={config['n_solve']}"
            f"_seed={config['seed']}"
        )
        ref_path = Path(tmpdir) / "references" / spec / "null=uniform_ball" / "references.npz"
        meta_path = Path(tmpdir) / "references" / spec / "null=uniform_ball" / "metadata.json"

        saved = np.load(ref_path)
        metadata = json.load(open(meta_path))

        assert "X_solve" in saved.files
        assert "X_eval" in saved.files
        assert "Y_mmd" not in saved.files
        assert metadata["mmd_reference"] == "X_eval"
        assert metadata["n"] == config["n"]
        assert metadata["n_eval_source"] == config["n_eval_source"]
        assert metadata["n_solve"] == config["n_solve"]
        assert saved["X_eval"].shape == (config["n_eval_source"], 3)

        rng = np.random.default_rng(2027)
        sample = sample_null("uniform_ball", config["n"], rng=rng)
        stats = compute_gof_statistics(sample, "uniform_ball", references, config)

        for name in ("potential", "w2", "mmd"):
            value = stats[name]
            assert np.isfinite(value), f"{name} is not finite: {value}"

        diagnostics = stats["diagnostics"]
        assert "success" in diagnostics
        assert "grad_inf" in diagnostics

        print("smoke references")
        print(f"  spec:           {spec}")
        print(f"  reference keys: {sorted(saved.files)}")
        print(f"  mmd_reference:  {metadata['mmd_reference']}")
        print("smoke statistics")
        print(f"  potential: {stats['potential']:.12g}")
        print(f"  w2:        {stats['w2']:.12g}")
        print(f"  mmd:       {stats['mmd']:.12g}")
        print("test_smoke.py passed.")


if __name__ == "__main__":
    main()
