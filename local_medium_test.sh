#!/bin/bash
set -euo pipefail

echo "Local tests are notebook-first. Opening notebooks/local_test.ipynb."
echo "For a medium test, set n_target=(256,), B=3, n_source=4096, and max_iter=80 in the parameter cell."

jupyter lab notebooks/local_test.ipynb
