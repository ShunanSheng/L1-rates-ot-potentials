#!/bin/bash
set -euo pipefail

echo "Local tests are notebook-first. Opening notebooks/local_test.ipynb."
echo "Use the default parameter cell for the smoke test."

jupyter lab notebooks/local_test.ipynb
