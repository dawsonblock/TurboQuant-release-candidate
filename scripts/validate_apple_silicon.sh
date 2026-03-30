#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools nox
python -m pip install -e '.[dev,apple]'

echo "Running strict preflight..."
python3 scripts/preflight.py --strict

echo "Running MLX tests via Nox..."
nox -s tests_mlx
