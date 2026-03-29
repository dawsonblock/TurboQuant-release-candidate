#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e '.[dev,apple]'
python -m compileall turboquant mlx_lm tests
pytest tests/unit/test_rotation.py -q
pytest tests/unit/test_quantizer.py -q
pytest tests/unit/test_pipeline.py -q
pytest tests/unit/test_kv_interface.py -q
pytest tests/integration/test_turboquant_generate.py -q
pytest tests/integration/test_turboquant_e2e.py -q
