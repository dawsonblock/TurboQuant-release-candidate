#!/usr/bin/env bash
set -e

echo "Running TurboQuant Local Validation..."

# 1. Preflight
python3 scripts/preflight.py

# 2. Run unit tests
echo "\nRunning unit tests..."
python3 -m pytest tests/unit/

# 3. Run integration tests
echo "\nRunning integration tests..."
python3 -m pytest tests/integration/

echo "\nAll local validation checks passed."
