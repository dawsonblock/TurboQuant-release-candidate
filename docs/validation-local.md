# Local validation on Apple Silicon

Public CI in this repository only checks packaging and syntax. It does **not** certify the MLX runtime path, because GitHub-hosted runners are not Apple Silicon and do not provide a usable `mlx` environment.

## Two-track testing model

| Track | What it tests | Where it runs |
|---|---|---|
| **Static** (`make test-static`) | Import smoke, version consistency, config schema | Any platform (CI + local) |
| **MLX** (`make test-mlx`) | KVCompressor, pipeline, calibration, streaming attention | Apple Silicon only |

## Quick start

```bash
# Static tests (safe everywhere)
make test-static

# Full Apple Silicon validation
./scripts/validate_apple_silicon.sh
```

The validation script:

- creates a fresh virtualenv
- installs the package in editable mode with `.[dev,apple]`
- compiles the source tree
- runs both static and MLX-dependent test suites

## Manual smoke test

For manual model smoke tests, run dense generation first, then the TurboQuant upgrade path on the same prompt and compare stability, memory use, and throughput.

## Makefile targets

```bash
make test-static      # nox -s tests_static (no MLX needed)
make test-mlx         # nox -s tests_mlx   (Apple Silicon only)
make validate-apple   # ./scripts/validate_apple_silicon.sh
```

