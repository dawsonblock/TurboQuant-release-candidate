PYTHON ?= python3
PIP ?= uv pip

.PHONY: help install-dev install-apple compile lint typecheck test test-static test-mlx certify-apple-runtime build-dist validate-apple clean

help:
	@printf "Targets:\n"
	@printf "  install-dev              Install editable package with dev extras\n"
	@printf "  install-apple            Install editable package with Apple Silicon MLX extras\n"
	@printf "  compile                  Compile all source and test modules\n"
	@printf "  lint                     Run Ruff linting and formatting via Nox\n"
	@printf "  typecheck                Run Mypy type validation via Nox\n"
	@printf "  test                     Alias for test-static (safe on any platform)\n"
	@printf "  test-static              Run static unit tests (no MLX required)\n"
	@printf "  test-mlx                 Run MLX integration tests (Apple Silicon only)\n"
	@printf "  certify-apple-runtime    Full Apple-Silicon runtime certification\n"
	@printf "  build-dist               Build wheel and sdist\n"
	@printf "  validate-apple           Run Apple Silicon runtime validation script\n"
	@printf "  clean                    Remove build artifacts\n"

install-dev:
	$(PIP) install -e '.[dev]'

install-apple:
	$(PIP) install -e '.[apple,dev]'

compile:
	$(PYTHON) -m compileall turboquant mlx_lm tests

lint:
	nox -s lint

typecheck:
	nox -s typecheck

test: test-static

test-static:
	nox -s tests_static

test-mlx:
	nox -s tests_mlx

certify-apple-runtime:
	./scripts/certify_apple_runtime.sh

build-dist:
	$(PYTHON) -m build

validate-apple:
	./scripts/validate_apple_silicon.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .nox .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

