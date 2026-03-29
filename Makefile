PYTHON ?= python3
PIP ?= uv pip

.PHONY: help install-dev install-apple compile lint typecheck test test-unit test-integration static-check build-dist validate-local clean

help:
	@printf "Targets:\n"
	@printf "  install-dev       Install editable package with dev extras\n"
	@printf "  install-apple     Install editable package with Apple Silicon MLX extras\n"
	@printf "  compile           Compile all source and test modules\n"
	@printf "  lint              Run Ruff linting and formatting via Nox\n"
	@printf "  typecheck         Run Mypy type validation via Nox\n"
	@printf "  test              Run unit tests locally (or across Python matrix implicitly)\n"
	@printf "  test-unit         Same as test\n"
	@printf "  test-integration  Run integration tests (requires MLX)\n"
	@printf "  static-check      Run linting, typechecking, and preflight script\n"
	@printf "  build-dist        Build wheel and sdist\n"
	@printf "  validate-local    Run Apple Silicon runtime validation script\n"
	@printf "  clean             Remove build artifacts\n"

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

test:
	nox -s tests

test-unit:
	nox -s tests -- tests/unit/

test-integration:
	pytest tests/integration/

static-check: lint typecheck
	$(PYTHON) scripts/preflight.py

build-dist:
	$(PYTHON) -m build

validate-local:
	./scripts/validate_apple_silicon.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .nox .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

