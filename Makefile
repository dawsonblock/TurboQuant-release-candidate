PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help install-dev install-apple compile static-check build-dist test-unit test-integration validate-local clean

help:
	@printf "Targets:\n"
	@printf "  install-dev       Install editable package with dev extras\n"
	@printf "  install-apple     Install editable package with Apple Silicon MLX extras\n"
	@printf "  compile           Compile all source and test modules\n"
	@printf "  static-check      Run non-MLX preflight checks\n"
	@printf "  build-dist        Build wheel and sdist\n"
	@printf "  test-unit         Run unit tests (requires MLX)\n"
	@printf "  test-integration  Run integration tests (requires MLX)\n"
	@printf "  validate-local    Run Apple Silicon runtime validation script\n"
	@printf "  clean             Remove build artifacts\n"

install-dev:
	$(PIP) install -e '.[dev]'

install-apple:
	$(PIP) install -e '.[apple,dev]'

compile:
	$(PYTHON) -m compileall turboquant mlx_lm tests

static-check:
	$(PYTHON) scripts/preflight.py

build-dist:
	$(PYTHON) -m build

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

validate-local:
	./scripts/validate_apple_silicon.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
