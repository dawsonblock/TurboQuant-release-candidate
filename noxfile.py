import nox

# Force nox to use uv as the default virtual environment backend
nox.options.default_venv_backend = "uv"

# Global options
nox.options.sessions = ["lint", "typecheck", "tests"]
nox.options.reuse_existing_virtualenvs = True

# List of python versions to matrix test
PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    # Install the package with test dependencies
    session.install(".[test]")

    # Run the tests using pytest
    session.run("pytest", "tests/unit/", *session.posargs)


@nox.session(python="3.11")  # Linting only needs to run on one version
def lint(session: nox.Session) -> None:
    """Run linting using ruff."""
    # Install ruff
    session.install("ruff")

    # Check linting and formatting
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    # Install mypy and package dependencies
    session.install("mypy", ".")

    # Target our primary packages
    session.run("mypy", "turboquant/", "mlx_lm/")
