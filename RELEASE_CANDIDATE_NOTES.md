# Release-candidate scaffolding notes

This pass does not claim MLX runtime certification. It adds repo-level controls that make the current supported slice easier to validate and harder to misread.

Added:
- `Makefile` with repeatable developer targets
- `scripts/preflight.py` for generic non-MLX runners
- `docs/supported-surface.md`
- `docs/release-checklist.md`
- `.github/workflows/package-build.yml`

Adjusted:
- `pyproject.toml` version bumped to `0.2.2`
- `README.md` tightened to remove misleading badge-style confidence
- `docs/validation-local.md` now points to preflight and Make targets
