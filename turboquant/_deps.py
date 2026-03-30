"""
Runtime dependency helpers for TurboQuant.

These utilities let the package import cleanly on any platform while
giving clear errors when MLX-dependent features are actually invoked.
"""

from __future__ import annotations

import platform
import importlib


def has_mlx() -> bool:
    """Return True if ``mlx`` is importable."""
    try:
        importlib.import_module("mlx")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def is_apple_silicon() -> bool:
    """Return True when running on macOS arm64 (Apple Silicon)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def require_mlx(feature: str = "this feature") -> None:
    """Raise ``ImportError`` with a clear message if MLX is missing.

    Parameters
    ----------
    feature
        Human-readable label inserted into the error message, e.g.
        ``"KVCompressor"`` or ``"calibrate()"``.
    """
    if not has_mlx():
        raise ImportError(
            f"TurboQuant: {feature} requires the `mlx` package, which is only "
            "available on Apple Silicon macOS.  Install it with:\n\n"
            "  pip install 'turboquant[apple]'\n\n"
            "For packaging and static tests on non-Apple platforms, "
            "use the `turboquant.config` module only."
        )
