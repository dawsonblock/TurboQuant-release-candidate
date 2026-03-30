"""
Durable state persistence for KV buffers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from turboquant.errors import TurboQuantStateError


def save_state(state: dict[str, Any], path: str | Path) -> None:
    """Save state robustly with write-then-rename."""
    raise NotImplementedError("Persistence API is not supported in the current TurboQuant release.")


def load_state(path: str | Path) -> dict[str, Any]:
    """Load and optionally verify checksum of the state."""
    raise NotImplementedError("Persistence API is not supported in the current TurboQuant release.")
