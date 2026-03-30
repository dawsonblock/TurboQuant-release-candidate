"""
TurboQuant — production KV-cache compression for MLX/Apple-Silicon LLMs.

Public API
----------
TurboQuantConfig          — runtime-immutable configuration
KVCompressor              — drop-in KV cache with compress/decompress
TurboQuantPipeline        — low-level encode/decode pipeline
calibrate                 — calibration pass over representative data
"""

from turboquant.config import TurboQuantConfig
from turboquant._deps import has_mlx, is_apple_silicon, require_mlx, check_mlx_version  # noqa: F401

# Validate MLX version bounds at import time (no-op if MLX is absent)
check_mlx_version()

# Lazy imports for MLX-dependent runtime symbols
def __getattr__(name: str):
    if name == "calibrate":
        require_mlx("calibrate()")
        from turboquant.calibration.fit_quantizer import calibrate
        return calibrate
    elif name == "TurboQuantPipeline":
        require_mlx("TurboQuantPipeline")
        from turboquant.core.pipeline import TurboQuantPipeline
        return TurboQuantPipeline
    elif name == "KVCompressor":
        require_mlx("KVCompressor")
        from turboquant.runtime.kv_interface import KVCompressor
        return KVCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TurboQuantConfig",
    "TurboQuantPipeline",
    "KVCompressor",
    "calibrate",
]

__version__ = "0.2.2"
