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

# Lazy imports for MLX-dependent runtime symbols
def __getattr__(name: str):
    if name == "calibrate":
        from turboquant.calibration.fit_quantizer import calibrate
        return calibrate
    elif name == "TurboQuantPipeline":
        from turboquant.core.pipeline import TurboQuantPipeline
        return TurboQuantPipeline
    elif name == "KVCompressor":
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
