"""
TurboQuant — production KV-cache compression for MLX/Apple-Silicon LLMs.

Public API
----------
TurboQuantConfig          — runtime-immutable configuration
KVCompressor              — drop-in KV cache with compress/decompress
TurboQuantPipeline        — low-level encode/decode pipeline
calibrate                 — calibration pass over representative data
"""
from turboquant.calibration.fit_quantizer import calibrate
from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.runtime.kv_interface import KVCompressor

__all__ = [
    "TurboQuantConfig",
    "TurboQuantPipeline",
    "KVCompressor",
    "calibrate",
]

__version__ = "0.2.1"
