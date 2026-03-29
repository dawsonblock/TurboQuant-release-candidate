import mlx.core as mx
from typing import Optional

from turboquant.core.quantizer import dequantize_groups
from turboquant.core.residual import decode_topk_residual
from turboquant.kernels.metal.runtime import decode_k_metal

HAS_METAL_KERNEL = False  # Placeholder until fully linked via custom C++ extension

def decode_k_block(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: Optional[mx.array],
    resid_idx: Optional[mx.array],
    config,
    d_head: int,
) -> mx.array:
    """
    Single entry point for all decode paths.
    Must return rotated K.
    """
    mode = getattr(config, "mode", "research")
    if mode == "kernel" or HAS_METAL_KERNEL:
        return decode_k_metal(packed_k, scales, resid_vals, resid_idx, config, d_head)

    return decode_k_fallback(packed_k, scales, resid_vals, resid_idx, config, d_head)

def decode_k_fallback(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: Optional[mx.array],
    resid_idx: Optional[mx.array],
    config,
    d_head: int,
) -> mx.array:
    d_pad = (d_head + config.k_group_size - 1) // config.k_group_size * config.k_group_size
    
    y_hat = dequantize_groups(
        packed_k, scales, config.k_bits, config.k_group_size, d_pad
    )
    
    if getattr(config, "mode", "research") != "fast":
        if config.residual_topk > 0 and resid_vals is not None:
            residual = decode_topk_residual(
                resid_vals, resid_idx, config.k_group_size
            )
            y_hat = y_hat + residual[..., :d_pad]

    return y_hat[..., :d_head]
