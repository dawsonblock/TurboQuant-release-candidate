from typing import Optional
import mlx.core as mx
from turboquant.core.quantizer import dequantize_groups
from turboquant.core.residual import decode_topk_residual

def decode_k_block(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: Optional[mx.array],
    resid_idx: Optional[mx.array],
    config,
    d_head: int,
) -> mx.array:
    \"\"\"
    Single entry point for all MLX-vectorized decode paths.
    Must return rotated K.
    \"\"\"
    return decode_k_fallback(packed_k, scales, resid_vals, resid_idx, config, d_head)

def decode_k_fallback(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: Optional[mx.array],
    resid_idx: Optional[mx.array],
    config,
    d_head: int,
) -> mx.array:
    d_pad = (
        (d_head + config.k_group_size - 1) // config.k_group_size * config.k_group_size
    )

    y_hat = dequantize_groups(
        packed_k, scales, config.k_bits, config.k_group_size, d_pad
    )

    if getattr(config, "mode", "research") != "fasfrom typing import Optional
imptoimport mlx.core as mx
fromotfrom turboquant.coreesidual = decode_topk_residual(resid_vals, resid_idx, conf
def decode_k_block(
    packed_k: mx.array,
    scales:., :d_pad]

    return y_hat[..., :d_head]
