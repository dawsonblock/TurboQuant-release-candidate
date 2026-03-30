import os
from pathlib import Path
import mlx.core as mx

_kernel_source = """
    uint gid = thread_position_in_grid.x;
    
    uint g = gid / GROUP_SIZE;

    // unpack
    uint elements_in_uint = 32u / BITS;
    uint word = packed[gid / elements_in_uint];
    uint shift = (gid % elements_in_uint) * BITS;
    uint code = (word >> shift) & ((1u << BITS) - 1u);

    float scale = (float)scales[g];
    float val = (float(code) * scale);

    // residual add
    uint local_idx = gid % GROUP_SIZE;

    #pragma unroll
    for (uint i = 0; i < TOPK; i++) {
        bool match = (resid_idx[g * TOPK + i] == local_idx);
        val += match ? (float)resid_vals[g * TOPK + i] : 0.0f;
    }

    out[gid] = half(val);
"""

_kernels = {}

def decode_k_metal(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: mx.array,
    resid_idx: mx.array,
    config,
    d_head: int,
):
    global _kernels
    
    cache_key = (config.k_bits, config.k_group_size, config.residual_topk)
    
    if cache_key not in _kernels:
        _kernels[cache_key] = mx.fast.metal_kernel(
            name="decode_k",
            input_names=["packed", "scales", "resid_idx", "resid_vals"],
            output_names=["out"],
            source=_kernel_source,
        )

    kernel = _kernels[cache_key]
    threadgroup_size = int(os.getenv("TQ_THREADGROUP_SIZE", "64"))

    if resid_vals is None:
        resid_vals = mx.zeros((1,), dtype=mx.float16)
        resid_idx = mx.zeros((1,), dtype=mx.uint16)

    total_elements = scales.size * config.k_group_size
    grid = (total_elements, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    out_shape = packed_k.shape[:-1] + (d_head,)
    
    out = kernel(
        inputs=[packed_k, scales, resid_idx, resid_vals],
        template=[("BITS", config.k_bits), ("GROUP_SIZE", config.k_group_size), ("TOPK", config.residual_topk)], 
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[out_shape], 
        output_dtypes=[mx.float16],
        stream=mx.gpu
    )

    return out[0]
