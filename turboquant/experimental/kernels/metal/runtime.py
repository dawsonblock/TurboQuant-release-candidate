import os
from pathlib import Path
import mlx.core as mx

_kernel_source = """
    uint gid = thread_position_in_grid.x;
    
    int group_size_val = group_size;
    int bits_val = bits;
    int topk_val = topk;
    
    int g = gid / group_size_val;

    // unpack
    int elements_in_uint = 32 / bits_val;
    uint word = packed[gid / elements_in_uint];
    int shift = (gid % elements_in_uint) * bits_val;
    uint code = (word >> shift) & ((1 << bits_val) - 1);

    float scale = (float)scales[g];
    float val = (float(code) * scale);

    // residual add
    int local_idx = gid % group_size_val;
    for (int i = 0; i < topk_val; i++) {
        if (resid_idx[g * topk_val + i] == local_idx) {
            val += (float)resid_vals[g * topk_val + i];
        }
    }

    out[gid] = half(val);
"""

def decode_k_metal(
    packed_k: mx.array,
    scales: mx.array,
    resid_vals: mx.array,
    resid_idx: mx.array,
    config,
    d_head: int,
):
    threadgroup_size = int(os.getenv("TQ_THREADGROUP_SIZE", "256"))

    if resid_vals is None:
        resid_vals = mx.zeros((1,), dtype=mx.float16)
        resid_idx = mx.zeros((1,), dtype=mx.uint16)

    bits_arg = mx.array(config.k_bits, dtype=mx.int32)
    group_size_arg = mx.array(config.k_group_size, dtype=mx.int32)
    topk_arg = mx.array(config.residual_topk, dtype=mx.int32)

    total_elements = scales.size * config.k_group_size
    grid = (total_elements, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    kernel = mx.fast.metal_kernel(
        name="decode_k",
        input_names=["packed", "scales", "resid_idx", "resid_vals", "bits", "group_size", "topk"],
        output_names=["out"],
        source=_kernel_source,
    )

    out_shape = packed_k.shape[:-1] + (d_head,)
    
    out = kernel(
        inputs=[packed_k, scales, resid_idx, resid_vals, bits_arg, group_size_arg, topk_arg],
        template=[], 
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[out_shape], 
        output_dtypes=[mx.float16],
        stream=mx.gpu
    )

    return out[0]


