#include <metal_stdlib>
using namespace metal;

kernel void decode_k(
    device uint *packed [[buffer(0)]],
    device half *scales [[buffer(1)]],
    device ushort *resid_idx [[buffer(2)]],
    device half *resid_vals [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant int &bits [[buffer(5)]],
    constant int &group_size [[buffer(6)]],
    constant int &topk [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    // compute group index
    int g = gid / group_size;

    // unpack (example for 4-bit)
    uint word = packed[gid / 8];
    int shift = (gid % 8) * bits;
    uint code = (word >> shift) & ((1 << bits) - 1);

    float scale = scales[g];
    float val = (float(code) * scale);

    // residual add
    for (int i = 0; i < topk; i++) {
        if (resid_idx[g * topk + i] == (gid % group_size)) {
            val += resid_vals[g * topk + i];
        }
    }

    out[gid] = half(val);
}
