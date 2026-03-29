import inspect

import mlx.core as mx

with open("mlxf.txt", "w") as f:
    f.write(str(inspect.signature(mx.fast.metal_kernel)))
    f.write("\n\n")
    f.write(mx.fast.metal_kernel.__doc__ or "No doc")
