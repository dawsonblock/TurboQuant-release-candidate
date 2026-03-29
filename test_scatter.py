import mlx.core as mx

values = mx.array([[[1.0, 2.0]], [[3.0, 4.0]]]) # 1, 2, 2
indices = mx.array([[[0, 2]], [[1, 3]]])
g = 4
out = mx.zeros((*values.shape[:-1], g), dtype=values.dtype)
out = mx.put_along_axis(out, indices, values, axis=-1)
print(out)
