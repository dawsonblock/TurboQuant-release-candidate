with open("turboquant/runtime/kv_interface.py") as f:
    lines = f.readlines()

out = []
for line in lines:
    out.append(line)
    if "def trim(self, n: int) -> int:" in line:
        prop = """
    @property
    def nbytes(self) -> int:
        return sum(
            getattr(self, k).nbytes
            for k in [
                "_k_packed", "_k_scales", "_resid_vals", "_resid_idx",
                "_v_packed", "_v_scales"
            ]
            if getattr(self, k) is not None
        )
"""
        out.insert(-1, prop)

with open("turboquant/runtime/kv_interface.py", "w") as f:
    f.writelines(out)
