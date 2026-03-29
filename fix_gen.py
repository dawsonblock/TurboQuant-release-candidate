with open("mlx_lm/generate.py", "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    if "from integrations.mlx.cache_adapter import TurboQuantKCache" in line:
        continue
    out.append(line)

for i, line in enumerate(out):
    if line.strip() == "from .sample_utils import make_sampler":
        out.insert(i, "from integrations.mlx.cache_adapter import TurboQuantKCache\n")
        break

with open("mlx_lm/generate.py", "w") as f:
    f.writelines(out)
