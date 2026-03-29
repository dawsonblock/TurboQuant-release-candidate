with open("tests/integration/test_turboquant_generate.py", "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    clean = line.strip()
    if clean == ",":
        continue
    out.append(line)

with open("tests/integration/test_turboquant_generate.py", "w") as f:
    f.writelines(out)
