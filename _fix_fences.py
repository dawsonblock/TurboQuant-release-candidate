#!/usr/bin/env python3
"""Fix broken closing fences in README.md. Delete after use."""
import pathlib

TICK = "\x60" * 3  # three backticks

p = pathlib.Path(__file__).parent / "README.md"
content = p.read_text()
lines = content.split("\n")
new_lines = []
in_fence = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith(TICK):
        if in_fence:
            # Closing fence — strip any trailing language tag
            new_lines.append(TICK)
            in_fence = False
        else:
            # Opening fence — keep as-is (may have language tag)
            new_lines.append(line)
            in_fence = True
    else:
        new_lines.append(line)

result = "\n".join(new_lines)
p.write_text(result)

# Verify
broken_tick_text = TICK + "text"
broken = [
    i
    for i, l in enumerate(result.split("\n"), 1)
    if l.strip() == broken_tick_text
]
print(f"Remaining broken fences: {broken}")
print(f"Total lines: {result.count(chr(10))}")
