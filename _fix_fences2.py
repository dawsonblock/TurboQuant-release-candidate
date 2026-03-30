#!/usr/bin/env python3
"""Fix broken closing fences in Markdown files. Delete after use."""
import pathlib
import sys

TICK = "\x60" * 3  # three backticks

for filename in sys.argv[1:]:
    p = pathlib.Path(filename)
    content = p.read_text()
    lines = content.split("\n")
    new_lines = []
    in_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(TICK):
            if in_fence:
                new_lines.append(TICK)
                in_fence = False
            else:
                new_lines.append(line)
                in_fence = True
        else:
            new_lines.append(line)

    result = "\n".join(new_lines)
    p.write_text(result)
    print(f"Fixed {filename}")
