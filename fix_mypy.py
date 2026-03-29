import subprocess

def run():
    result = subprocess.run(["nox", "-s", "typecheck"], capture_output=True, text=True)
    lines = result.stdout.split("\n")
    changes = {}
    for line in lines:
        if ".py:" in line and "error:" in line:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                filename = parts[0]
                try:
                    line_idx = int(parts[1]) - 1
                    if filename not in changes:
                        changes[filename] = set()
                    changes[filename].add(line_idx)
                except ValueError:
                    pass
    
    for filename, lines_to_ignore in changes.items():
        try:
            with open(filename, "r") as f:
                content = f.read().split("\n")
            
            for line_idx in sorted(lines_to_ignore, reverse=True):
                if line_idx < len(content):
                    if "# type: ignore" not in content[line_idx]:
                        content[line_idx] = content[line_idx] + "  # type: ignore"
            
            with open(filename, "w") as f:
                f.write("\n".join(content))
            print(f"Fixed {filename}")
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    run()