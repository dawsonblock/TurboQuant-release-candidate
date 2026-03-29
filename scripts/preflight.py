#!/usr/bin/env python3
import platform
import sys


def main():
    print("Running TurboQuant Preflight Checks...")

    # Check Python Version
    py_version = sys.version_info
    if py_version < (3, 9):
        print("ERROR: Python >= 3.9 is required. You are running", sys.version)
        sys.exit(1)
    print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Check Platform
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    if not (is_mac and is_arm):
        print("WARNING: You are not running on Apple Silicon (macOS + arm64).")
        print("         MLX acceleration may be disabled or you may need to use CPU. Validation tests may fail.")
    else:
        print("✓ Platform is Apple Silicon (darwin-arm64)")

    # Check MLX
    try:
        import mlx.core as mx
        print("✓ MLX backend initialized. Default device:", mx.default_device())
    except ImportError:
        print("ERROR: Could not import `mlx.core`. Ensure mlx is installed.")
        sys.exit(1)

    print("\nPreflight checks passed! Readiness confirmed.")

if __name__ == "__main__":
    main()
