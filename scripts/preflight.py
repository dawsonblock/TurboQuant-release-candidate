#!/usr/bin/env python3
import platform
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Preflight Checks")
    parser.add_argument("--strict", action="store_true", help="Fail if MLX or Apple Silicon requirements are missing")
    args = parser.parse_args()

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
        print("         MLX acceleration will be disabled or unsupported. Validation tests may fail.")
        if args.strict:
            print("ERROR: Strict mode requires Apple Silicon.")
            sys.exit(1)
    else:
        print("✓ Platform is Apple Silicon (darwin-arm64)")

    # Check MLX
    try:
        import mlx.core as mx

        print("✓ MLX backend initialized. Default device:", mx.default_device())
    except ImportError:
        print("WARNING: Could not import `mlx.core`. Expected if running off-platform for static checks.")
        if args.strict:
            print("ERROR: Strict mode requires MLX. Ensure mlx is installed.")
            sys.exit(1)

    print("\nPreflight checks complete.")


if __name__ == "__main__":
    main()
