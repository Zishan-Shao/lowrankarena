#!/usr/bin/env python3
"""
Check whether all required dependencies are installed.
"""

import sys

def check_dependency(module_name, import_name=None):
    """Check a single dependency."""
    if import_name is None:
        import_name = module_name

    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {module_name:20} {version}")
        return True
    except ImportError as e:
        print(f"✗ {module_name:20} NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("SVD-Benchmark Dependency Check")
    print("=" * 60)
    print()

    dependencies = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("evaluate", "evaluate"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("tqdm", "tqdm"),
        ("triton", "triton"),
    ]

    print("Python version:")
    print(f"  {sys.version}")
    print()

    print("Package dependency check:")
    all_ok = True
    for module_name, import_name in dependencies:
        if not check_dependency(module_name, import_name):
            all_ok = False

    print()
    print("=" * 60)

    if all_ok:
        print("✓ All dependencies are installed!")
        print()

        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
            else:
                print("⚠ CUDA not available (CPU-only mode)")
        except:
            pass

        print()
        return 0
    else:
        print("✗ Missing dependencies! Please run:")
        print("  pip install -r requirements.txt")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
