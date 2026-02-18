#!/usr/bin/env python3
"""
检查所有必需的依赖是否已安装
"""

import sys

def check_dependency(module_name, import_name=None):
    """检查单个依赖"""
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
    print("SVD-Benchmark 依赖检查")
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

    print("Python 版本:")
    print(f"  {sys.version}")
    print()

    print("依赖包检查:")
    all_ok = True
    for module_name, import_name in dependencies:
        if not check_dependency(module_name, import_name):
            all_ok = False

    print()
    print("=" * 60)

    if all_ok:
        print("✓ 所有依赖已安装!")
        print()

        # 检查 CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA 版本: {torch.version.cuda}")
            else:
                print("⚠ CUDA 不可用 (仅 CPU 模式)")
        except:
            pass

        print()
        return 0
    else:
        print("✗ 缺少依赖! 请运行:")
        print("  pip install -r requirements.txt")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
