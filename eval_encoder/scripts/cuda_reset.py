#!/usr/bin/env python3
"""
CUDA GPU 重置工具
用于在出现CUDA错误后重置GPU状态
"""

import torch
import gc
import sys

def reset_cuda():
    """重置CUDA状态"""
    print("[cuda_reset] Resetting CUDA state...")

    if not torch.cuda.is_available():
        print("[cuda_reset] CUDA not available, skipping")
        return

    try:
        # 1. 清空缓存
        torch.cuda.empty_cache()

        # 2. 同步所有CUDA操作
        torch.cuda.synchronize()

        # 3. 重置峰值内存统计
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # 4. 垃圾回收
        gc.collect()

        # 5. 显示当前状态
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"[cuda_reset] GPU {i}: {mem_allocated:.1f} MB allocated, {mem_reserved:.1f} MB reserved")

        print("[cuda_reset] ✅ CUDA reset complete")
        return True

    except Exception as e:
        print(f"[cuda_reset] ❌ Failed to reset CUDA: {e}")
        return False

if __name__ == "__main__":
    success = reset_cuda()
    sys.exit(0 if success else 1)
