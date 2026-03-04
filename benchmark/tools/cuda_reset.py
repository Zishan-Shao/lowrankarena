#!/usr/bin/env python3
"""
CUDA GPU reset utility.
Used to reset GPU state after CUDA errors occur.
"""

import torch
import gc
import sys

def reset_cuda():
    """Reset CUDA state."""
    print("[cuda_reset] Resetting CUDA state...")

    if not torch.cuda.is_available():
        print("[cuda_reset] CUDA not available, skipping")
        return

    try:
        # 1. Clear cache
        torch.cuda.empty_cache()

        # 2. Synchronize all CUDA operations
        torch.cuda.synchronize()

        # 3. Reset peak memory statistics
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # 4. Garbage collection
        gc.collect()

        # 5. Show current state
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
