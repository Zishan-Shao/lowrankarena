"""
Refactored AdaSVD implementation with FlashSVD support.

This module provides:
- adaptive_rank_selection: Train hypernetwork to generate per-operation ranks
- Naive and FlashSVD backends for inference
"""
