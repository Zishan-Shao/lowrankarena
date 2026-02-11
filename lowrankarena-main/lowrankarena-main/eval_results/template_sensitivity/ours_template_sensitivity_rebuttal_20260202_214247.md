# Template Stability Evaluation

Model: /home/zs89/FlashSVDTrain/checkpoints/llama_2_7b_hf_act_lora_mixedwhiten_mixedlora_0.4_enhanced.pt
Device: cuda
Batch size: 8
Template profile: rebuttal
Dtype: bfloat16

Padding side: right
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 39.85 | 17.25 | 44.47 | 13.63 |
| qa | 41.45 | 14.98 | 41.48 | 12.76 |
| mc_letters | 41.75 | 13.49 | 41.75 | 13.49 |
| instruction | 42.79 | 14.43 | 43.61 | 13.86 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.80 | 36.00 |
| arc_easy | 52.46 | 46.32 |
| arc_challenge | 27.42 | 34.11 |
| winogrande | 50.20 | 49.49 |
| hellaswag | 39.52 | 49.08 |
| piqa | 69.26 | 71.00 |
| mathqa | 17.32 | 25.32 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 45.00 | 47.20 |
| arc_easy | 56.32 | 48.42 |
| arc_challenge | 29.43 | 30.77 |
| winogrande | 50.04 | 50.43 |
| hellaswag | 27.15 | 28.03 |
| piqa | 62.51 | 61.21 |
| mathqa | 19.73 | 24.31 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 45.20 | 45.20 |
| arc_easy | 57.72 | 57.72 |
| arc_challenge | 40.13 | 40.13 |
| winogrande | 48.62 | 48.62 |
| hellaswag | 24.95 | 24.95 |
| piqa | 55.77 | 55.77 |
| mathqa | 19.87 | 19.87 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 46.40 | 50.20 |
| arc_easy | 59.47 | 59.47 |
| arc_challenge | 36.79 | 36.79 |
| winogrande | 49.01 | 49.49 |
| hellaswag | 25.95 | 25.15 |
| piqa | 60.99 | 59.74 |
| mathqa | 20.94 | 24.45 |
