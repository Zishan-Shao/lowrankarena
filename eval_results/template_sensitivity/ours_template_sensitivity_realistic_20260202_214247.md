# Template Stability Evaluation

Model: /home/zs89/FlashSVDTrain/checkpoints/llama_2_7b_hf_act_lora_mixedwhiten_mixedlora_0.4_enhanced.pt
Device: cuda
Batch size: 8
Template profile: realistic
Dtype: bfloat16

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 39.70 | 17.17 | 44.43 | 13.66 |
| qa | 41.16 | 14.55 | 41.61 | 12.59 |
| mc_letters | 43.24 | 14.05 | 43.24 | 14.05 |
| instruction | 42.92 | 14.54 | 43.77 | 13.89 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.20 | 35.40 |
| arc_easy | 52.46 | 46.32 |
| arc_challenge | 27.42 | 34.11 |
| winogrande | 50.20 | 50.51 |
| hellaswag | 39.62 | 49.18 |
| piqa | 68.61 | 70.46 |
| mathqa | 17.36 | 25.01 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 44.20 | 48.40 |
| arc_easy | 56.32 | 48.42 |
| arc_challenge | 29.43 | 30.77 |
| winogrande | 49.49 | 50.36 |
| hellaswag | 27.25 | 27.83 |
| piqa | 61.26 | 60.50 |
| mathqa | 20.20 | 24.96 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 48.60 | 48.60 |
| arc_easy | 57.72 | 57.72 |
| arc_challenge | 40.13 | 40.13 |
| winogrande | 51.30 | 51.30 |
| hellaswag | 25.03 | 25.03 |
| piqa | 58.87 | 58.87 |
| mathqa | 21.03 | 21.03 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 46.80 | 50.20 |
| arc_easy | 59.47 | 59.47 |
| arc_challenge | 36.79 | 36.79 |
| winogrande | 49.80 | 50.28 |
| hellaswag | 25.92 | 24.98 |
| piqa | 60.94 | 59.85 |
| mathqa | 20.74 | 24.80 |
