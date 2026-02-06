# Template Stability Evaluation

Model: Qinsi1/DobiSVD-Llama-2-7b-hf-0.4
Device: cuda
Batch size: 8
Template profile: realistic
Dtype: bfloat16

Padding side: right
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.71 | 14.35 | 30.71 | 12.13 |
| qa | 29.87 | 14.30 | 30.96 | 11.80 |
| mc_letters | 30.53 | 12.55 | 30.53 | 12.55 |
| instruction | 30.04 | 14.22 | 31.36 | 11.85 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 22.60 |
| arc_easy | 24.74 | 22.28 |
| arc_challenge | 24.41 | 29.10 |
| winogrande | 50.20 | 49.96 |
| hellaswag | 25.77 | 24.94 |
| piqa | 52.67 | 48.48 |
| mathqa | 14.82 | 17.61 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.60 | 23.00 |
| arc_easy | 27.02 | 25.09 |
| arc_challenge | 23.08 | 27.09 |
| winogrande | 50.28 | 50.04 |
| hellaswag | 25.66 | 25.04 |
| piqa | 52.56 | 48.26 |
| mathqa | 14.91 | 18.19 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 25.20 | 25.20 |
| arc_easy | 20.70 | 20.70 |
| arc_challenge | 27.76 | 27.76 |
| winogrande | 49.57 | 49.57 |
| hellaswag | 24.82 | 24.82 |
| piqa | 49.62 | 49.62 |
| mathqa | 16.02 | 16.02 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 14.60 | 23.40 |
| arc_easy | 24.56 | 24.56 |
| arc_challenge | 28.76 | 28.76 |
| winogrande | 49.96 | 50.43 |
| hellaswag | 25.61 | 25.46 |
| piqa | 52.18 | 48.59 |
| mathqa | 14.59 | 18.30 |
