# Template Stability Evaluation

Model: Qinsi1/DobiSVD-Llama-2-7b-hf-0.4
Device: cuda
Batch size: 8
Template profile: rebuttal
Dtype: bfloat16

Padding side: right
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 30.32 | 14.33 | 32.47 | 11.66 |
| qa | 30.50 | 14.20 | 33.14 | 11.59 |
| mc_letters | 31.28 | 12.47 | 31.28 | 12.47 |
| instruction | 30.62 | 14.46 | 32.58 | 12.39 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 17.40 | 32.80 |
| arc_easy | 27.19 | 27.54 |
| arc_challenge | 22.41 | 24.08 |
| winogrande | 50.75 | 50.43 |
| hellaswag | 25.70 | 24.51 |
| piqa | 53.43 | 49.24 |
| mathqa | 15.33 | 18.68 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 17.60 | 33.60 |
| arc_easy | 27.02 | 27.37 |
| arc_challenge | 23.75 | 27.42 |
| winogrande | 50.59 | 50.51 |
| hellaswag | 25.62 | 24.37 |
| piqa | 53.54 | 50.05 |
| mathqa | 15.35 | 18.66 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 25.20 | 25.20 |
| arc_easy | 26.67 | 26.67 |
| arc_challenge | 24.41 | 24.41 |
| winogrande | 50.43 | 50.43 |
| hellaswag | 24.75 | 24.75 |
| piqa | 50.49 | 50.49 |
| mathqa | 16.98 | 16.98 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.80 | 29.20 |
| arc_easy | 24.04 | 24.04 |
| arc_challenge | 27.76 | 27.76 |
| winogrande | 50.99 | 51.54 |
| hellaswag | 25.67 | 24.85 |
| piqa | 53.92 | 51.63 |
| mathqa | 15.17 | 19.02 |
