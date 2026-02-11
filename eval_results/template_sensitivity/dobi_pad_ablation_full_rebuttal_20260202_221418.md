# Template Stability Evaluation

Model: Qinsi1/DobiSVD-Llama-2-7b-hf-0.4
Device: cuda
Batch size: 8
Template profile: rebuttal
Dtype: bfloat16
Pad ablation: full

## Pad variant: left_no_fix
Padding side: left
fix_pad_query_mask: False

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.51 | 14.03 | 31.28 | 11.05 |
| qa | 29.28 | 14.04 | 31.38 | 11.03 |
| mc_letters | 31.58 | 12.05 | 31.58 | 12.05 |
| instruction | 29.89 | 13.45 | 31.14 | 11.37 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 27.60 |
| arc_easy | 28.42 | 25.79 |
| arc_challenge | 21.07 | 26.09 |
| winogrande | 48.70 | 48.86 |
| hellaswag | 25.63 | 25.03 |
| piqa | 52.23 | 47.50 |
| mathqa | 15.15 | 18.12 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 28.80 |
| arc_easy | 26.67 | 26.49 |
| arc_challenge | 21.07 | 24.75 |
| winogrande | 48.62 | 48.93 |
| hellaswag | 25.67 | 25.27 |
| piqa | 52.29 | 47.33 |
| mathqa | 15.24 | 18.08 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 50.43 | 50.43 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.49 | 50.49 |
| mathqa | 21.43 | 21.43 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 27.80 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 48.22 | 48.54 |
| hellaswag | 25.76 | 24.47 |
| piqa | 52.07 | 48.48 |
| mathqa | 15.58 | 17.68 |

## Pad variant: left_fix
Padding side: left
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.51 | 14.03 | 31.28 | 11.05 |
| qa | 29.28 | 14.04 | 31.38 | 11.03 |
| mc_letters | 31.58 | 12.05 | 31.58 | 12.05 |
| instruction | 29.89 | 13.45 | 31.14 | 11.37 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 27.60 |
| arc_easy | 28.42 | 25.79 |
| arc_challenge | 21.07 | 26.09 |
| winogrande | 48.70 | 48.86 |
| hellaswag | 25.63 | 25.03 |
| piqa | 52.23 | 47.50 |
| mathqa | 15.15 | 18.12 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 28.80 |
| arc_easy | 26.67 | 26.49 |
| arc_challenge | 21.07 | 24.75 |
| winogrande | 48.62 | 48.93 |
| hellaswag | 25.67 | 25.27 |
| piqa | 52.29 | 47.33 |
| mathqa | 15.24 | 18.08 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 50.43 | 50.43 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.49 | 50.49 |
| mathqa | 21.43 | 21.43 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 27.80 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 48.22 | 48.54 |
| hellaswag | 25.76 | 24.47 |
| piqa | 52.07 | 48.48 |
| mathqa | 15.58 | 17.68 |

## Pad variant: right_no_fix
Padding side: right
fix_pad_query_mask: False

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.51 | 14.03 | 31.28 | 11.05 |
| qa | 29.28 | 14.04 | 31.38 | 11.03 |
| mc_letters | 31.58 | 12.05 | 31.58 | 12.05 |
| instruction | 29.89 | 13.45 | 31.14 | 11.37 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 27.60 |
| arc_easy | 28.42 | 25.79 |
| arc_challenge | 21.07 | 26.09 |
| winogrande | 48.70 | 48.86 |
| hellaswag | 25.63 | 25.03 |
| piqa | 52.23 | 47.50 |
| mathqa | 15.15 | 18.12 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 28.80 |
| arc_easy | 26.67 | 26.49 |
| arc_challenge | 21.07 | 24.75 |
| winogrande | 48.62 | 48.93 |
| hellaswag | 25.67 | 25.27 |
| piqa | 52.29 | 47.33 |
| mathqa | 15.24 | 18.08 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 50.43 | 50.43 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.49 | 50.49 |
| mathqa | 21.43 | 21.43 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 27.80 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 48.22 | 48.54 |
| hellaswag | 25.76 | 24.47 |
| piqa | 52.07 | 48.48 |
| mathqa | 15.58 | 17.68 |

## Pad variant: right_fix
Padding side: right
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.51 | 14.03 | 31.28 | 11.05 |
| qa | 29.28 | 14.04 | 31.38 | 11.03 |
| mc_letters | 31.58 | 12.05 | 31.58 | 12.05 |
| instruction | 29.89 | 13.45 | 31.14 | 11.37 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 27.60 |
| arc_easy | 28.42 | 25.79 |
| arc_challenge | 21.07 | 26.09 |
| winogrande | 48.70 | 48.86 |
| hellaswag | 25.63 | 25.03 |
| piqa | 52.23 | 47.50 |
| mathqa | 15.15 | 18.12 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 15.40 | 28.80 |
| arc_easy | 26.67 | 26.49 |
| arc_challenge | 21.07 | 24.75 |
| winogrande | 48.62 | 48.93 |
| hellaswag | 25.67 | 25.27 |
| piqa | 52.29 | 47.33 |
| mathqa | 15.24 | 18.08 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 50.43 | 50.43 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.49 | 50.49 |
| mathqa | 21.43 | 21.43 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 27.80 |
| arc_easy | 24.91 | 24.91 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 48.22 | 48.54 |
| hellaswag | 25.76 | 24.47 |
| piqa | 52.07 | 48.48 |
| mathqa | 15.58 | 17.68 |
