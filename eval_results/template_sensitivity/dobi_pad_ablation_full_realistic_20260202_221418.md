# Template Stability Evaluation

Model: Qinsi1/DobiSVD-Llama-2-7b-hf-0.4
Device: cuda
Batch size: 8
Template profile: realistic
Dtype: bfloat16
Pad ablation: full

## Pad variant: left_no_fix
Padding side: left
fix_pad_query_mask: False

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.22 | 14.27 | 30.88 | 12.94 |
| qa | 29.54 | 14.19 | 32.43 | 12.11 |
| mc_letters | 31.41 | 13.15 | 31.41 | 13.15 |
| instruction | 29.76 | 14.15 | 31.50 | 12.08 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.40 | 26.20 |
| arc_easy | 27.89 | 24.39 |
| arc_challenge | 17.73 | 17.73 |
| winogrande | 49.49 | 50.04 |
| hellaswag | 25.70 | 25.59 |
| piqa | 51.90 | 51.69 |
| mathqa | 15.44 | 20.56 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 26.40 |
| arc_easy | 27.54 | 25.61 |
| arc_challenge | 19.40 | 25.75 |
| winogrande | 49.64 | 50.36 |
| hellaswag | 25.81 | 25.96 |
| piqa | 52.34 | 52.34 |
| mathqa | 15.46 | 20.60 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 28.07 | 28.07 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 52.17 | 52.17 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.44 | 50.44 |
| mathqa | 15.35 | 15.35 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.20 | 24.60 |
| arc_easy | 27.72 | 27.72 |
| arc_challenge | 21.40 | 21.40 |
| winogrande | 50.20 | 49.88 |
| hellaswag | 25.51 | 25.96 |
| piqa | 52.07 | 50.60 |
| mathqa | 15.24 | 20.36 |

## Pad variant: left_fix
Padding side: left
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.22 | 14.27 | 30.88 | 12.94 |
| qa | 29.54 | 14.19 | 32.43 | 12.11 |
| mc_letters | 31.41 | 13.15 | 31.41 | 13.15 |
| instruction | 29.76 | 14.15 | 31.50 | 12.08 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.40 | 26.20 |
| arc_easy | 27.89 | 24.39 |
| arc_challenge | 17.73 | 17.73 |
| winogrande | 49.49 | 50.04 |
| hellaswag | 25.70 | 25.59 |
| piqa | 51.90 | 51.69 |
| mathqa | 15.44 | 20.56 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 26.40 |
| arc_easy | 27.54 | 25.61 |
| arc_challenge | 19.40 | 25.75 |
| winogrande | 49.64 | 50.36 |
| hellaswag | 25.81 | 25.96 |
| piqa | 52.34 | 52.34 |
| mathqa | 15.46 | 20.60 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 28.07 | 28.07 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 52.17 | 52.17 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.44 | 50.44 |
| mathqa | 15.35 | 15.35 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.20 | 24.60 |
| arc_easy | 27.72 | 27.72 |
| arc_challenge | 21.40 | 21.40 |
| winogrande | 50.20 | 49.88 |
| hellaswag | 25.51 | 25.96 |
| piqa | 52.07 | 50.60 |
| mathqa | 15.24 | 20.36 |

## Pad variant: right_no_fix
Padding side: right
fix_pad_query_mask: False

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.22 | 14.27 | 30.88 | 12.94 |
| qa | 29.54 | 14.19 | 32.43 | 12.11 |
| mc_letters | 31.41 | 13.15 | 31.41 | 13.15 |
| instruction | 29.76 | 14.15 | 31.50 | 12.08 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.40 | 26.20 |
| arc_easy | 27.89 | 24.39 |
| arc_challenge | 17.73 | 17.73 |
| winogrande | 49.49 | 50.04 |
| hellaswag | 25.70 | 25.59 |
| piqa | 51.90 | 51.69 |
| mathqa | 15.44 | 20.56 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 26.40 |
| arc_easy | 27.54 | 25.61 |
| arc_challenge | 19.40 | 25.75 |
| winogrande | 49.64 | 50.36 |
| hellaswag | 25.81 | 25.96 |
| piqa | 52.34 | 52.34 |
| mathqa | 15.46 | 20.60 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 28.07 | 28.07 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 52.17 | 52.17 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.44 | 50.44 |
| mathqa | 15.35 | 15.35 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.20 | 24.60 |
| arc_easy | 27.72 | 27.72 |
| arc_challenge | 21.40 | 21.40 |
| winogrande | 50.20 | 49.88 |
| hellaswag | 25.51 | 25.96 |
| piqa | 52.07 | 50.60 |
| mathqa | 15.24 | 20.36 |

## Pad variant: right_fix
Padding side: right
fix_pad_query_mask: True

## Summary
| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |
|---|---:|---:|---:|---:|
| plain | 29.22 | 14.27 | 30.88 | 12.94 |
| qa | 29.54 | 14.19 | 32.43 | 12.11 |
| mc_letters | 31.41 | 13.15 | 31.41 | 13.15 |
| instruction | 29.76 | 14.15 | 31.50 | 12.08 |

## Template: plain
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.40 | 26.20 |
| arc_easy | 27.89 | 24.39 |
| arc_challenge | 17.73 | 17.73 |
| winogrande | 49.49 | 50.04 |
| hellaswag | 25.70 | 25.59 |
| piqa | 51.90 | 51.69 |
| mathqa | 15.44 | 20.56 |

## Template: qa
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.60 | 26.40 |
| arc_easy | 27.54 | 25.61 |
| arc_challenge | 19.40 | 25.75 |
| winogrande | 49.64 | 50.36 |
| hellaswag | 25.81 | 25.96 |
| piqa | 52.34 | 52.34 |
| mathqa | 15.46 | 20.60 |

## Template: mc_letters
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 22.00 | 22.00 |
| arc_easy | 28.07 | 28.07 |
| arc_challenge | 26.09 | 26.09 |
| winogrande | 52.17 | 52.17 |
| hellaswag | 25.73 | 25.73 |
| piqa | 50.44 | 50.44 |
| mathqa | 15.35 | 15.35 |

## Template: instruction
| Task | Acc | Acc Norm |
|---|---:|---:|
| openbookqa | 16.20 | 24.60 |
| arc_easy | 27.72 | 27.72 |
| arc_challenge | 21.40 | 21.40 |
| winogrande | 50.20 | 49.88 |
| hellaswag | 25.51 | 25.96 |
| piqa | 52.07 | 50.60 |
| mathqa | 15.24 | 20.36 |
