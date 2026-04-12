# `compress/quant/`

This directory contains LowRankArena wrappers for quantization-oriented artifact generation.

Quantization is kept as a first-class local path because it is the most likely compression family to remain runnable in the shared benchmark environment.

## Wrappers

- [`awq.py`](./awq.py): adapter for AWQ-style exports.
- [`gptq.py`](./gptq.py): adapter for GPTQ-style exports.
- [`rtn.py`](./rtn.py): adapter for RTN-style quantization flows.

## Scope

- Plan or execute quantized artifact generation.
- Export metadata in the same shape used by the other compression families.
- Stay aligned with the main benchmark runtime whenever practical.
- Keep benchmark-time loading, eval, memory, and speed out of this directory.
