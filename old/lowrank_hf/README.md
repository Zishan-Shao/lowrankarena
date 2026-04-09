# Low-Rank HF Export Contract

This directory defines a minimal contract for serving low-rank LLM checkpoints
through `transformers` and `vLLM --model-impl transformers`.

## Scope

This contract is designed for methods that:

- keep the base architecture unchanged
- only replace selected `nn.Linear` modules with low-rank factors
- represent each factorized linear as `A @ B`

Supported base architectures in this repo:

- `Llama`
- `Qwen2`
- `Qwen3`

If a method changes attention semantics, KV-cache layout, or introduces custom
non-linear blocks, it should use a dedicated model wrapper instead of this
contract.

## Required Config Fields

Custom model packages should save these fields in `config.json`:

- `auto_map`
- `architectures`
- `low_rank_method`
- `low_rank_schema`
- `low_rank_format_version`
- `low_rank_modules`

Example:

```json
{
  "architectures": ["LowRankLlamaForCausalLM"],
  "auto_map": {
    "AutoConfig": "configuration_lowrank_llama.LowRankLlamaConfig",
    "AutoModel": "modeling_lowrank_llama.LowRankLlamaModel",
    "AutoModelForCausalLM": "modeling_lowrank_llama.LowRankLlamaForCausalLM"
  },
  "low_rank_method": "svdllm_v2_whitening_only",
  "low_rank_schema": "ABLinear",
  "low_rank_format_version": 1,
  "low_rank_modules": {
    "model.layers.0.self_attn.q_proj": {"rank": 921}
  }
}
```

## Required Weight Layout

Every replaced linear should be stored as:

- `<module>.ALinear.weight`: `[out_features, rank]`
- `<module>.ALinear.bias`: `[out_features]` if needed
- `<module>.BLinear.weight`: `[rank, in_features]`

Example:

- `model.layers.0.self_attn.q_proj.ALinear.weight`
- `model.layers.0.self_attn.q_proj.BLinear.weight`

## Integration Checklist

For a new low-rank method:

1. Start from a base HF architecture such as `LlamaForCausalLM`.
2. Keep module names aligned with the original dense model paths.
3. Export per-module rank metadata into `config.low_rank_modules`.
4. Save factor weights using the `ALinear` / `BLinear` layout above.
5. Copy `configuration_lowrank_llama.py` and `modeling_lowrank_llama.py` into the model directory.
6. Load with:

```bash
vllm serve /path/to/model \
  --model-impl transformers \
  --trust-remote-code
```

## Notes

- `SVDLLM` old `.pt` checkpoints are pickle artifacts and should be converted
  with `baselines/SVD-LLM/huggingface_repos/export_svdllm_lowrank.py`.
- This format is architecture-preserving. It measures low-rank serving without
  densifying back into standard dense weights.
