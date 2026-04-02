# Eval Pipeline Issues & Fixes

## 1. `SiLUActivation` missing in newer transformers

**症状**
```
AttributeError: Can't get attribute 'SiLUActivation' on <module 'transformers.activations' ...>
```
加载 SVD-LLM / Basis Sharing checkpoint 时 pickle 无法找到 `SiLUActivation` 类。

**原因**
旧版 transformers 有 `transformers.activations.SiLUActivation`，新版删除了。老 checkpoint 的 pickle 里引用了这个类名，加载时崩溃。

**修复**
在 `torch.load` 之前注入 compat 类（`eval_decoder.py: _load_model_pt` / conversion scripts）：
```python
import transformers.activations as _act
import torch.nn as _nn, torch.nn.functional as _F
if not hasattr(_act, "SiLUActivation"):
    class _SiLUActivationCompat(_nn.Module):
        inplace = False          # class-level attr survives empty __dict__
        def forward(self, x):
            return _F.silu(x)
    _act.SiLUActivation = _SiLUActivationCompat
```
注意：不能直接 `_act.SiLUActivation = _nn.SiLU`，因为 unpickle 后 `nn.SiLU` 实例的 `__dict__` 为空，访问 `.inplace` 仍会崩溃。

---

## 2. `nn.SiLU` 缺少 `inplace` 属性（ASVD）

**症状**
```
AttributeError: 'SiLU' object has no attribute 'inplace'
```
ASVD 模型的 PPL 评测和 lm-eval 均失败。

**原因**
ASVD 的 `SVDLinear` 内部用 `nn.SiLU` 存储激活函数。老 checkpoint unpickle 后 `nn.Module.__setstate__` 没有恢复 `inplace` 实例属性，而 `nn.SiLU.forward` 直接访问 `self.inplace`。

**修复**
在 `_load_model_pt` 里 patch `nn.SiLU.forward`（对精度/速度无影响）：
```python
def _silu_forward_safe(self, x):
    return torch.nn.functional.silu(x, inplace=getattr(self, "inplace", False))
torch.nn.SiLU.forward = _silu_forward_safe
```

---

## 3. `torch.dtype` 不可 JSON 序列化

**症状**
```
TypeError: Object of type dtype is not JSON serializable
```
调用 `model.config.save_pretrained(output_dir)` 时崩溃（转换脚本）。

**原因**
部分老 checkpoint 的 `model.config` 里有 `torch.dtype` 类型的属性（如 `torch_dtype`），`json.dumps` 无法序列化。

**修复**
`convert_svdllm_to_hf_dir.py` 和 `convert_asvd_to_hf_dir.py` 保存前转成字符串：
```python
for attr in list(vars(cfg)):
    if isinstance(getattr(cfg, attr, None), torch.dtype):
        setattr(cfg, attr, str(getattr(cfg, attr)))
cfg.save_pretrained(args.output)
```

---

## 4. 转换脚本"已存在目录"误判为完成

**症状**
转换中途失败后再次运行，脚本跳过了未完成的 checkpoint（`[skip] already exists`）。

**原因**
脚本检查目录是否存在（`-d "$out"`），但目录在写入第一个文件（`model.pt`）后即被创建，后续失败不影响目录存在性。

**修复**
改为检查 `lowrank_config.json` 是否存在（该文件是转换流程的最后一步写入）：
```bash
if [[ -f "$out/lowrank_config.json" ]]; then
    echo "[skip] already complete: $out"
    continue
fi
```

---

## 5. lm-eval auto batch 探测失败（GQA 模型）

**症状**
```
[error] lm-eval failed: The size of tensor a (32) must match the size of tensor b (8) at non-singleton dimension 1
```
出现在 SVDLLMv2 等使用 GQA（32 Q heads / 8 KV heads）的模型上，PPL 正常但 lm-eval 失败。

**原因**
lm-eval 使用 `batch_size=auto` 时会递增探测最大 batch size，探测过程中触发 GQA 相关的形状冲突。

**修复**
eval 脚本固定 `--lmeval_batch_size 2`，跳过 auto 探测：
```bash
python eval_decoder.py ... --lmeval_batch_size 2
```

---

## 6. `mathqa` 数据集 script-based，新版 datasets 不支持

**症状**
```
[error] lm-eval failed: Dataset scripts are no longer supported, but found math_qa.py
```

**原因**
`datasets >= 3.0` 禁止执行自定义加载脚本，`trust_remote_code` 同时废弃。mathqa 的 HF 原始仓库是 script-based。

**修复**
1. 用 parquet mirror 下载到本地：
   ```bash
   python tools/download_mathqa.py  # 保存到 data/mathqa/test.jsonl
   ```
2. 自定义 lm-eval task YAML（`evaluate/tasks/mathqa_local.yaml`），使用本地 JSON 文件。
3. `run_lmeval` 自动检测本地文件，将 `mathqa` 替换为 `mathqa_local` 并传入 `TaskManager(include_path=...)` 。

---

## 7. `whitening_only` 和 `whitening_then_update` 互相覆盖

**症状**
CSV 里只有 `whitening_only` 的结果，`whitening_then_update` 被去重跳过。

**原因**
两者都被映射为 method=`SVDLLMv1`，去重逻辑按 `(model_tag, method, keep_ratio)` 三元组判断，相同 keep_ratio 下第二个被跳过。

**修复**
eval 脚本从 checkpoint 目录名提取 variant tag，拼接为完整 method 名：
```bash
tag=$(basename "$ckpt_dir" | sed 's/^hf_//' | sed 's/_[0-9]*\.[0-9]*$//')
method="${parent_method}_${tag}"
# e.g. SVDLLMv1_whitening_only, SVDLLMv1_whitening_then_update
```

---

## 8. PTB 数据集无法加载

**症状**
```
[warn] PTB not available on this machine — skipping
```
或 `trust_remote_code is not supported anymore`。

**原因**
HF 上的 PTB mirror 均为 script-based，`datasets >= 3.0` 不支持。

**修复**
预先下载到本地（`data/ptb/ptb_test.txt`），eval 时优先读本地文件：
```bash
python tools/download_ptb.py
```
`_iter_texts` 优先检查 `data/ptb/ptb_test.txt`，不存在才尝试 HF。
