
import math
import torch
from tqdm import tqdm

@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=("wikitext2",), model_seq_len=2048, batch_size=4, device="cuda", label="PPL", max_batches=None):
    # Lightweight token-level PPL evaluator (enough to satisfy imports)
    from utils.data_utils import get_test_data

    model.eval().to(device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    out = {}

    for ds in datasets:
        loader = get_test_data(ds, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        total_nll = 0.0
        total_tokens = 0

        for i, batch in enumerate(tqdm(loader, desc=f"{label}[{ds}]")):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)

            outputs = model(input_ids=batch, use_cache=False, return_dict=True)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_nll += loss.sum().item()
            total_tokens += loss.numel()

        out[ds] = math.exp(total_nll / total_tokens) if total_tokens else float("nan")

    print(f"{label}: {out}")
    return out

@torch.no_grad()
def eff_eval(*args, **kwargs):
    # Optional; not needed for lm-eval. Kept only because some scripts import it.
    print("[eff_eval] Not implemented in this minimal evaluater.py (not required for lm-eval).")

