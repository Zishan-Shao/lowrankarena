import torch
import numpy as np
from tqdm import tqdm

from utils.data_utils import get_test_data


@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    ppls = {}

    print("=" * 80)
    print("Starting perplexity evaluation on {} dataset(s)".format(len(datasets)))
    print("=" * 80)

    for dataset in datasets:
        print("\n" + "-" * 80)
        print("Evaluating on dataset: {}".format(dataset))
        print("-" * 80)
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            # Handle both tensor and dict batches
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch, use_cache=False)
                input_ids = batch['input_ids']
            else:
                batch = batch.to(device)
                output = model(input_ids=batch, use_cache=False)
                input_ids = batch
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl

        # Print result immediately after computing for this dataset
        print("\n>>> RESULT for {}: PPL = {:.4f}".format(dataset, ppl))
        print("-" * 80)

    # Print summary of all results at the end
    print("\n" + "=" * 80)
    print("SUMMARY - PPL for all datasets:")
    print("=" * 80)
    for dataset, ppl in ppls.items():
        print("  {}: {:.4f}".format(dataset, ppl))
    print("=" * 80)
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    return ppls
