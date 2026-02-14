import torch
import torch.nn as nn
from tqdm import tqdm
import os

from datautils import get_eval_loaders
#from lm_eval.base import BaseLM
from lm_eval import evaluator
from lm_eval.api.model import LM as BaseLM
from datasets import load_dataset
import time
import re

# change the one with newer version as HFLM already implemented methods
from lm_eval.models.huggingface import HFLM


TASKS = ["wikitext", "c4", "openbookqa", "arc_easy", "arc_challenge",
         "hellaswag", "winogrande", "piqa", "mathqa"]

class EvalLM(HFLM):
    def __init__(self, model, tokenizer, batch_size=1, seqlen=2048):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=str(model.device),
        )
        # compatibility with old ASVD eval code
        self.seqlen = seqlen



@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i : i + 1, :-1].to(model.device)
        labels = dataset[i : i + 1, 1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    use_bos=False,
):
    """
    model: huggingface model object (possibly decomposed)
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str, tasks split by ',' (e.g. "arc_easy,hellaswag")
    num_fewshot: number of examples in few-shot context
    eval_ppl: str datasets split by ',' (e.g. "wikitext2,ptb,c4")
    """

    results = {}

    # -----------------------
    # Perplexity evaluation
    # -----------------------
    if eval_ppl:
        base_seqlen = 2048
        device = model.device

        use_cache = getattr(model.config, "use_cache", False)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.eval()

        for dataset in [d.strip() for d in eval_ppl.split(",") if d.strip()]:
            seqlen = base_seqlen - (1 if use_bos else 0)

            # Cache only the input_ids tensor so torch.load works safely under PyTorch >=2.6
            cache_testenc = f"/tmp/{dataset}_input_ids_{model_name.replace('/', '_')}.pt"
            testenc = None
            if os.path.exists(cache_testenc):
                try:
                    testenc = torch.load(cache_testenc)
                    if not torch.is_tensor(testenc):
                        testenc = None
                except Exception:
                    testenc = None

            if testenc is None:
                testloader = get_eval_loaders(dataset, tokenizer)
                testenc = testloader.input_ids
                torch.save(testenc, cache_testenc)

            nsamples = testenc.numel() // seqlen
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)

                if use_bos:
                    bos_tokens_tensor = torch.full(
                        (batch.size(0), 1),
                        tokenizer.bos_token_id,
                        dtype=batch.dtype,
                        device=device,
                    )
                    batch_in = torch.cat([bos_tokens_tensor, batch], dim=1)
                    logits = model(input_ids=batch_in).logits
                    logits = logits[:, 1:, :]  # drop BOS position
                else:
                    logits = model(input_ids=batch).logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().to(device)

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

                if i == limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
            print(dataset, ppl.item())
            results[dataset] = ppl.item()

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache

    # -----------------------
    # Downstream task eval (lm-eval-harness)
    # -----------------------
    if tasks == "longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=full_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "small_longbench":
        from tools.eval_longbench import eval_longbench, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=small_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "mmlu":
        # keep original behavior if you rely on these exact task IDs
        tasks = "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
    elif tasks == "llmqat":
        tasks = "lambada_openai,openbookqa"

    tasks = tasks.strip() if isinstance(tasks, str) else tasks
    if tasks:
        # only build the lm-eval wrapper if we actually run tasks
        lm = EvalLM(model, tokenizer, batch_size=batch_size)

        task_list = [t.strip() for t in str(tasks).split(",") if t.strip()]
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=task_list,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            log_samples=False,
        )

        if t_results and "results" in t_results:
            t_results = t_results["results"]
            # mean over acc / acc_norm metrics when present
            acc_list = []
            for _task_name, metrics in t_results.items():
                if not isinstance(metrics, dict):
                    continue
                if "acc_norm" in metrics:
                    acc_list.append(metrics["acc_norm"])
                elif "acc" in metrics:
                    acc_list.append(metrics["acc"])

            if len(acc_list) > 0:
                t_results["mean"] = sum(acc_list) / len(acc_list)
                print(f"===== mean acc: {t_results['mean']} =====")

            results.update(t_results)
            print(results)

    return results
