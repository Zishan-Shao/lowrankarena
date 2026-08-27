from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

# Required to allow lm-eval to load datasets that need trusted remote code
from datasets import config as ds_config
ds_config.HF_DATASETS_TRUST_REMOTE_CODE = True


def evaluate_with_harness(model, tokenizer, config):
    wrapped_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        max_length=2048,
        batch_size=8,
    )
    results = simple_evaluate(
        model=wrapped_model,
        tasks=list(config.tasks),
        num_fewshot=getattr(config, "num_fewshot", None),
        log_samples=False,
    )
    return results
