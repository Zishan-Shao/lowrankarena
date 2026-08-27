import logging
import os

import wandb
from hydra.core.hydra_config import HydraConfig

from aa_svd.evaluate.lm_eval import evaluate_with_harness
from aa_svd.evaluate.ppl import evaluate_ppl
from aa_svd.utils import save_json, get_dtype

logger = logging.getLogger(__name__)


def _results_path(subdir, dataset_name, step):
    """Build a result file path under the Hydra output directory."""
    name = (
        "result.json" if dataset_name is None
        else f"result_{dataset_name}.json"
    )
    if step is not None:
        name = f"{step}_{name}"
    hydra_dir = HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(hydra_dir, subdir)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, name)


def evaluate(cfg, model, tokenizer, dataset_name=None, step=None):
    eval_dtype = get_dtype(cfg.get("eval_dtype", None))
    if eval_dtype is not None:
        model = model.to(eval_dtype)

    use_lm_eval = (
        hasattr(cfg.evaluate, "lm_eval")
        and hasattr(cfg.evaluate.lm_eval, "tasks")
        and cfg.evaluate.lm_eval.tasks is not None
    )
    if use_lm_eval:
        logger.info("Using lm_evaluation_harness for language task evaluation")
        results = evaluate_with_harness(
            model=model,
            tokenizer=tokenizer,
            config=cfg.evaluate.lm_eval,
        )

        if wandb.run:
            to_log = {
                f"eval/{key}": results["results"][key]["acc,none"]
                for key in results["results"]
            }
            if dataset_name is not None:
                to_log = {
                    f"{k}_{dataset_name}": v for k, v in to_log.items()
                }
            if step is not None:
                to_log["step"] = step
            wandb.log(to_log)

        results_file = _results_path("lm_eval_results", dataset_name, step)
        save_json(results, results_file)
        logger.info(f"Evaluation results saved to {results_file}")

    compute_ppl = (
        hasattr(cfg.evaluate, "ppl")
        and hasattr(cfg.evaluate.ppl, "datasets")
        and cfg.evaluate.ppl.datasets is not None
    )
    if compute_ppl:
        logger.info("Using perplexity evaluation")
        results = evaluate_ppl(
            model=model,
            tokenizer=tokenizer,
            config=cfg.evaluate.ppl,
        )

        if wandb.run:
            suffix = f"_{dataset_name}" if dataset_name is not None else ""
            to_log = {f"eval/{k}{suffix}": v for k, v in results.items()}
            if step is not None:
                to_log["step"] = step
            wandb.log(to_log)

        results_file = _results_path("ppl_eval_results", dataset_name, step)
        save_json(results, results_file)
        logger.info(f"Perplexity results saved to {results_file}")
