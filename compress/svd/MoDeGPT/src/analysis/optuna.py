import optuna
from src.adapters.CompressionConfig import CompressionConfig

import os

from src.run_modegpt import main


def objective(trial: optuna.Trial):
    """
    Used for llama3-8B to ensure replicability of original paper.

    Minimize for ppl
    """

    config = CompressionConfig(
        model="Qwen/Qwen3-32B",
        device=0,
        calib_size=128,
        calibs_batch_size=16,
        output_dir=f"{os.environ.get('TMPDIR', '/tmp')}/compressed_output/Qwen3-32B/",
        note=f"optuna sparsity Qwen3-32B trial {trial.number}",
        compression_ratio=0.5,
        order=f"mlp,qk,vo",
        max_sparsity=0.95,
        nystrom_ridge=trial.suggest_categorical("nystrom_ridge", [1, 1e-1, 1e-2, 1e-3, 1e-4]),
        sparsity_smoothing=trial.suggest_float("sparsity_smoothing", 0.0225, 0.15),
        ridge_vo=trial.suggest_categorical("ridge_vo_cat", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        ridge_qk=trial.suggest_categorical("ridge_qk_cat", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        dataset="alpaca",
    )

    score = main(trial=trial, config=config)

    return score


study = optuna.create_study(
    study_name="modegpt-qwen3-32b-better",
    storage="sqlite:///sweep-modegpt-qwen3-32b.db",
    load_if_exists=True,
    direction="minimize",
)

study.optimize(objective, n_trials=20)

print(study.best_params)
print(study.best_value)
