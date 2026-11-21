"""Train and evaluate the shipped PPO example through the common Python API."""

import yaml

from sequifier.config.train_config import SequifierConfig
from sequifier.config.training_resolution import resolve_training_run
from sequifier.rl_api import evaluate_policy, load_policy
from sequifier.scaffolds import template_text
from sequifier.train import run_training


def main():
    authored = SequifierConfig.model_validate(
        yaml.safe_load(template_text("train-ppo"))
    )
    config = resolve_training_run(authored)
    result = run_training(config)
    actor = load_policy(result.model_path)
    print("Selected return:", result.phases[-1].selected_metric)
    print(evaluate_policy(actor, config.training_plan[0].evaluation))


if __name__ == "__main__":
    main()
