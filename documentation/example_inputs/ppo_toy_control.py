"""Train and evaluate the shipped PPO example through the common Python API."""

from importlib.resources import files

from sequifier.config.train_config import load_train_config
from sequifier.rl_api import evaluate_policy, load_policy
from sequifier.train import run_training


def main():
    config = load_train_config(
        str(files("sequifier").joinpath("templates", "train-ppo.yaml")), {}, False
    )
    result = run_training(config)
    actor = load_policy(result.model_path)
    print("Selected return:", result.phases[-1].selected_metric)
    print(evaluate_policy(actor, config.training_plan[0].evaluation))


if __name__ == "__main__":
    main()
