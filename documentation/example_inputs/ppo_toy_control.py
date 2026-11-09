"""Train the categorical control example with the canonical Python policy API."""

from importlib.resources import files

from sequifier.config.train_config import load_train_config
from sequifier.rl_api import (
    PolicyRunConfig,
    evaluate_policy,
    load_policy,
    run_policy_training,
)


def main():
    config = load_train_config(
        str(files("sequifier").joinpath("templates", "train-ppo.yaml")), {}, False
    )
    if not isinstance(config, PolicyRunConfig):
        raise TypeError("the example requires a policy run config")
    result = run_policy_training(config)
    actor = load_policy(result.model_path)
    print("Selected return:", result.phases[-1].selected_metric)
    print(evaluate_policy(actor, config.phases[0].evaluation))


if __name__ == "__main__":
    main()
