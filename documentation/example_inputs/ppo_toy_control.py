"""Python PPO workflow. Run from the repository root after installing Sequifier.

    python documentation/example_inputs/ppo_toy_control.py

The bundled deterministic adapter is implemented in sequifier.rl.control.
Its random-policy expected return is 0; optimal eight-action return is 8.
This example trains a policy; it is not a test or a validation benchmark.
"""

from typing import cast

from sequifier.config.train_config import PPOTrainConfig, load_train_config
from sequifier.rl_api import evaluate_policy, load_policy, run_ppo_training


def main():
    loaded_config = load_train_config(
        "documentation/example_inputs/configs/train-ppo-toy.yaml", {}, False
    )
    if not isinstance(loaded_config, PPOTrainConfig):
        raise TypeError("the example requires an online PPO config")
    config = cast(PPOTrainConfig, loaded_config)
    result = run_ppo_training(config)
    actor = load_policy(config.model_path)
    print("Best evaluated return:", result.best_metric)
    print(evaluate_policy(actor, config.training_plan.phases[0].evaluation))


if __name__ == "__main__":
    main()
