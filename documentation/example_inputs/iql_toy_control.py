"""Small offline control dataset and separate BC/IQL rollout comparison.

Generate data with ``python iql_toy_control.py generate data/offline-transitions.csv``.
After training both actors, run
``python iql_toy_control.py evaluate path/to/bc.pt path/to/iql.pt``.
This is an example benchmark, not part of the test suite.
"""

import csv
import random
import statistics
import sys
from pathlib import Path

from sequifier.config.policy_infer_config import PolicyRolloutConfig
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.environment import (
    EnvironmentCapabilities,
    OnlineResetResult,
    OnlineStepResult,
)
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.trajectory.schema import SeedHistory

HORIZON = 12


class ToyControlAdapter:
    protocol_version = 3
    implementation_id = "sequifier.toy-control.v1"
    capabilities = EnvironmentCapabilities(seeded_reset=True)

    def reset(self, seed):
        initial = OnlineResetResult(
            {"position": float(random.Random(seed).choice((-4, -3, 3, 4)))}
        )
        return self.reset_from_observation(initial, seed)

    def reset_from_observation(self, initial, seed):
        self.position = int(initial.observation["position"])
        self.steps = 0
        return initial

    def step(self, action):
        self.position += -1 if action["action"] == "left" else 1
        self.steps += 1
        done = self.position == 0
        return OnlineStepResult(
            observation={"position": float(self.position)},
            reward=1.0 if done else -0.05,
            terminated=done,
            truncated=self.steps == HORIZON and not done,
        )

    def close(self):
        pass

    def snapshot_state(self):
        raise NotImplementedError("snapshot capability is not supported")

    def restore_state(self, state):
        raise NotImplementedError("snapshot capability is not supported")

    def create_group_start(self, seed):
        raise NotImplementedError("group-start capability is not supported")

    def reset_group(self, token, rollout_seed):
        raise NotImplementedError("group-start capability is not supported")


def generate(path):
    rng = random.Random(1010)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sequenceId",
                "itemPosition",
                "position",
                "action",
                "reward",
                "terminated",
                "truncated",
                "bootstrap_allowed",
                "next_position",
            ],
        )
        writer.writeheader()
        for episode in range(500):
            position = rng.choice((-4, -3, 3, 4))
            quality = 0.85 if episode % 2 else 0.25
            for step in range(HORIZON):
                good = "left" if position > 0 else "right"
                action = (
                    good
                    if rng.random() < quality
                    else ("right" if good == "left" else "left")
                )
                successor = position + (-1 if action == "left" else 1)
                terminated = successor == 0
                truncated = step + 1 == HORIZON and not terminated
                writer.writerow(
                    {
                        "sequenceId": episode,
                        "itemPosition": step,
                        "position": float(position),
                        "action": action,
                        "reward": 1.0 if terminated else -0.05,
                        "terminated": terminated,
                        "truncated": truncated,
                        "bootstrap_allowed": not terminated,
                        "next_position": float(successor),
                    }
                )
                position = successor
                if terminated or truncated:
                    break


def evaluate(paths):
    config = PolicyRolloutConfig.model_validate(
        {
            "environment": {
                "factory": "iql_toy_control:ToyControlAdapter",
                "implementation_id": "sequifier.toy-control.v1",
                "units": {},
            },
            "max_steps": HORIZON,
            "trajectory_path": "unused.json",
            "seed_row_index": 0,
            "max_episode_steps": HORIZON,
        }
    )
    for path in paths:
        policy = load_policy(path)
        returns = []
        for episode, position in enumerate((-4, -3, 3, 4) * 25):
            seed = SeedHistory(
                observations={
                    "position": [0.0] * (policy.config.context_length - 1)
                    + [float(position)]
                },
                attention_valid_mask=[False] * (policy.config.context_length - 1)
                + [True],
                static_inputs={},
            )
            trajectory = collect_policy_rollout(
                policy,
                ToyControlAdapter(),
                seed,
                config,
                HORIZON,
                random_seed=episode,
            )
            returns.append(trajectory.total_return)
        print(path, "mean", statistics.mean(returns), "std", statistics.stdev(returns))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "generate":
        generate(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2:])
    else:
        raise SystemExit(
            "usage: iql_toy_control.py generate FILE | evaluate BC_MODEL IQL_MODEL"
        )
