# Trajectory-backed policy training (unreleased)

`sequifier train` accepts a single `return_conditioned` phase with named trajectory
sources and categorical actors. Set `return_conditioned: false` on the actor to
train behavior cloning through the same workflow. Sources use explicit JSON file
lists or version-1 manifests, with automatic grouped episode splitting or authored
training/validation sources. Normalization fits the training partition only.

Runs save a best-action-NLL actor and a training report with source paths,
partition IDs, exclusions, metrics and optimizer-step counts. These CPU actor
artifacts support `load_policy` and `collect_policy_rollout`; exact training resume
and policy exports remain planned.

The experimental `sequifier policy` command and synthetic control/dependent-feature
demos have been removed, including their configuration types and public exports.
Applications supply their own trajectories and environment adapters.
`sequifier make` now includes `configs/train-policy.yaml`.

Mixed algorithm phases, transfer, IQL/PPO, distributed RL, multimodal trajectories
and scalable sharded storage remain future increments. No tests or training runs
were executed for this increment, as requested.
