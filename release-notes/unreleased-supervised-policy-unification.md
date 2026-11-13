# Supervised and policy model unification

- Attach policy tasks directly to ordinary supervised model artifacts and
  supported supervised run checkpoints, preserving prediction heads and inputs.
- Train named combinations of categorical and real action heads with imitation,
  IQL, PPO, DPO, and GRPO. Retain autoregressive/composite decoders and hashed
  categorical candidate semantics.
- Add bounded real distributions, optional point masses, conditional activity,
  bounded global exploration scales, and transform-latent replay.
- Preserve full execution-schema payloads, including static, depth, image, and
  mask inputs, throughout policy history and storage.
- Add mixed supervised/policy run plans with native phase recovery, atomic
  handoffs, model identity checks, and source/environment preflight.
- Separate persisted forward attention from optimization objectives and allow
  explicit prediction or policy inference views.
- Add shared real prediction distributions and supervised Gaussian NLL support.
- Write model artifact version 3 and typed policy trajectory/collection version 2;
  retain readers for earlier model artifacts and single-categorical interfaces.

Focused CPU tests now cover phase-owned optimization, all policy algorithms,
checkpoint recovery, mixed handoffs and forecast/policy inference and rollout.
See [the usage guide](../documentation/supervised-policy-learning.md) and
[phase ownership notes](unreleased-phase-ownership.md) for the implemented contracts.
