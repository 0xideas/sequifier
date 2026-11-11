# Configuration ownership

- Give every algorithm phase its own evaluation declaration, monitor and cadence.
  Supervised phases also own epoch checkpoint cadence through `checkpoint_interval`.
- Separate model-owned Gaussian scales (`initial`, `trainable`, `min`, `max`) from
  policy action behavior. Real actions select `scale: model` or `scale: {fixed: ...}`;
  action bounds, point masses and conditional activity remain task declarations.
- Preserve model scale parameters when a policy uses fixed exploration. Explicit
  model declarations may add new scale parameters to an initialized artifact.
- Keep frozen normalization global or inherited from an artifact; remove phase copies.
- Derive OneCycleLR budgets for supervised and policy training. Explicit `total_steps`
  values are assertions checked against the derived budget.
- Remove global optimization aliases, policy inference/evaluation aliases and rollout
  seed aliases. CLI overrides use canonical `--global-training`/`--training-plan`
  mappings. Scalar action mappings use `actions`/`previous_actions`; structured
  records use `action_record`/`previous_action_record`.

No configuration migration or compatibility layer is provided. Existing configuration
files and tests are intentionally left unchanged by this refactor. No tests were
written or run for this change.
