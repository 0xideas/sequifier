# Phase-owned training and explicit inference routes

- Move batching and role-keyed optimization into each training phase. Supervised
  uses `optimization.model`; policy roles remain algorithm-specific. Reject
  legacy global optimization fields and remove the global reset switch.
- Add phase `optimizer_state: reset | continue`, with reset as the default and
  validated compatible native continuation. Reject mixed-plan continuation and
  policy OneCycle continuation. Preserve supervised epoch, update and metric
  scheduling; resolve epoch OneCycle chain budgets and validate batch budgets, including interleaving and dataset-boundary flushes.
- Store resolved phase settings in checkpoint/config fingerprints and native
  artifacts needed for recovery.
- Validate policy inference directly against its own schema and reject fields
  belonging to another route. CLI overrides use the same validation.
- Rename inference execution determinism to `enforce_determinism` and policy
  inference/evaluation selection to `action_selection`. Seeded sampling and
  deterministic execution remain separate controls.
- Use `rollout.environment` and select rollout schemas from the artifact interface and `task`.
  Forecast version-1 bindings and policy version-3 environments remain distinct.
  Document forecast `seed_window_index` versus policy `seed_row_index` units.
- Update shipped examples, scaffolding, Python examples and consolidated docs.
  CPU smoke tests cover native supervised and all policy algorithms, mixed
  handoff/recovery, inference, rollout and configuration rejection.

All algorithms own evaluation on their phase. Offline policy evaluation uses
`evaluation.sources` and `evaluation.monitor: {metric, mode}` without aliases.
Generative inference dispatches from the selected interface, with optional `task`;
`model_type` is `generative` or `embedding`. Rollout indices are `seed_window_index`
(forecast) and `seed_row_index` (policy), including trajectory metadata.
Scalar action mappings use `actions`/`previous_actions`; structured records use
`action_record`/`previous_action_record`.
