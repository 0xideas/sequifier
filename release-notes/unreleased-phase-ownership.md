# Phase-owned training and explicit inference routes

- Move batching and role-keyed optimization into each training phase. Supervised
  uses `optimization.model`; policy roles remain algorithm-specific. Remove global
  optimization fields and the global reset switch.
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
- Use `rollout.environment` and select rollout schemas from `model_type`.
  Forecast version-1 bindings and policy version-3 environments remain distinct.
  Document prepared-window versus raw-row `seed_index` units.
- Update shipped examples, scaffolding, Python examples and consolidated docs.
  CPU smoke tests cover native supervised and all policy algorithms, mixed
  handoff/recovery, inference, rollout and configuration rejection.

This is a breaking authoring change; obsolete placements and names are rejected.
