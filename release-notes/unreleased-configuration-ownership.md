# Configuration ownership

- Replace authored `global_training` and root resources with explicit `run`, durable `model.execution` and named interface input contracts.
- Keep dataset declarations focused on source identity and split policy. Move losses, metrics, curriculum and trainability to phases.
- Share typed catalog parameter selectors between optimizer routing and every algorithm's role trainability; supervised phases retain source-dependent masks.
- Put evaluation cadence, initial/snapshot triggers and batching on each phase. Keep retained snapshots phase-owned and rolling recovery run-owned.
- Add observation-based stopping with explicit `end_phase` or `end_plan` scope across native and mixed runners, with checkpointed patience and stop state.
- Update CLI mappings, search paths, preprocessing/inference resource ownership, scaffold templates, examples, notebooks and documentation.
- Bump native recovery formats because phase ownership and stopping state changed; portable execution artifacts remain usable for initialization/inference.

No tests were written or run for this implementation, at the user's request. Validation is limited to static source inspection, syntax, lint and type analysis.
