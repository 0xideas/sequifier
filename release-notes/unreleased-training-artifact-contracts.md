# Unified training artifact contracts

- Honour `run.outputs` for supervised-only, policy-only and mixed plans, including
  a canonical final portable model and per-phase model/report records.
- Reject overlapping output files, reserved phase/log/archive trees, protected
  inputs and generated destinations escaping through symlinks before publication.
- Store exact recovery behind a shared `sequifier_run_checkpoint`
  envelope with an embedded executable model and runner-owned recovery payload.
- Publish versioned run manifests and phase reports, retaining
  supervised manifest discovery beside the rolling checkpoint.
- Read policy units from the model interface across every export/loading route.
- Write policy observations through the supervised scalar metric tables with
  explicit version, phase, algorithm, iteration and primary-metric fields. Add
  `run.logging.log_interval_iterations` and support policy visualization.
- Align prediction identity columns, categorical probability files and
  `output_path` across inference tasks. Wrap algorithm-specific training collection
  archives in a shared, versioned envelope.

This is a new major-version contract. Previous native checkpoint formats remain
readable; earlier metric schemas are not migrated.
