# Canonical public configuration names

- Use named training interfaces, datasets, parts, bindings, and phases for every
  cardinality. Remove singleton aliases and boolean evaluation shorthand.
- Standardize dimensions, counts, batch/optimizer/iteration clocks, selectors,
  checkpoint intervals, paths, monitoring direction, and probability names.
- Require explicit inference `task` and list-valued `model_paths`. Named media
  paths are mappings; preprocessing widths are always `window_lengths`.
- Nest registered component constructor settings under `arguments` and declare
  initialization methods as mappings. Remove scalar, method, and dtype aliases.
- Give hyperparameter search explicit `$choices`, `$fixed`, `$range`, and
  `$variants` operators with canonical training paths and independent sampler seed.
- Resolve run `seed: random` once. CLI flags mirror the canonical keys, with no
  short aliases or abbreviations and explicit boolean override values.
- Update runtime consumers, fingerprints, scaffolding, templates, and guides.

This is a breaking authoring change without an alias compatibility layer. See
[the complete naming reference](../documentation/configs/naming.md). Configurations
embedded in artifacts/checkpoints must also use the canonical reconstruction
schema; old embedded declarations are not silently translated.

No tests were written or run for this change, as requested. Verification is limited
to static source syntax, reference searches, and diff inspection.
