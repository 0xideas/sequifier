# Multi-phase categorical policy training (unreleased)

- One `SequifierConfig` authoring schema with `training_plan.phases`, CLI dispatch and Python `run_training` entry point
  support ordered imitation, IQL, PPO, DPO and GRPO phases with natural budgets.
- A frozen architecture/interface/codec/normalization/context/units contract spans
  phases. Selected actors are copied at handoff; compatible native phases may explicitly
  continue optimizer/critic state. Mixed plans reset optimization at handoffs.
  Best/last selection is explicit, ties retain the earliest best, and final export
  uses the final phase's selection.
- DPO adds versioned reward-free preference spans, connected-component-safe
  partitions, branch-history validation, summed causal action likelihoods and
  canonical sigmoid DPO. Chunked backward scoring bounds activation memory.
- GRPO adds explicit reproducible group-start capabilities, independent RNG
  streams, bounded complete groups, population-normalized advantages, clipped
  ratios and a separately named sampled frozen-reference penalty, without a critic.
- Version-3 local online adapters serve PPO, GRPO and policy rollout. Explicit
  seeded reset preserves supplied initial observations during inference. Exact continuation requires
  validated snapshots or documented reproducible group starts; non-exact restarts
  report a discontinuity. Failed/partial groups are never reused.
- One atomic version-6 (with version-5 resume support) `sequifier_policy_run` checkpoint owns phase progress,
  role-keyed optimizer/network state, references, source identities and RNG.
  Phase actors are durable exports referenced by checksum; reports are derived
  and regenerated on resume. No phase manifests are needed.
- Shared phase sessions handle selection, checkpoints and optimization. Each
  phase configures optimizer roles under `optimization`; gradient validation
  and clipping precede all optimizer steps in a transaction.
- Actor-only portable artifacts remain usable for inference, rollout and cross-run
  initialization. Default output is `models/<name>/model.pt`; phase reports and
  selected actors use indexed directories.
- Templates, examples, public exports and guides use the canonical schema. The
  pre-existing supervised training runtime remains independent.

Policy bindings support categorical and real action heads, including joint actions,
through the shared model execution schema. Policy algorithms currently execute
locally; distributed supervised execution remains available. See the current
[phase ownership notes](unreleased-phase-ownership.md) and
[policy guide](../documentation/policy-learning.md) for the supported surface.
