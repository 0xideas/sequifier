# Multi-phase categorical policy training (unreleased)

- One `PolicyRunConfig`, CLI dispatch and Python `run_policy_training` entry point
  support ordered imitation, IQL, PPO, DPO and GRPO phases with natural budgets.
- A frozen architecture/interface/codec/normalization/context/units contract spans
  phases. Selected actors are copied at handoff; optimizer/critic state is local.
  Best/last selection is explicit, ties retain the earliest best, and final export
  uses the final phase's selection.
- DPO adds versioned reward-free preference spans, connected-component-safe
  partitions, branch-history validation, summed causal action likelihoods and
  canonical sigmoid DPO. Chunked backward scoring bounds activation memory.
- GRPO adds explicit reproducible group-start capabilities, independent RNG
  streams, bounded complete groups, population-normalized advantages, clipped
  ratios and a separately named sampled frozen-reference penalty, without a critic.
- Version-3 local online adapters serve PPO and GRPO. Exact continuation requires
  validated snapshots or documented reproducible group starts; non-exact restarts
  report a discontinuity. Failed/partial groups are never reused.
- One atomic `sequifier_policy_run` checkpoint embeds phase references, selected
  actors, source/representation hashes, sampler/RNG state and committed progress.
  The former unreleased single-algorithm configuration/checkpoint paths are removed.
- Actor-only portable artifacts remain usable for inference, rollout and cross-run
  initialization. Default output is `models/<name>/actor.pt`; phase reports and
  selected actors use indexed directories.
- Templates, examples, public exports and guides use the canonical schema. The
  pre-existing supervised training runtime remains independent.
- No tests or training jobs were written/run for this change. Static compilation,
  lint and type analysis are the verification scope; numerical learning and exact
  continuation acceptance evidence remains to be established.

Scope: one categorical action, scalar/categorical inputs, causal fixed histories,
local single-process FP32. Return conditioning supports imitation only. Text/LLM,
continuous/joint actions, images/depth, vector/distributed execution, reward models,
optimizer/critic inheritance and mid-group checkpoints remain deferred.
