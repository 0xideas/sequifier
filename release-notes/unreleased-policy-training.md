# Integrated policy training and inference (unreleased)

Offline IQL now accepts one fixed-dataset `iql` phase for an unconditioned,
single-categorical actor. It adds strict transition/successor validation,
independent value and twin-Q networks, target-Q updates, weighted actor
extraction, a complete learner checkpoint, and actor-only model export.
Continuous/joint actions, distributed IQL, and mixed plans remain future
work. No tests were written or run for this increment, as requested.

For imitation, `sequifier train` accepts one trajectory-backed `policy_imitation` phase;
`return_conditioned: false` selects behavior cloning. Policies now use the
standard configurable ingestion, backbone, decoder, and network implementation.
Encoded categorical observations and matching forecast-backbone transfer are
supported. PyTorch execution supports CPU, CUDA, and MPS.

The selected best model is an ordinary `sequifier_model` artifact, usable with
`sequifier infer`, Python policy APIs, and generic model-loading tools. Inference
supports raw CSV/Parquet decisions and environment execution through the shared
rollout coordinator. Each training epoch writes a standard resumable run
checkpoint, including optimizer/scheduler/scaler, random/data-order state,
training progress, and best weights. The older standalone actor format requires
retraining. The default model path is now `models/<name>/<name>.pt`.

File-backed training uses a bounded episode cache, incremental normalization,
and context-sized window construction. Behavior cloning retains observed
state/action pairs from truncated and cutoff episodes without return labels.
Grouped episode splitting and training-only normalization remain in place.

`sequifier make` includes policy train, IQL train, PPO train, and infer templates. Mixed algorithm
phases, distributed policy learning, multimodal trajectories,
and CLI ONNX policy execution remain future work.

Policy sequence representation follow-up:

- Train BC directly from CSV/Parquet observation/action tables. Rewards,
  successors, bindings and seed histories are not required for BC.
- JSON recordings and existing scalar prepared histories adapt into the same
  columnar episode representation, with optional transition capabilities.
- Policy windows use explicit same-position labels and causal attention, named
  tensors, standard sequence batches and the supervised loss service.
- Categorical maps and sample statistics use shared preprocessing functions and
  fit only training episodes. New action vocabularies use global/decoder mappings;
  existing policy artifacts retain their legacy layout.
- Policy configuration uses shared model/dataset/phase authoring. Redundant
  compact registries and dense input APIs have been removed. Preprocess can
  optionally commit raw Parquet sequence shards
  with a checksummed manifest.
- Repository tests were moved to `../sequifier-test/tests/sequifier-refactor-unit`;
  no tests were added. Verification uses lint, compilation and workflow smoke runs.

Architecture consolidation:

- Policy runs use the ordinary TrainingRun, TrainingEngine, evaluation, run state,
  checkpoint restore and loader-state services. The private policy loop is gone.
- Model components and global optimization settings use their shared schemas;
  policy records retain observation/action alignment rather than copying models.
- CSV inference and training share table-to-episode validation and causal windows.
  Tabular prepared windows share the stored-tensor reader with supervised data.
- Sequence import writes columnar episodes directly, retaining optional transitions.
- Earlier private policy run checkpoints require a new run initialized from their
  model weights. Portable models and execution weights remain readable.


Scoped online PPO:

- A single `ppo` phase trains an unconditioned categorical actor against one
  persistent local version-2 adapter, without datasets or supervised budgets.
- Shared bounded causal histories, compact sealed buffers, true-successor
  bootstrap, FP32 GAE and ratios, clipped policy/value objectives, entropy,
  sample-weighted gradient accumulation and independent actor/V optimizers.
- Frozen codecs and explicit normalization; strict BC/IQL actor warm starts.
- Separate seeded, bounded rollout evaluation selects actors by completed return
  or another declared environment metric, with censored metrics kept explicit.
- Atomic version-1 PPO learner checkpoints support declared exact simulator/RNG
  snapshots or non-trajectory-equivalent environment resets. Actor export stays
  an ordinary portable `sequifier_model` and includes no learner networks.
- Structured reports, optional independent collection archives, a deterministic
  categorical control adapter, CLI template and Python workflow are included.
- No tests were written or run. No training/benchmark runs or learning claims
  accompany this implementation; only static validation was performed.
- Vector/remote/Gym adapters, continuous/joint actions, multimodal/distributed
  PPO, mixed phases, reward providers, generalized monitors/search and ONNX
  policy execution remain deferred. The parent RL plan remains incomplete.
