# Learning architecture

All learning uses `SequifierConfig` and `training_plan.phases`; there is no second
policy architecture or run discriminator. See the [training reference](configs/train.md),
[policy guide](policy-learning.md), [artifact transfer guide](supervised-policy-learning.md),
and [inference reference](configs/infer.md) for the supported configuration surface.

| Owner | Contract |
| --- | --- |
| Model execution | Named interfaces, forward attention, context geometry, codecs, normalization, input schemas, decoder order and prediction scales. These survive changes of learning objective. |
| Authored run | Model or initialization artifact, datasets, policy bindings, ordered phases, output paths and global execution settings. |
| Phase | Algorithm budget, batching, trainable roles, optimizer/scheduler settings, evaluation, freezing and handoff selection. |
| Sources | Prepared histories, raw sequences and preference manifests with explicit timing, episode boundaries and optional transition capabilities. |
| Native runner | Algorithm-specific sampling, objectives, critics, environment state and in-phase recovery. |
| Coordinator | Dependency validation, incoming/selected artifacts, phase boundaries, receipts and reports. |

`ModelExecution` exposes execution facts without assigning them dataset ownership.
Compatibility adapters retain existing resolved training objects and artifact formats.
Policy bindings select existing targets; lagged actions and return conditioning are
explicit inputs. Categorical support, real scales/bounds/atoms and conditional activity
have the same meaning during training, collection, replay and inference.

An internal preparation owner retains resolved source identities, episode catalogs
and partitions from metadata fitting through preflight and execution. Feature codecs
and execution schemas refresh dataset views without selecting the partitions again;
changes to episode interpretation or split settings create new views. Dependencies
are still revalidated before execution and at phase boundaries. This preparation
state is private and does not enter configuration serialization or checkpoints.

Historical decisions are reconstructed at their actual observation boundary.
Termination disables bootstrap; truncation preserves the true final successor;
a collection cutoff does not invent either boundary. Training/validation splits
keep connected episode, group and duplicate-content components together.

Phase handoffs publish selected weights before committing progress. Checkpoints
retain optimizer/sampler/RNG state and validate representation, source, adapter and
reference identities. Mixed plans use native checkpoints inside an atomic run
envelope. Existing checkpoint formats, public configuration aliases, standalone
scaffold templates and portable model formats remain compatibility boundaries.
