# Training artifacts

Supervised learning, policy imitation, IQL, PPO, DPO, GRPO and mixed plans use the
same public output, checkpoint, report and scalar metric contracts. Each reader requires the current version for its contract. See
[ContractVersions](../../src/sequifier/contract_versions.py) for the IDs.

## Configuration and layout

All paths are relative to `run.project_root` unless absolute.

| Setting | Default |
| --- | --- |
| `run.outputs.model_path` | `models/<model_name>.pt` |
| `run.outputs.report_path` | `outputs/training/<model_name>/report.json` |
| `run.outputs.phase_directory` | `outputs/training/<model_name>/phases` |
| `run.recovery.resume.checkpoint_path` | `checkpoints/runs/<model_name>/<model_name>-latest.pt` |

Every runner publishes a final portable PT model, independently of the optional
best/last, embedding and ONNX exports controlled by `run.export`. Supervised
training retains its existing `<model>-best-<epoch>` and `<model>-last-<epoch>`
exports. The canonical final model follows the final executed phase's
`model_selection.handoff`.

Every completed phase publishes `<phase_directory>/<index:03d>-<name>/model.pt`
and `report.json`. Mixed-plan native recovery is stored in that phase's `run.pt`;
native child artifacts are below `native/`. Policy snapshots use
`checkpoints/transaction-NNNNNNNN.pt` below the native phase directory. Supervised
snapshots retain their model/epoch filenames below the same phase `checkpoints/`
directory; the rolling checkpoint remains at its configured run-level path.

Phase trees, logging directories and collection archive directories are reserved
output namespaces. Model, report, checkpoint and manifest files must be distinct
and cannot overlap reserved namespaces or source data. Validation runs before
publication, including generated phase destinations and symlink escapes.

## Portable models and recovery

Portable models use `artifact_type: sequifier_model`, `format_version: 105`.
Their execution contract embeds interfaces, codecs, normalization, policy
bindings and units. Policy execution reads units from the selected interface;
provenance is not a second source of execution configuration.

All newly written run checkpoints have the same envelope:

```text
artifact_type: sequifier_run_checkpoint
format_version: 100
model: <portable model payload, or null before the first model exists>
recovery_type: supervised | policy | learning
recovery: <runner-owned recovery state>
```

The recovery payload retains optimizer, scheduler, sampler, RNG, selection and
algorithm-specific state. Exact recovery validates the runner type as well as
its own state version and contracts. A checkpoint cannot resume another runner.
Inference, initialization and analysis use the common embedded `model`, without
having to parse recovery state or supply training YAML. Loading a checkpoint
that precedes the first executable model raises an explicit error. A mixed-plan
envelope embeds the last committed incoming phase model; its native phase
checkpoint contains in-phase progress.

## Reports and discovery

All final reports use `artifact_type: sequifier_run_manifest`,
`format_version: 108`, at the configured report path and at `manifest.json` beside
the rolling checkpoint. Common fields identify the logical run, current session,
status, completion reason, final model, checkpoint, exports and completed phases.
Additional fields contain runner-specific provenance.

Each phase report uses `artifact_type: sequifier_phase_report`,
`format_version: 107`. Its common fields include phase index, name, algorithm,
model and checkpoint paths, monitor/direction, handoff, selected metric and
transaction, selected weight hash, clocks, metrics, provenance and stop scope.
Unavailable measurements are null. Algorithm-specific metrics keep their names
and units; a reward monitor is not represented as a loss.

## Logging and scalar metrics

Every native runner configures the model-scoped logger under `logs/<model_name>`.
Structured scalar observations use the supervised writer's `*-training.csv`,
`*-training-full.csv`, `*-validation.csv` and `*-validation-full.csv` tables.
All tables carry `schema_version: 109`. Full tables include phase, algorithm and
iteration identity, run/session IDs, UTC timestamps and progress. Named scalar
observations use `metric`, `target`, `value` and `is_primary`; policy observations
use `target: __total__`. Nested diagnostic records remain in phase reports.

`run.logging.log_interval_batches` controls supervised training observations;
`run.logging.log_interval_iterations` controls policy training observations and
defaults to 10. Mixed plans may configure both. Evaluations are written whenever
the phase evaluation schedule runs, and final policy training progress is always
written. Policy iteration clocks count committed updates, even when an iteration
contains several optimizer steps. Policy visualizations display the actual
monitor names on a linear scale and do not invent baseline measurements.

Hyperparameter search accepts supervised plans. Policy and mixed plans are not yet
supported. Its default objective is the final phase's evaluation monitor; the
search direction must match. Explicit output destinations are isolated per
sampled run.

## Inference and collection outputs

Forecast and policy predictions share `sequenceId`, `subsequenceId`,
`windowStartOffset`, `itemPosition` and named prediction columns. Independent
policy rows receive synthetic sequence/item identities. Categorical probabilities
are separate target-specific files under
`outputs/probabilities/<model>/<target>/part-NNN.<format>`, with decoded category
labels as column names. Policy predictions additionally include
`action_log_probability` when probabilities are requested. Conditional policy
probabilities remain conditional on the selected preceding actions.

All inference routes accept `output_path` for one model. It overrides the first
prediction/embedding file; later chunks append `-part-NNN` to its stem. Probability
files keep their canonical target-specific destinations.

Forecast and policy rollout use the shared validated trajectory record.
Training-time PPO and GRPO archives instead use the common
`sequifier.collection-archive` envelope (version 122) with `algorithm` and
`collection` fields. The collection payload contains algorithm-specific replay
and diagnostic information. `read_collection_archive` reads the envelope;
`load_collection_record` additionally validates PPO collections. These archives
are diagnostics, not interchangeable offline training trajectories.
