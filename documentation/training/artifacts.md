# Training artifacts

Supervised learning, policy imitation, IQL, PPO, DPO, GRPO and mixed plans use the
same output, checkpoint, report and scalar metric formats. Sequifier checks
generated artifact formats when it loads them.

## Configuration and layout

All paths are relative to `run.project_root` unless absolute.

| Setting | Default |
| --- | --- |
| `run.outputs.model_path` | `models/<model_name>.pt` |
| `run.outputs.report_path` | `outputs/training/<model_name>/report.json` |
| `run.outputs.phase_directory` | `outputs/training/<model_name>/phases` |
| `run.recovery.resume.checkpoint_path` | `checkpoints/runs/<model_name>/<model_name>-latest.pt` |

The final model and recovery checkpoint paths must have a `.pt` extension. Every runner publishes this
portable model independently of the optional
best/last, embedding and ONNX exports controlled by `run.export`. Supervised
training retains its existing `<model>-best-<epoch>` and `<model>-last-<epoch>`
exports. The canonical final model follows the final executed phase's
`model_selection.handoff`.

Every completed phase publishes `<phase_directory>/<index:03d>-<name>/model.pt`
and `report.json`. Each group in a mixed plan stores its native recovery checkpoint
in the first phase's `run.pt`. The coordinator publishes each selected model and
report after committing the group's progress. Each phase keeps its snapshots in
its own `checkpoints/` directory. Policy snapshots use `transaction-NNNNNNNN.pt`
and supervised snapshots use model/epoch filenames. The rolling run checkpoint
uses the configured checkpoint path.

Phase trees, logging directories and collection output directories are reserved
output namespaces. Model, report, checkpoint and manifest files must be distinct
and cannot overlap reserved namespaces or source data. Validation runs before
publication, including generated phase destinations and symlink escapes.

## Portable models and recovery

Portable models use the `sequifier_model` artifact type.
Their execution contract embeds interfaces, codecs, normalization, policy
bindings and units. Policy execution reads units from the selected interface;
provenance is not a second source of execution configuration.

All newly written run checkpoints have the same envelope:

```text
artifact_type: sequifier_run_checkpoint
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

Sequifier writes a run manifest at the configured report path and at
`checkpoints/<model_name>.json` under the project root, including when the rolling
checkpoint uses a custom path. Common fields identify the logical run, current session,
status, completion reason, final model, checkpoint, exports and completed phases.
Ordinary budget completion uses `completion_reason: normal_completion` for all
runners and phase reports. Earlier reports can also contain `completed` for this outcome.
Additional fields contain runner-specific provenance.

Each phase report includes phase index, name, algorithm,
model and checkpoint paths, monitor/direction, handoff, selected metric and
transaction, selected weight hash, clocks, metrics, provenance and stop scope.
Unavailable measurements are null. Algorithm-specific metrics keep their names
and units; a reward monitor is not represented as a loss.

## Logging and scalar metrics

Every native runner logs training start, progress and completion to stderr and
the model-scoped log under `logs/<model_name>`. Policy phases also log their
name and algorithm when they start.
Structured scalar observations use the supervised writer's `*-training.csv`,
`*-training-full.csv`, `*-validation.csv` and `*-validation-full.csv` tables.
Condensed and full tables identify the phase, algorithm, iteration and run-wide
`global_step`. Full tables also include session IDs and UTC timestamps. When
continuing a log with the older condensed header, new condensed rows go to
`<prefix>.phases-training.csv` and `<prefix>.phases-validation.csv`. The original
files retain the byte positions referenced by older recovery checkpoints. Named scalar
observations use `metric`, `target`, `value` and `is_primary`; policy observations
use `target: __total__`. Policy rows name the dataset and view in `dataset` and
`part` when the observation can be attributed to each. Online observations leave
both empty. The phase report records the policy interface in `provenance.model_interface`. Imitation rows record the sampler epoch (zero at initial evaluation, then one-based).
Within a `run_id`, `global_step` advances once per supervised training batch or
completed policy iteration. It carries across phase transitions, optimizer resets,
and checkpoint recovery. Metrics from the same training position share a step,
including an evaluation before the next update. Sparse logging can skip step
numbers. Phase reports record the final value in `clocks.global_step`, alongside
the algorithm's own training counters.

Run `python tools/extract_validation_losses_to_table.py <model-validation-full.csv>`
to write a sibling `-wide.csv` with one row per evaluation. It keeps the run,
phase, algorithm, dataset, progress and timing fields. Each recorded metric and
target has a column. The primary metric, target and value also appear together.

| Phase | Validation measures available in the wide table |
|---|---|
| Supervised | Total and per-target loss, baseline loss, and per-target accuracy when recorded |
| Policy imitation | Held-out action negative log likelihood |
| IQL | Held-out action negative log likelihood, Bellman residual, value estimate and minimum Q estimate |
| DPO | Preference accuracy, margin, DPO loss and chosen/rejected sequence lengths |
| PPO and GRPO | Completed return and length, return spread and range, censoring, timeouts, success rate when available, and GRPO score when evaluated |

An evaluation with an unavailable policy monitor retains its diagnostics and
leaves the primary value empty. An empty metric cell in a mixed run means the
evaluation has no recorded value for that measure.

Iteration-budgeted policy phases leave `epoch` empty. Policy plots use the
committed iteration for progress. Nested diagnostic records remain in phase reports.

`run.logging.log_interval_batches` controls supervised training observations;
`run.logging.log_interval_iterations` controls policy training observations and
defaults to 10. Mixed plans may configure both. Evaluations are written whenever
the phase evaluation schedule runs, and final policy training progress is always
written. Each policy training observation describes one sampled iteration.
New tables use `seconds_per_iteration` for the elapsed policy iteration, including
collection and optimization. Policy rows use `iteration` and `iterations_total`
for progress. Their batch fields are empty. Supervised training rows use
`seconds_per_batch`, `batch`, `batches_total` and `window_batches`.
Resuming an existing table with the previous header preserves its original columns
and timing convention so checkpoint byte offsets remain valid. Validation `elapsed_seconds` measures the
evaluation itself. Condensed policy tables contain the primary training metric
or available evaluation monitor. Full tables include secondary metrics, numeric
status flags and non-actor optimizer learning rates. An unavailable evaluation
monitor is identified in the operational log. Policy iteration clocks count committed updates, even when an iteration
contains several optimizer steps. Policy visualizations display the actual
monitor names on a linear scale and do not invent baseline measurements.

Hyperparameter search accepts supervised plans. Policy and mixed plans are not yet
supported. Its default objective is the final phase's evaluation monitor; the
search direction must match. Explicit output destinations are isolated per
sampled run.

## Inference and collection outputs

Forecast and policy predictions share `sequenceId`, `subsequenceId`,
`windowStartOffset`, `itemPosition` and named prediction columns. Independent policy rows use distinct sequence IDs and item position zero. `windowStartOffset` is
relative to the subsequence. Each policy decision materializes a single window
in its subsequence, so its offset is zero. Categorical probabilities
are separate target-specific files under
`outputs/probabilities/<model>/<target>/part-NNN.<format>`, with decoded category
labels as column names. Policy predictions additionally include
`action_log_probability` when probabilities are requested. Conditional policy
probabilities remain conditional on the selected preceding actions.

All inference routes accept `output.path` for one model. A relative path resolves
under `run.project_root`. An absolute path is used directly. It overrides the first
prediction/embedding file; later chunks append `-part-NNN` to its stem. Probability
files keep their canonical target-specific destinations. Policy prediction writes
up to 256 decisions per part, the same chunk size as indexed forecast windows.
Every categorical probability part follows its prediction part’s row order.
The inference batch size controls model execution independently of file boundaries.

Forecast and policy rollout use the shared validated trajectory record.
Training-time PPO and GRPO publish a sequence manifest with a single Parquet
decision table. The `__collection` struct holds algorithm-specific diagnostics.
The manifest, supervised preprocessing and policy import share the same row
coordinates.
