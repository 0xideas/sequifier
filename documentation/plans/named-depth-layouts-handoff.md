# Named depth layouts implementation handoff

Implementation date: 2026-09-10.

The implementation includes the requested nested-composite and frozen-gradient
amendments, plus explicit ONNX dropout capability handling. No tests or acceptance
fixtures were written or run, as requested. This report does not certify training,
resume, statistical dropout behavior, numerical parity, or a provider/version
compatibility matrix.

## Implemented

- A named depth-layout registry with selected-interface compatibility signatures,
  strict feature membership and module-key validation, and legacy metadata defaults.
- Disk-backed complete-item indexing across source files, logical-item `max_rows`,
  shallow consistency checks before lossy conversions, per-column observation
  populations, mapping/normalization before final real-value narrowing, and bounded
  dense window extraction into version 2 PT payloads. The initial raw adapter
  supports one layout and requires a child on every selected real item.
- Central PT loading, saving, concatenation, and validation; eager/lazy folder and
  single-file loading; aligned depth-mask gathering; payload version propagation.
  External payloads can have multiple capacities and independently empty layouts.
- A reusable encoder architecture/stack, explicit capacity plumbing, CLS depth
  pooling, physical-slot position semantics, masked-value sanitization before
  arithmetic, mixed-dtype boundaries, and interface-owned depth encoders.
- Recursive composite parsing, resolution, column-sharing validation, width
  calculation, construction, initialization ownership, and dataset freezing.
  Nested composites require explicit widths; unused/auxiliary policies belong at
  the root. Leaf temporal positions are not reapplied by composite merges.
- Optional isolated component initialization namespaces with stable versioned
  seed derivation; child override inheritance; semantic depth parameter groups;
  unique-layer compilation and RNG-preserving warm-up.
- Frozen-gradient suppression before AMP overflow detection, preserving backward
  traversal. Existing accumulation policy boundaries, momentum association, and
  optimizer transaction ordering remain in use. The implementation follows the
  overflow-detection ordering described in [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).
- Portable PT artifact version 2 with execution schemas and initialization
  provenance, backward-readable version 1 flat artifacts, strict run provenance,
  and normalization of inactive legacy fields during resume comparison.
- Descriptor-driven generative and embedding ONNX export, symbolic batches with
  fixed capacities, static activation capture using the existing encoder/decoder
  loops, FP32 export copies, explicit attention dropout lowering, metadata/graph
  cross-checks, staged external-data publication, and runtime feed validation.
- Per-site structural dropout verification: invoked dropout module scopes,
  alias-normalized invocation counts, retained ratios, and enabled training
  operands are checked and recorded, including local ONNX functions. This is
  structural evidence only; it does not replace statistical acceptance.
- Fixed ONNX evaluation/stochastic mode metadata. Mismatched inference requests
  fail. PT dropout mode switching includes attention modules as well as ordinary
  dropout modules. ORT CLI seeding occurs once before session creation.
- Disposable-process preflight and successful-result caching keyed by complete
  graph, exporter implementation, toolchain, provider/session, and protocol
  identity. Every actual publication still performs graph/session/output checks.

## Static checks performed

Python source was parsed with `ast.parse` without importing project modules.
Changed Python files were formatted and checked with locally cached Ruff 0.11.4:

```text
ruff check --no-cache --select I --fix <changed Python files>
ruff format --no-cache --line-length 85 <changed Python files>
ruff check --no-cache --select F,E9 <changed Python files>
git diff --check
```

These checks passed. They inspect syntax, imports, undefined names, formatting,
and patch whitespace. They are not execution, model, numerical, or test evidence.
No preprocessing, training, backward pass, optimizer step, model export, ONNX
session, or preflight worker was executed during this implementation.

## Validation still required outside this handoff

The original plan's acceptance requirements remain pending. In particular:

- Compare flat keys, ordered parameters/aliases, seeded initialization and
  subsequent RNG state, architecture fingerprints, and optimizer resume mapping.
- Exercise fragmented repeated rows, logical limits, split/window alignment,
  statistics, gap policies, narrowing, selected-interface compatibility, and
  single/multiple-layout external payloads.
- Verify a nondegenerate joint shallow/deep backward and optimizer update,
  independently empty layouts, frozen-module gradient traversal, nonzero weight
  decay and preexisting momentum, child unfreezing, and accumulation transitions.
- Verify finite active gradients with an overflowing frozen-only gradient under
  AMP. The intended outcome is a healthy active update without a false overflow.
- Compare continuous and reconstructed/resumed trajectories, optimizer/scaler
  state, sampling, and RNG progression, including compilation and supported
  distributed/mixed-precision environments.
- Verify generative and selected-embedding ONNX parity with meaningful masks and
  several batch sizes. Publication validation is implemented but has not run.
- Isolate depth-attention, temporal-attention, feed-forward, and pooled-output
  dropout independently. Check retained graph behavior and statistical effect
  for each site, then session/seed reproducibility. Aggregate output variation
  alone is insufficient. These requested fixtures were deliberately not written.
- Verify preflight cache hit, miss, invalidation, failure, parent RNG isolation,
  and distributed error propagation.
- Measure memory and throughput for the flattened `B*T` depth batch and its
  attention cost; exportability alone does not establish deployment suitability.

## Operational notes

The exporter checks an API minimum of PyTorch 2.6, ONNX 1.17, ONNXScript 0.5.4,
and ONNX Runtime 1.20. Source inspection included a locally cached PyTorch 2.6
exporter. These minimums have not been exercised as a compatibility matrix;
existing dependency locks were not regenerated. Older installed packages receive
an explicit exporter capability error.

The repeated-row adapter spills raw indexing to SQLite and processes dense
windows sequentially. It currently does not use `n_cores` for parallel depth
materialization. Categorical vocabularies and the lightweight sequence index
still grow with dataset cardinality. Use a fresh output location when a previous
split folder contains stale shards.

ONNX artifacts may include uniquely named external tensor sidecars. Move them
with the graph. Publication commits the graph after its sidecars, preserving a
previous graph on failure. Sidecars from a replaced graph may remain on disk.

The preflight record explicitly distinguishes its graph/session checks from the
pending focused stochastic acceptance. A cached structural capability result is
not a claim of statistical dropout or training correctness.
