# Implementation plan: named depth layouts, PT/ONNX, and encoder controls

This standalone plan supersedes the previous named-depth-layout plan. It preserves joint deep/shallow preprocessing, reusable transformer layers, flat-data compatibility, and initial PT/ONNX export and inference. It incorporates the five review findings: complete export-preflight fingerprints, RNG-isolated preflight, external mask gap validation, canonical masked-value sanitization, and backward/optimizer acceptance evidence.

Depth ingestion receives the same semantic initialization and freezing controls as the temporal backbone, with explicit composite-branch initialization precedence and freezing. Optional component initialization seeds are available consistently to the backbone and ingestion components. Runtime dropout uses the existing seeded, checkpointed run RNG lifecycle. All APIs and YAML additions below are proposed; this document does not claim implementation or demonstrated exporter compatibility.

## Implementation amendments (2026-09-10)

The implementation request explicitly forbids writing or running tests. Acceptance
conditions below remain the required future validation work; the handoff records
static review only and does not claim experimentally established compatibility.

Nested composites are in scope across parsing, recursive resolution, module
construction, initialization/freezing ownership, and width calculation. A nested
composite requires `output_dim`; the root retains its historical inferred width
when omitted. Sharing is checked at each composite between its immediate child
subtrees. Unused/auxiliary-input policies apply to the root interface only.
Per-feature temporal positions occur once at leaf output, and global temporal
positions remain in the backbone. Nested merging adds no temporal positions.

Frozen gradients must be cleared before `GradScaler.unscale_` records overflow,
while backward can traverse frozen modules. Acceptance must independently cover
finite active gradients and a frozen-only infinite gradient, plus momentum,
weight decay, accumulation boundaries, and resumed optimizer association.

ONNX dropout acceptance must isolate depth attention, temporal attention,
feed-forward, and pooled-output sites. Each fixture must disable other dropout
sites and establish both retained graph behavior and a statistical effect at
that site. Aggregate variation is supplementary evidence only. New graphs carry
a fixed evaluation/stochastic mode: `export_with_dropout` and `infer_with_dropout`
must match, otherwise inference errors. PT can switch modes. Legacy graphs keep
the existing fallback and cannot guarantee a missing dropout capability contract.

## 1. Scope and completion contract

The supported flow is:

```text
Repeated raw rows
  → complete logical outer items
  → mapped/normalized shallow scalars and fixed-capacity depth arrays
  → temporal windows plus explicit masks
  → one depth transformer per ingestion branch
  → one vector per outer item
  → existing temporal backbone and shallow decoder
  → portable PT or ONNX model
```

Initial preprocessing accepts zero or one raw depth layout. Metadata, tensor storage, loading, interface resolution, ingestion, PT artifacts, and ONNX artifacts support any number of named layouts, each with its own capacity. Multiple-layout input can initially come from the tensor payload API; separate raw child sources remain future work.

Support joint deep inputs, shallow inputs, and shallow targets; causal/next-item training; single-pass generative inference; and item-level embedding inference. Both inference modes work with PT and ONNX artifacts, including composite ingestion and multiple layouts. Preserve existing target offsets, prediction positions, categorical decoder codecs, and selected temporal embedding layers.

Deep targets, BERT objectives consuming deep inputs, autoregressive generation consuming deep inputs, and `mask_column` with depth preprocessing remain outside this feature. These are objective/data-contract restrictions independent of model artifact format. Existing flat behavior, including flat BERT and autoregressive inference, remains supported.

“Full ONNX support” means every depth architecture admitted by the new configuration has an exportable execution path, with named depth-mask inputs and functioning generative and embedding inference. It does not mean adding the excluded objectives, exporting preprocessing into ONNX, or accepting arbitrary temporal/depth capacities in one model.

No separate test-suite implementation workstream is specified. Each implementation step supplies reproducible acceptance evidence for its contracts, including backward execution and optimizer updates. Compatibility, training, and numerical acceptance conditions define completion; neither a written export file nor forward-only parity is sufficient.

## 2. Named layout configuration

Add `src/sequifier/config/depth_layout.py`:

```python
class DepthLayoutModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_column: str
    columns: list[str] = Field(min_length=1)
    context_length: int = Field(gt=0)
    position_base: Literal[0, 1] = 0
    allow_gaps: bool = False

class DepthLayoutRegistryModel(RootModel[dict[str, DepthLayoutModel]]):
    root: dict[str, DepthLayoutModel] = Field(default_factory=dict)
```

Reject duplicate feature names within a layout and membership in multiple layouts. Reject any position column that is also a layout feature, `sequenceId`, or `itemPosition`. Layout feature columns must not be outer coordinate columns. Validate module keys against actual module-storage constraints, including attribute collisions if names are used in a `ModuleDict`; identifier syntax alone is insufficient.

Expose `deep_columns`, `column_to_layout`, `layout_for_column`, `relevant_layouts`, and `is_deep_column`. Keep deterministic serialization and define whether each returned collection is ordered. Ingestion column order determines feature concatenation; registry insertion order must not silently alter execution or compatibility signatures.

Add an empty-default registry to preprocessing and dataset metadata. Missing legacy fields mean `depth_layouts={}` and `tensor_payload_version=1`.

## 3. Preprocessing configuration and activation

Depth preprocessing requires `write_format: pt`, `merge_output: false`, at most one layout, and no `mask_column`. Explicit `selected_columns` must contain the configured layout features. Read the position column automatically; exclude it from feature selection, feature statistics, and `column_data_types`.

Every source file must contain the configured position column. Reusing `metadata_config_path` requires an identical complete preprocessing layout definition, in addition to existing mapping/type compatibility checks.

A column called `subItemPosition` does not activate depth behavior. Preserve flat datasets that use this name as an ordinary feature. When flat validation finds duplicate outer coordinates, inspect the source schema when necessary—even if feature selection omitted `subItemPosition`—to provide a targeted configuration suggestion.

```yaml
write_format: pt
merge_output: false
selected_columns: [accountType, subitemType, subitemAmount, nextAction]
depth_layouts:
  subitems:
    position_column: subItemPosition
    columns: [subitemType, subitemAmount]
    context_length: 16
    position_base: 0
    allow_gaps: false
```

## 4. Raw validation and position semantics

Preserve existing supported outer identifier handling and signed-Int64 storage requirements. Validate integer outer item and depth positions, coordinate nulls, selected-value nulls/non-finite values, and representable coordinate ranges before arithmetic or narrowing casts.

For depth position `p`, require `position_base <= p < position_base + context_length`; tensor slot is `p - position_base`. Reject duplicate outer/depth coordinates. Sort positions and, with `allow_gaps=false`, require slots `0..n-1`. Unused tail capacity is always allowed. With gaps enabled, preserve physical slot positions; do not compact occupied rows.

Validate continuity of distinct outer `itemPosition` values within a complete sequence, not on repeated raw positions. For every shallow column, require exactly one distinct value per outer item, including eventual targets. Perform consistency validation before lossy transformations can make inconsistent values appear identical.

The initial repeated-row adapter requires at least one child for every real outer item. A row with missing child values is not an encoding for an empty collection.

## 5. Explicit preprocessing lifecycle

Separate logical grouping from dense materialization. Replace the original ambiguous “canonicalize, then map before scattering” sequence with:

```text
Discover source coordinates and complete item/sequence boundaries
  → select complete logical items under max_rows
  → validate selected complete sequences/items
  → collect per-column observations
  → finalize or load categorical mappings and real statistics
  → map/normalize and cast selected real observations
  → scatter into canonical dense arrays
  → compute splits, worker boundaries, and stored windows
```

Use a lightweight logical item index or grouped raw representation before dense arrays exist. Keep `CanonicalItemFrame` for the final one-row-per-item representation only. Its columns contain shallow scalars, fixed-size deep arrays, and fixed-size boolean masks.

For depth data, `max_rows` counts logical outer items. Choose items deterministically by `(sequenceId, itemPosition)` under the existing identifier ordering. Select a prefix of that ordered item index; gather every raw row for each selected item across all files. This may read later files to complete an already selected item. Do not apply file-local raw-row truncation first. Preserve existing flat `max_rows` behavior.

Compute depth-run mappings/statistics only from selected logical items. Preserve the current preprocessing policy of collecting statistics before split extraction; changing to train-only statistics is a separate behavior change. Schema and coordinate discovery may inspect all files, while selected-value validation and observation collection concern the selected population.

Fragment detection must precede per-file duplicate/continuity checks that would reject legitimate fragments. Files containing part of an item or sequence must feed the same logical aggregation. Count a shallow item once even if its children span several files.

Do not materialize the entire folder densely merely to enforce `max_rows`. Use coordinate scans and bounded item/sequence processing, with temporary partitioning/spill when source fragments exceed the memory budget. Statistics and materialization can be separate passes over the selected source population. Dense memory scales with the current processing chunk and layout capacities, rather than the complete folder.

## 6. Observation populations and dense fill rules

Refactor statistics around column-specific observation views and counts. Shallow columns contribute once per selected outer item; deep columns contribute once per occupied child slot. Apply the same views to categorical vocabulary construction and precomputed-map handling. Unoccupied slots never affect statistics or vocabularies.

After mapping and normalization, cast occupied values to the final configured dtype and reject non-finite results caused by narrowing overflow. Then scatter them into arrays. Define padding independently from raw data values:

- Categorical slots use the existing valid categorical padding-token ID; validate that it is within the feature vocabulary.
- Real slots use finite `0` in the stored output dtype, after normalization.
- Masks use `bool`, where `true` means an actual child exists.
- Temporal left padding fills every layout mask with `false` and fills feature arrays with the same safe values.

Masked slots must still satisfy storage/input validation: legal categorical indices and finite real values in the declared input dtype. Canonical writers emit the padding values above; external callers may supply other legal, finite masked values. Attention masking does not excuse invalid inputs. The model additionally sanitizes masked values before feature computation, as specified in the forward-computation section; finiteness alone does not guarantee safe intermediate arithmetic.

The storage/model contract permits an empty collection for an individual layout on a real outer item, to support independent future child sources and externally authored multi-layout payloads. Its mask is all false and its representation is learned through CLS-only encoding. The initial raw adapter cannot produce this case. Missing required masks are errors, never implicit empty collections.

## 7. Splitting and temporal window extraction

Window lengths, strides, within-sequence split ratios, and worker boundaries count logical outer items. Between-sequence splitting remains sequence-level. `allow_sequence_splitting` permits boundaries between outer items, never between child rows. Preserve existing window placement, left padding, target-offset reserve, subsequence IDs, and start-position calculations.

Add direct extraction from `CanonicalItemFrame` to tensors, bypassing the long Polars window representation for depth data. For sequence length `L`, extract shallow `[L]`, deep `[L,D_layout]`, and mask `[L,D_layout]` arrays; apply identical temporal starts/padding to all of them. Accumulate bounded batches by split and save through the central serializer. Different layouts need not share a depth capacity.

Flat CSV/Parquet and flat PT preprocessing keep their existing extraction paths.

## 8. Central PT payload and metadata

Add `src/sequifier/io/pt_payload.py` with a `StoredTensorBatch` containing `sequences`, `sequence_ids`, `subsequence_ids`, `start_item_positions`, `left_pad_lengths`, and `depth_valid_masks`.

Flat preprocessing continues writing the existing five-element tuple. Depth output writes:

```python
{
    "format": "sequifier_tensor_batch",
    "version": 2,
    "sequences": {"shallow": Tensor[N,W], "deep": Tensor[N,W,D]},
    "metadata": {"depth_valid_masks": {"subitems": Tensor[N,W,D]}},
    "sequence_ids": Tensor[N],
    "subsequence_ids": Tensor[N],
    "start_item_positions": Tensor[N],
    "left_pad_lengths": Tensor[N],
}
```

Centralize loading, legacy adaptation, saving, concatenation, and validation. Validate against expected dataset layouts: feature rank, common `N/W`, per-layout `D`, required mask names, boolean masks, metadata lengths/types, valid padding lengths, categorical ranges, and finite values. Require all depth masks to be false in temporally padded slots. For every supplied layout with `allow_gaps=false`, require each depth mask to be a prefix of true entries followed by false entries. An all-false mask is valid, including on a real outer item; `[true, false, true]` and `[false, true, false]` are invalid. With `allow_gaps=true`, arbitrary occupied physical slots are valid. Enforce this rule on externally authored payloads and direct runtime inputs as well as raw preprocessing; it is part of the portable layout contract. Validate against dataset layouts when validating a complete stored payload and against relevant layouts when validating a selected execution view.

Reject unsupported versions and inconsistent schemas instead of silently adapting them.

Keep payload version, existing stored-window-layout version, and model-artifact version as separate concepts. Add layout metadata and payload version to preprocessing metadata, split-folder metadata, inline metadata extraction, resume manifests, training/inference resolution, and PT artifacts. Normalize omitted legacy fields before resume comparison or hashing.

Define a portable selected-interface data-compatibility signature containing relevant deep feature membership, layout name, capacity, position base/column, and gap policy. Compare it alongside existing type, vocabulary, normalization, and temporal-window compatibility checks. Unused layouts and unused feature additions within a stored layout do not invalidate an otherwise compatible selected interface. Full preprocessing metadata reuse remains stricter. This data-compatibility signature is not the export-preflight fingerprint; the latter also identifies the entire resolved graph and export environment.

## 9. Dataset loading and batch contracts

Update eager and lazy PT folder loaders and every PT inference reader to use `load_pt_payload`. Preserve worker/rank allocation, sharing/pinning, sampling order, and resume positions. Load/share/pin masks with their corresponding features; lazy batches must not accidentally retain masks from another file.

Extend `build_window_batch` to accept the mask registry. Gather masks using exactly the feature input slice, not the target slice. Retain trailing dimensions and the existing `sample_valid_mask` behavior.

```text
inputs:    shallow [B,T], deep [B,T,D_layout]
targets:   shallow [B,T]
metadata:  attention_valid_mask [B,T]
           target_valid_mask [B,T]
           depth_valid_mask:<layout> [B,T,D_layout]
```

Use one shared `depth_mask_metadata_key()` helper. Reserve internal mask namespaces or use temporary names that cannot overwrite user feature columns during Polars transformations.

## 10. Shared transformer configuration and capacity plumbing

Extract `TransformerEncoderArchitectureConfig` in `config/components.py` with `dim_model`, `num_layers`, attention, feed-forward, normalization, position encoding, dropout, and shared-layer groups. Move common head/divisibility, RoPE, positive-size, and layer-sharing validation into it.

Keep `BackboneArchitectureConfig` as a subclass with the existing `max_context_length` and `positional_encoding_scope` fields. Preserve its default inference, public YAML shape, full serialized field values, and backbone-specific `range`/`range_concat` rules.

Add `TransformerEncoderStack(architecture, max_context_length)` in `model/encoder_stack.py`. Explicitly pass capacity through `SequifierEncoderLayer` to `SelfAttention` and RoPE caches. The reusable architecture has no hidden dependency on `architecture.max_context_length`.

The stack owns layer construction, sharing, execution, final normalization, and optional activation capture. Position addition and temporal attention policy remain outside it. Parameterize tracing prefixes; depth execution must not emit misleading `backbone.*` sites or label flattened item/depth axes as ordinary batch/time. Preserve existing temporal trace names and embedding capture behavior.

## 11. Flat checkpoint and initialization compatibility

Let `TransformerBackbone` inherit shared stack behavior while retaining `backbone.layers.*`, `backbone.final_norm.*`, positional module keys, persistent buffers, and shared aliases. Inheritance alone is not the compatibility mechanism.

Provide a protected deferred layer-construction path so the backbone can register and construct temporal position modules before layers, in the same order as today. Preserve constructor RNG consumption and centralized initialization traversal for flat configurations. Do not add trainable modules to a flat model merely because the shared base exists.

Preserve ordered parameter names, shapes, aliases, optimizer-group membership/order, and architecture fingerprints. These protect ordinary optimizer-state resume as well as model weight loading. If preservation proves impossible, introduce an explicit named-state migration before claiming compatibility; never rely solely on successful `load_state_dict`.

The backbone repository continues to contain only the temporal backbone. Depth encoders remain interface-owned.

## 12. Depth ingestion configuration

Add `DepthTransformerIngestionConfig`, register it in the discriminated ingestion union and compiler, and support it inside composite branches:

```python
class DepthTransformerIngestionConfig(IngestionComponentBase):
    type: Literal["depth_transformer"] = "depth_transformer"
    layout: str
    columns: list[str] = Field(min_length=1)
    output_dim: int = Field(gt=0)
    architecture: TransformerEncoderArchitectureConfig
    feature_embedding_dims: dict[str, int] | None = None
    pooling: Literal["cls"] = "cls"
```

Validate unique columns, positive feature widths, exact dimension-key coverage, layout membership, interface membership, and supported depth positions (`learned`, `sinusoidal`, `rope`). Reject depth `range`/`range_concat`. Mixed real/categorical features require explicit widths. For homogeneous features, allocate widths using the existing helper with `architecture.dim_model` as the budget; preserve its divisibility/minimum-width rules. Explicit widths may sum to a different width, followed by an input projection.

`output_dim` controls the pooled branch width only. `architecture.dropout` controls depth position/transformer dropout; inherited ingestion `dropout` controls the pooled outer-item output/temporal-position stage. The two settings have distinct meanings.

Depth capacity comes exclusively from metadata. Stack capacity is `D+1` for CLS.

## 13. Depth forward computation and positions

Validate input contracts before compiled/exported graph execution using shared validators. The forward graph uses tensor operations and static configuration, without tensor-value-dependent Python validation or branching.

For each deep feature, first use its boolean layout mask in tensor `where` operations to replace unoccupied values with the configured categorical padding ID or real zero. Sanitize in the incoming dtype before narrowing casts, embedding scaling, real projections, or other feature arithmetic. Do not multiply arbitrary values by zero to sanitize them. Host validators still reject illegal indices and non-finite inputs; sanitization does not silently accept a malformed input contract. This tensor path is identical for PT, compilation, and ONNX.

Embed/project the sanitized `[B,T,D]` features, concatenate in configured column order, and project to depth model width `C`. Reshape to `[B*T,D,C]`, prepend a learned CLS token, then apply positions to the complete `[B*T,D+1,C]` sequence.

Define CLS position as `0` and physical depth slot `s` as position `s+1` for all position types. Learned and sinusoidal positions are added outside the stack; RoPE applies inside attention. Internal gaps retain their position IDs. Changing raw `position_base` does not shift normalized slot IDs.

Extend validity with an always-valid CLS key. Use a broadcastable key-validity mask `[B*T,1,1,D+1]` with a documented `true=allowed` convention. All layers exclude invalid keys. CLS ensures every query has at least one valid key, including empty collections and padded outer items. Canonical input sanitization prevents arbitrary masked feature values from overflowing before this mask is applied; attention masking alone is not that safeguard. Avoid materializing full quadratic masks unnecessarily.

Run bidirectional depth attention, select CLS, project to `output_dim`, reshape to `[B,T,output_dim]`, and apply the existing outer ingestion-position policy at branch output width. Apply temporal positions exactly once; temporal global positions remain the backbone's responsibility. Preserve the network's existing zeroing of padded outer representations before/after the temporal backbone.

Cast concatenated embeddings, CLS values, position values, and projection inputs at module boundaries using existing dtype helpers. Free parameters such as CLS need explicit runtime dtype alignment when layer-type overrides create mixed precision.

## 14. Joint ingestion example

```yaml
ingestion:
  type: composite
  branches:
    item:
      type: embedding
      columns: [accountType]
      output_dim: 32
    subitems:
      type: depth_transformer
      layout: subitems
      columns: [subitemType, subitemAmount]
      feature_embedding_dims:
        subitemType: 48
        subitemAmount: 16
      output_dim: 96
      pooling: cls
      architecture:
        dim_model: 64
        num_layers: 2
        attention:
          type: mha
          n_heads: 4
          output_projection: true
        feed_forward:
          dim: 256
          activation: swiglu
        normalization:
          type: rmsnorm
          norm_first: true
        position_encoding:
          type: learned
        dropout: 0.1
        shared_layer_groups: []
  merge:
    type: concat
```

The branches produce `[B,T,32]` and `[B,T,96]`. Existing composite merging and backbone adaptation handle the result. Existing branch-sharing and unused-input policies apply. Flat ingestion types cannot consume deep features; a depth branch cannot consume shallow features or features from another layout.

The depth branch also accepts the following sibling fields alongside `architecture` (proposed configuration):

```yaml
initialization_seed: 20260910
initialization:
  attention.qkv:
    weight: {method: xavier_uniform, gain: 1.0}
  embedding.position:
    weight: {method: normal, mean: 0.0, std: 0.02}
  free_parameter:
    weight: {method: normal, mean: 0.0, std: 0.02}
```

The same `initialization_seed` and `initialization` controls are available on `model.backbone`. In a composite, every branch can provide its own initialization settings/seed. Freezing remains a dataset training policy, illustrated in the component/branch freezing section.

## 15. Resolution, initialization, grouping, and compilation

Propagate relevant layouts through resolved interfaces, selected interfaces, selected dataset parts, runtime model views, and factory/compiler contexts. Reject deep targets and the excluded deep objectives early. Require PT when reading preprocessed deep data, regardless of whether the model artifact is PT or ONNX. Permit unused stored layouts.

Extend positional-embedding discovery in both `model/initialization.py` and `model/parameter_groups.py` for depth and pooled temporal positions, or replace attribute-name dependence with explicit semantic registration. Register cell input and pooled output projections consistently. CLS belongs to the existing free-parameter policy. Implement the explicit initialization precedence and once-per-parameter traversal specified below; the current top-level factory call is not sufficient to honor composite branch overrides. Integrate component and branch freezing through the same parameter inventory.

Retain semantic groups such as `attention.qkv`, `feed_forward.input`, and `normalization`, with component ownership stored separately. Extend optimizer-group ID derivation to use ownership for new depth transformer parameters, producing `ingestion.attention.qkv` for depth attention while preserving current flat IDs, including existing flat attention-bearing ingestion types. A blanket ownership rewrite must not change those legacy groups. Carry stable named-branch ownership separately from semantic groups and optimizer IDs. Do not introduce new semantic `LayerGroup` literals merely to express component ownership.

Build warm-up inputs from the same interface tensor contract used by ONNX export. Compile unique layers in every stack and restore shared aliases; do not compile only `network.backbone.layers`. Preserve existing outer compilation and distributed policies.

## 16. Controlled randomization and reproducibility

Controlled randomization covers configured random initialization distributions, run-seeded training/inference dropout, optional independent component initialization seeds, and checkpoint restoration. It does not add random perturbations to trained weights or reset dropout RNG on every forward call.

Keep the existing run-level `seed` and CLI `--randomize` semantics. Record the actual resolved run seed even when randomized. With all new controls omitted, use the existing constructor and initializer RNG consumption order and checkpoint RNG lifecycle. Preserve flat model values, parameter order, and subsequent RNG state under the same seed and supported execution environment.

Add an optional `initialization_seed: int | None = None` to `BackboneComponentConfig` and `IngestionComponentBase`, using one shared validation/serialization helper. Accept a documented nonnegative signed-Int64 seed range and reject booleans. Omit inactive defaults from legacy canonical serialization; initialization seeds do not belong to `TransformerEncoderArchitectureConfig` or the backbone architecture fingerprint. They do belong to authored/resolved training configuration and strict resume comparison when explicitly set.

A seed on a component isolates both its constructor randomness and its configured initialization from the run RNG. An explicitly seeded ingestion branch can be rebuilt with the same initial values despite unrelated sibling construction or initialization. Seed resolution follows these rules:

- Without an explicit seed on a node or ancestor, retain the ordinary run RNG stream.
- An explicit seed starts a component initialization namespace. Nested branches without explicit seeds derive stable sub-seeds from that namespace and their declared relative branch paths; explicit child seeds start their own namespaces.
- Use a documented, versioned stable digest over seed, path, and phase, never Python `hash()` or registry iteration position. Constructor and configured-initializer phases use separate deterministic sub-seeds. The explicit root seed has no dependence on unrelated ancestors or siblings.
- Isolate construction and custom initialization separately with RNG capture/restore. Cover Python, NumPy, Torch CPU, and any accelerator RNG actually touched; do not reset or allocate unrelated devices merely to save state. Keep normal model construction on CPU where the existing factory does so.
- Shared parameter objects are constructed/initialized once by their canonical owner. Reject conflicting seed or initialization requests through other aliases. Existing flat shared-layer construction remains unchanged when the new controls are absent.

Seeded construction must operate on the complete owned module, including feature encoders, CLS, position modules, transformer layers, final normalization, and projections. For seeded composites, give merge-owned modules and the separate ingestion adapter stable ownership/phase identities so branch insertion does not perturb independently seeded siblings. Inspect optional seed policy outside any compiled forward; no per-forward global RNG swapping is introduced.

Runtime dropout remains governed by the run RNG and saved per-rank states, analogous to the backbone. `architecture.dropout` and ingestion `dropout` remain independently configurable for every depth branch. No independent per-branch runtime dropout generator is added: a component's `initialization_seed` controls initial parameters, not dropout draws. Same-seed runtime reproducibility assumes the same execution/sampling order, batching, hardware, and supported toolchain; cross-provider bitwise identity is not promised. Distributed model initialization must agree across ranks; rank-specific stochastic runtime behavior follows existing policy.

Restore RNG after reconstruction, compilation warm-up, and other startup work and before resumed sampling/training, using the existing checkpoint ordering. Warm-up and validation/export preflight must preserve the training RNG, including when they fail. Portable model artifacts preserve initialization configuration as provenance but load saved weights without reapplying it. Exact run checkpoints additionally contain the runtime RNG states and the seed-derivation version when used.

PT inference uses the existing inference seed policy. For stochastic ONNX inference, specify the supported ORT seed/session-creation ordering and repeatability boundary for the declared toolchain. Seed once at the appropriate session/run boundary, not on each batch. Check that repeated calls advance randomness and that new sessions initialized under the documented seed policy reproduce the supported behavior. ONNX RNG is not a serialized PT RNG stream; PT/ONNX dropout draws need not match. Preserve evaluation and stochastic-export behavior independently.

## 17. Initialization configuration and composite precedence

Depth ingestion inherits `ModelInitializationConfig` directly, providing the same methods and method parameters as the backbone: normal/uniform, Xavier/Kaiming variants, constant/zeros/ones, preserve, and other currently supported methods. Do not create a second depth-only initialization schema. The existing canonical recursive hyperparameter search must accept and validate depth architecture fields, branch initialization settings, and initialization seeds.

Resolve effective configuration before constructing the initialization worklist. An ingestion-level override applies to matching parameters in its subtree; a more specific composite branch override takes precedence for the same `(semantic group, weight-or-bias)` target. Unspecified child targets inherit their nearest configured ancestor. An empty mapping inherits; an explicit `preserve` override suppresses inherited initialization for that target and keeps its constructed value. Parent settings still apply to composite merge-owned modules. The separate ingestion adapter inherits the top-level ingestion initialization policy, not an arbitrary child's policy.

Add a canonical ownership/initialization worklist in the factory or initializer so every parameter is initialized exactly once under its resolved policy and seed. Do not initialize the parent subtree and then initialize child subtrees again. Retain the original initialization traversal and RNG draw order for configurations that have no new branch overrides or seeds. Existing configured branch overrides that were previously ignored must now work; document that correction and reject incompatible exact resume rather than silently changing initialization provenance.

Use the following semantic assignments consistently for initialization, dtype handling, freezing, and parameter reporting:

| Depth module/parameter | Existing semantic group |
| --- | --- |
| Categorical feature embeddings | `embedding.input` |
| Per-real-feature projection | `real_feature_projection` |
| Learned depth and pooled temporal positions | `embedding.position` |
| Attention Q/K/V projections | `attention.qkv` |
| Attention output projection, if present | `attention.output` |
| Feed-forward input/gate projections | `feed_forward.input` |
| Feed-forward output projection | `feed_forward.output` |
| Layer norms and final normalization | `normalization` |
| Concatenated-cell input and pooled output projections | `ingestion.output_projection` |
| CLS | `free_parameter` (configured via the existing `weight` policy) |

Sinusoidal and RoPE positional buffers have no trainable initialization or freezing target. Classify only modules that exist; identity output projections have no parameters. Semantic groups describe parameter function; component/branch ownership identifies where the parameter belongs. Shared layers have aliases but one parameter identity, one initialization, and one optimizer state.

Report unmatched initialization overrides in their effective ownership scope, following the existing warning/error policy. An override intended for one branch must not appear successfully matched merely because a sibling or backbone has that semantic group. Validate unsupported branch names and ambiguous ownership early.

## 18. Component and named-branch freezing

Keep dataset-scoped `freeze.backbone`, `freeze.ingestion`, `freeze.decoder`, and `freeze.ingestion_adapter`. The entire depth encoder is owned by ingestion, so freezing backbone attention cannot freeze depth attention, and ingestion attention policies cannot freeze the backbone. Preserve existing semantic `freeze`/`freezing_except` meanings and mutual exclusion.

Extend the ingestion freezing policy with a recursive `branches` mapping keyed by declared composite branch names. A proposed shape is:

```python
class IngestionFreezingConfig(LayerFreezingConfigFields):
    branches: dict[str, "IngestionFreezingConfig"] = Field(default_factory=dict)
```

Use this model for `DatasetFreezingSpecModel.ingestion`; omit empty `branches` when serializing legacy configuration. Branch names are resolved against the selected interface's actual ingestion tree. Reject unknown names and child policies below non-composite leaves. Recursive mapping avoids ambiguous string paths and supports nested composites.

Distinguish a node's local policy presence from recursive policy activity. Update dataset `active` checks, normalization, YAML serialization, and policy resolution so a branch-only policy activates freezing without interpreting its parent as `freezing_except: []`. A parent with neither local selector delegates to its children and keeps inherited decisions for its own remaining scope.

Each explicit policy computes trainable/frozen decisions for every parameter in its subtree. A more specific branch's active `freeze` or `freezing_except` policy replaces the inherited decisions for that branch, allowing selective unfreezing as well as freezing. A branch containing only a nested `branches` map leaves other inherited decisions intact. `freeze: []` explicitly unfreezes its scope; `freezing_except: []` freezes the entire scope. Parent policy continues to govern merge-owned parameters; the separate ingestion adapter remains controlled by `freeze.ingestion_adapter`.

Example dataset freezing policy:

```yaml
freeze:
  backbone:
    freeze: [attention.qkv]
  ingestion:
    branches:
      subitems:
        freezing_except: [free_parameter, ingestion.output_projection]
      item:
        freeze: []
  ingestion_adapter: false
```

Here the depth branch trains only CLS and its cell/pooled projections, while the shallow `item` branch remains trainable. To freeze the complete `subitems` branch, use `freezing_except: []`. To freeze just its attention and feed-forward blocks, enumerate their existing semantic groups. This exposes full encoder freezing as well as backbone-equivalent group selection without adding semantic literals for ownership.

Resolve policies to canonical parameter IDs before optimization, then bind to identities after wrapping/compilation. Deduplicate shared aliases. If distinct ownership paths make contradictory decisions for the same shared parameter within one dataset policy, reject the policy and report both paths; it cannot be frozen through one alias and trainable through another. Different datasets may legitimately have different decisions and must retain the existing per-dataset update behavior. Keep existing warnings for unmatched `freeze` groups and errors for unmatched `freezing_except` groups, evaluated within the requested branch scope.

Use `requires_grad=False` only for parameters permanently frozen under the runtime's applicable dataset policies. For source-specific freezing, preserve gradient propagation through the frozen encoder to any trainable upstream components; do not wrap the encoder in `no_grad()` or detach its output. Apply update suppression through the existing optimization policy so frozen parameters receive no optimizer, momentum, or decoupled-weight-decay update. Freezing does not change `.train()`/`.eval()` or disable dropout; users control dropout explicitly and its RNG advances according to execution mode.

Audit gradient accumulation boundaries: an optimizer step must not combine microbatches with incompatible freezing policies and then apply only the last policy. Preserve existing valid scheduling; if such mixing is possible, split/flush the accumulation boundary or reject the incompatible schedule before training. Avoid silently discarding previously accumulated trainable contributions.

Preserve optimizer group ordering/IDs for flat models and once-per-identity membership for shared depth parameters. Persist resolved freezing selections and ownership in run checkpoints/resume manifests under existing version conventions. Verify transitions between dataset policies and restoration of optimizer state; changed policies must follow existing explicit resume compatibility rules. Portable inference artifacts retain useful model/configuration provenance but do not require dataset freezing policies to execute.

## 19. Shared execution input/output contract

Add a small artifact/runtime schema module, for example `src/sequifier/artifacts/execution_schema.py`. Derive one ordered contract from the selected interface and use it for dummy inputs, ONNX wrapper binding, validation, and inference feeds.

Each input descriptor records graph name, feature or metadata role, source key, dtype, rank, and axis constraints. Order features by interface declaration and required layout masks deterministically. Include only metadata used for execution; targets and training-only masks are not model inputs.

Preserve legacy feature names such as `<column>_in` where possible and detect name collisions before export. Store an explicit graph-name-to-source-key mapping so consumers need not infer roles from suffixes. Record output names, target/layer identity, dtype, axis order, and interpretation: categorical log probabilities, real predictions, or embeddings.

New depth ONNX graphs use symbolic batch size shared across all feature/mask inputs, fixed temporal context `T`, and fixed `D` per layout. This permits an unpadded final inference batch. Static-batch legacy graphs remain supported. Export with a representative batch of at least two to avoid accidentally specializing the symbolic batch dimension to one; actual inference must accept batch one as well.

Keep temporal/depth capacities static because they determine learned positions and model context semantics. Variable occupied depth is expressed exclusively through masks.

## 20. PT model artifacts

Extend `artifacts/model_export.py`, `artifacts/model_config.py`, and artifact metadata validation to serialize relevant layouts and the execution contract with ingestion configuration. Reconstruct depth interfaces without access to training data or the original training YAML.

Do not confuse model execution with data-reader requirements: the current execution-only reconstruction hardcodes `read_format="parquet"`. Replace that placeholder appropriately for deep artifacts or separate execution validation from reader validation so it cannot falsely reject a portable deep model.

Keep legacy artifact defaults, categorical codecs, normalization metadata, selected embedding layers, target offsets, and temporal storage/view metadata. Explicitly version new required artifact contracts using the repository's artifact-version conventions. Preserve effective component/branch initialization settings and explicit initialization seeds as provenance; execution-only reconstruction must never initialize over loaded trained weights. Strict run checkpoints additionally preserve branch ownership, freezing selections, and RNG provenance, separately from the portable execution schema.

## 21. ONNX wrapper and graph export

Extend `export/onnx.py` to construct tensors from the execution contract: shallow `[B,T]`, deep `[B,T,D]`, outer validity `[B,T]`, and required depth validity `[B,T,D]`. Dummy categorical values must be legal vocabulary IDs. Use safe occupied/padded examples rather than assuming integer one is always a valid category.

Replace `_OnnxWrapper`'s `values[:-1]`/single-last-mask convention with descriptor-driven feature and metadata binding. Export generative and embedding wrappers through the same input binding. Retain existing sequence-major output conventions and make the output descriptors authoritative for consumers.

Use the existing `dynamo=True` exporter with explicit shared batch `dynamic_shapes`. Start with the existing opset 18 and standard operator decompositions. Depth support must not depend on adopting a newer fused ONNX Attention operator. Resolve unsupported translations with an equivalent export lowering for the shared attention/normalization operations, not by omitting masks or silently changing the architecture.

Cover admitted MHA/MQA/GQA, learned/sinusoidal/RoPE positions, LayerNorm/RMSNorm, supported feed-forward activations, projection settings, shared layers, composites, and differing layout capacities. Export paths must preserve scaling, mask polarity, dtype casts, and dropout semantics. Avoid changing eager flat attention behavior to accommodate one exporter.

Keep normal ONNX execution FP32 using the existing export-copy conversion policy; extend conversion to new modules/free parameters/buffers. Training and PT can retain layer-specific dtypes. Record actual exported dtypes and compare against the corresponding FP32 export reference. Audit registered dtype-dependent buffers when converting the copy, not just trainable parameters. Unsupported provider/precision combinations must have explicit capability errors; they are not a reason to blanket-reject depth.

Preserve `export_with_dropout` and `infer_with_dropout` behavior for depth as well as temporal layers. Deterministic output parity uses evaluation mode. Stochastic exports require retained runtime dropout and stochastic-behavior validation, not bitwise comparison of unrelated random draws.

## 22. Embedding export and artifact publication

Audit `export/embedding.py` and the shared stack extraction together. Existing embeddings select temporal backbone activations; depth encoding must execute before those captures. If generic tracing objects prevent export capture, provide a static activation-selection path using the same layer loop and the same selected sites. Do not silently fall back to final-layer-only embeddings.

New ONNX artifacts store a versioned execution schema and relevant layout definitions in model metadata, alongside existing categorical decoder codecs and embedding-layer names. Include selected-interface column types, normalization/vocabulary metadata needed for existing inference resolution, context/target-offset semantics, and output contracts. Embed model execution metadata consistently with PT rather than requiring an unrelated training config.

Cross-check the metadata contract against the actual exported graph input/output names, shapes, and dtypes. Handle graph-pruned optional/unused inputs explicitly: retain full logical interface metadata but bind feeds to actual graph inputs. Required used depth masks must survive export.

Extend `ExportService` for each selected interface and both output modes, including runs containing flat and deep interfaces. Keep ONNX defaults enabled; no depth-specific ONNX rejection is introduced. Perform configuration/toolchain checks before training, and representative generative/embedding export preflight before a long depth training run when those modes are requested. Use a separate, versioned export-preflight fingerprint; never cache by the selected-interface data-compatibility signature alone.

The fingerprint includes the entire resolved graph-affecting configuration: temporal and every depth architecture; feature widths and input/output projections; composites and sharing; selected decoder/objective/prediction semantics; execution input/output schema and ordering; fixed capacities and symbolic-batch policy; selected embedding sites; actual export dtypes and conversion policy; dropout probabilities and evaluation/stochastic mode; exporter options, opset, decompositions and custom lowerings; Sequifier/exporter implementation identity; exact PyTorch/ONNX/ONNXScript/ORT versions; and provider/session settings and relevant device identity. Canonicalize deterministically and include the preflight fixture/validation-protocol version. Record the stochastic seed policy for stochastic validation. Dataset path, unrelated unused layouts, training epoch, and ordinary weight values need not invalidate architecture capability preflight unless they influence graph construction.

Cache only completed successful checks and their diagnostic/validation record. A model with identical layouts but different attention, decoder, composite structure, embedding selection, or dropout mode requires a different key. Preflight caching proves capability for that graph family; each actual trained artifact still receives graph/session/output validation. Do not repeat unchanged capability preflight each epoch.

Run preflight in a disposable subprocess with its own fixed recorded seed and model whenever practical. The worker must use the resolved execution schema and the same exporter/lowerings as publication. No training model or live optimizer is passed for mutation. If an in-process path is necessary, create a disposable model and restore all touched Python, NumPy, CPU/accelerator RNG states in `finally`, on success and failure; also restore any changed deterministic/backend flags. Do not reseed a live ORT session or process-global inference RNG in a shared serving process. Verify cache hit, miss, and failed preflight leave the parent RNG and training parameters/buffers/modes unchanged. Distributed preflight must use established rank-zero coordination and propagate failures without hanging other ranks.

Publish only after graph checking, session creation, and output validation. Write to a staging location; handle any ONNX external-data files as part of the artifact, then publish the complete artifact set. Surface failures with interface, architecture, operator/export stage, and diagnostic report. Do not report successful export for a partial or unusable file.

## 23. ONNX inference

Update `infer_config.py` and `infer.py` to distinguish preprocessed input `read_format: pt` from model artifact `.onnx`. Remove the proposed deep-interface ONNX ban. Resolve layouts from model metadata and validate any supplied dataset/config metadata against it before batching.

Load both PT tensor payload versions, select required features/layout masks, and use the existing temporal sampling plan to gather aligned input windows. Pass rank-three arrays and depth masks through device/NumPy conversion without flattening the depth axis. Normalize/map only where the existing input path requires it; preprocessed payloads must not be transformed twice.

Construct session feeds from actual graph input descriptors and the stored name mapping. Validate rank, fixed dimensions, common batch size, categorical bounds, mask dtype/shape, no-gap prefix semantics where configured, and required values before `session.run`. Recheck occupied real values after conversion to the actual graph input dtype to catch narrowing overflow. Canonicalize masked values before any host-side narrowing when needed, and retain the graph-side sanitization contract. Match ONNX input dtypes explicitly; boolean masks must remain boolean. Do not infer depth validity from zero-valued features.

For symbolic batch graphs, pass the actual final batch size. For legacy static-batch graphs, preserve batch padding/repetition behavior while applying it identically to every feature and metadata tensor and trimming synthetic output rows. Handle empty datasets without calling the runtime with an invalid empty batch. Arrays are sliced along batch/time while preserving all trailing dimensions.

Use output descriptors and runtime output names rather than relying solely on alphabetical dictionary order. Preserve categorical probabilities/sampling, real-value denormalization, selected embeddings, prediction positions, and filtering by outer output validity. Both output modes remain indexed by `sequenceId`, `subsequenceId`, and `itemPosition`; there is no depth-coordinate output.

Legacy flat ONNX models without the new schema use the existing naming/metadata fallback. A new deep graph with missing or contradictory layout metadata fails clearly. Permit CPU execution and existing supported accelerator providers, checking provider availability under the established device policy.

## 24. Acceptance conditions and toolchain policy

The current dependency ranges do not by themselves prove every exporter/runtime combination works. Establish and document a supported minimum PyTorch, ONNX, ONNXScript, and ONNX Runtime combination for the chosen export APIs; update dependency constraints if needed. Verify against the declared support range rather than assuming current documentation applies to every older allowed installation.

Required completion evidence includes unchanged flat model keys, ordered parameters, initialization behavior, architecture fingerprints, and resumed optimizer association; aligned item statistics/windows across file boundaries; and PT artifact reconstruction with identical deterministic outputs.

For ONNX, require valid graph/session creation and numerical agreement with the FP32 export reference for generative outputs and every supported temporal embedding selection. Exercise meaningful masks: internal gaps, unequal occupied depths, full capacity, empty collections, temporally padded items, different masks at unchanged tensor shapes, and different capacities across layouts. Verify invalid-slot changes cannot affect valid item representations for arbitrary accepted legal categorical indices and finite real inputs, including extreme finite values that would overflow a projection without sanitization. Verify masked real inputs have zero influence on valid-output gradients. Apply no-gap fixtures only to layouts admitting them: internal-gap masks must be rejected when `allow_gaps=false`. Verify causal outputs cannot depend on future outer inputs.

Confirm symbolic batch behavior at batch one, the representative batch, and a partial final batch. Ensure both masks and features remain runtime inputs, with no example-value specialization. Keep dtype-appropriate explicit error tolerances and stochastic-export acceptance separate from deterministic parity. Memory/performance assessment must include the `B*T` depth batch and attention cost; successful export is not evidence of acceptable deployment memory use.

Training completion evidence must include a small, nondegenerate joint shallow/deep run with backward execution and an optimizer step. Confirm finite loss/gradients and actual updates to the unfrozen feature projections/embeddings, depth attention/feed-forward blocks, CLS, positions where trainable, pooled projections, and temporal/decoder parameters where configured. Select fixtures that activate each claimed trainable path; do not demand nonzero gradients from structurally inactive or unoccupied feature paths. Include independently empty layouts without losing the CLS-only gradient path.

Verify the following additional contracts with executable acceptance checks alongside their implementation work:

- Identical seeds/configuration reproduce initialized depth weights and the supported training trajectory. Explicit branch initialization seeds isolate branch initial values from unrelated sibling construction; constructor and custom-initializer isolation leave ambient RNG unchanged. Omitted controls preserve flat initialization and subsequent RNG state.
- Exercise a parent initialization override, a child override for the same group, inheritance of another weight/bias target, and `preserve`; confirm each shared parameter is initialized once. Cover CLS, both positional stages, and both depth projections. Validate initialization choices through canonical hyperparameter search.
- Run complete branch freezing, attention-only freezing, `freezing_except`, selective child unfreezing, and simultaneous independent backbone/shallow/deep policies. Frozen values remain unchanged with nonzero weight decay and preexisting optimizer momentum, while eligible values update. Check gradients still traverse frozen modules and freezing does not implicitly disable dropout.
- Exercise two dataset policies that freeze the same shared encoder differently; verify update transitions, accumulation-boundary handling, and optimizer association. Reject contradictory alias decisions within one policy. Shared layers appear once in optimizer membership.
- Compare a continuous depth training run with a checkpoint/save/reconstruct/resume run at a supported checkpoint boundary. Match selected batches, loss trajectory, model values, optimizer state, freezing selections, and stochastic RNG progression under the declared deterministic environment. Include representative mixed precision, unique/shared layer compilation, and distributed execution where supported. Check checkpoints after scaler/optimizer handling according to existing boundary semantics.
- Verify preflight cache invalidation when architecture, composite, output mode, embedding selection, dropout mode, dtype policy, provider, or toolchain changes while layouts remain identical. Verify hit, miss, and failure do not alter training RNG state or the next initialized model/dropout draw.
- Retain separate stochastic ONNX behavior checks: meaningful output variation across successive calls, documented seed/session reproducibility on supported providers, and no accidental reseeding per batch. Evaluation parity remains deterministic and does not compare unrelated PT/ORT random draws.

Record commands, fixtures, supported toolchain/provider versions, tolerances, and observed results in a completion report. Forward-only checks or manual inspection of serialized configuration do not satisfy these training controls.

## 25. Implementation order and file ownership

1. Record flat compatibility contracts and inspect toolchain/export capabilities with representative depth-shaped primitives.
2. Extract common architecture validation and implement explicit stack/layer/attention capacity plumbing; preserve backbone registration and initialization order.
3. Add layout registry, metadata defaults/signatures, and the central PT payload adapter.
4. Implement complete-item source indexing, logical `max_rows`, per-column observations, bounded materialization, and direct window extraction.
5. Extend split metadata/resume manifests, eager/lazy loaders, and batch-mask gathering.
6. Propagate resolved layouts; implement/register depth ingestion, position semantics, safe padding, and the corrected composite example.
7. Implement canonical component/branch ownership, initialization override precedence and once-per-identity traversal; add optional isolated component initialization seeds. Integrate semantic grouping, dtype boundaries, RNG-preserving warm-up, compilation, and factory/artifact reconstruction.
8. Implement recursive dataset branch freezing, shared-alias conflict handling, optimizer/accumulation policy integration, and checkpoint/resume compatibility.
9. Add shared execution input/output schemas and update PT artifacts.
10. Extend generative/embedding ONNX wrappers, dynamic batch export, metadata, complete preflight fingerprinting, isolated preflight execution, and complete artifact publication.
11. Implement schema-driven ONNX inference, PT-payload mask propagation, legacy graph fallback, and output mapping.
12. Complete flat/depth compatibility, backward/optimizer, initialization/randomness/freezing, checkpoint/resume, cache-isolation, and export acceptance evidence; document supported toolchain/provider behavior.
13. Update `documentation/configs/preprocess.md`, `train.md`, `infer.md`, `documentation/training/runtime-architecture.md`, and `README.md` with complete PT/ONNX examples, component/branch initialization and freezing examples, seed/dropout semantics, resume behavior, and migration semantics. Include the new metadata/external-payload mask rules and preflight cache contract.

Primary existing modules affected: `preprocess.py`; `config/{components,preprocess_config,metadata,train_config,infer_config}.py`; PT folder loaders and `io/window_sampling.py`; `model/{backbone,layers,ingestions,ingestion_compiler,factory,initialization,parameter_groups,parameter_catalog}.py`; `runtime/{builder,random_state}.py`; `training/{runtime,optimization,engine}.py`; `config/{freezing_config,initialization_config,canonical_hyperparameter_search_config}.py`; `artifacts/{model_export,model_config,run_checkpoint}.py`; `export/{onnx,embedding,service}.py`; and `infer.py`. New modules are the layout registry, encoder stack, PT payload abstraction, and execution schema, plus small shared ownership/initialization-seed helpers and a preflight worker/cache module where appropriate. Audit remaining metadata whitelists and direct `torch.load` call sites rather than assuming the named modules are exhaustive.

## 26. Future independent child sources

Extend source orchestration with an outer item source and one child source per layout, joined by outer coordinates. Keep absent children as an all-false mask for that layout, preserving the CLS-only contract. Validate shallow values at the item source and avoid Cartesian products between independent child collections. Existing storage/model/export contracts already support different capacities and independently empty collections.

## References for export implementation

The current PyTorch documentation describes the `dynamo` exporter, `dynamic_shapes`, diagnostics, and runtime verification options: [torch.onnx](https://docs.pytorch.org/docs/stable/onnx). Use these capabilities only within the declared supported toolchain.

ONNX Runtime exposes actual graph input/output descriptors and custom model metadata through its inference session API: [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary). Use those descriptors to validate feeds and artifacts rather than guessing graph shape or input identity.
