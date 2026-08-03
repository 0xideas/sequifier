from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
from loguru import logger

from sequifier.model.ingestions import (
    STRUCTURED_BLOCK_HANDLERS,
    AxisShape,
    BaseFeatureIngestion,
    CompositeFeatureIngestion,
    DirectEmbedFeatureIngestion,
    FeaturePoolFeatureIngestion,
    GroupedFeatureIngestion,
    PassThroughFeatureIngestion,
    SiameseFeatureIngestion,
    StructuredFeatureIngestion,
    TemporalConvFeatureIngestion,
    _add_ingestion_position_encoding,
    _feature_dims_for_columns,
    _split_columns,
)


@dataclass(frozen=True)
class ResolvedIngestionBranch:
    """Validated branch configuration together with its tensor contract."""

    name: Optional[str]
    config: Any
    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    real_columns: tuple[str, ...]
    width: int
    usage: str
    layout: Optional[Any] = None


@dataclass(frozen=True)
class IngestionPlan:
    """Canonical internal form for both public ingestion-spec shapes."""

    branches: tuple[ResolvedIngestionBranch, ...]
    merge_type: Optional[str]
    is_composite: bool
    transformer_input_width: int


@dataclass(frozen=True)
class BuiltIngestion:
    """An ingestion module and the width it produces."""

    module: BaseFeatureIngestion
    width: int


@dataclass(frozen=True)
class IngestionBuildContext:
    hparams: Any
    direct_real_dtype_provider: Callable[[], torch.dtype]
    device_max_concat_length: int
    add_ingestion_position: bool


@dataclass(frozen=True)
class IngestionHandler:
    resolve_columns: Callable[[Any, Any, bool], tuple[list[str], Optional[Any]]]
    validate: Callable[[Any, str, list[str], Any, Optional[Any]], None]
    build: Callable[
        [ResolvedIngestionBranch, IngestionBuildContext], BaseFeatureIngestion
    ]


def _layout_for_config(hparams: Any, config: Any) -> Any:
    if hparams.feature_layout is None:
        raise ValueError(
            f"Ingestion layout {config.layout!r} requires top-level feature_layout"
        )
    if config.layout not in hparams.feature_layout:
        raise ValueError(f"Unknown feature_layout {config.layout!r}")
    return hparams.feature_layout[config.layout]


def _flat_columns_for_config(
    hparams: Any,
    config: Any,
    allow_default_columns: bool,
) -> tuple[list[str], Optional[Any]]:
    columns = config.columns
    if columns is None:
        if not allow_default_columns:
            raise ValueError(f"{config.type} ingestion branches must configure columns")
        columns = hparams.input_columns
    return list(columns), None


def _grouped_columns_for_config(
    hparams: Any,
    config: Any,
    allow_default_columns: bool,
) -> tuple[list[str], Optional[Any]]:
    _ = hparams, allow_default_columns
    return (
        [
            column
            for group_columns in config.groups.values()
            for column in group_columns
        ],
        None,
    )


def _structured_columns_for_config(
    hparams: Any,
    config: Any,
    allow_default_columns: bool,
) -> tuple[list[str], Optional[Any]]:
    _ = allow_default_columns
    layout = _layout_for_config(hparams, config)
    return list(layout.columns), layout


def _validate_ingestion_columns(hparams: Any, usage: str, columns: list[str]) -> None:
    missing_columns = set(columns) - set(hparams.input_columns)
    if missing_columns:
        raise ValueError(
            f"{usage} references unknown input columns: {sorted(missing_columns)}"
        )

    auxiliary_columns = set(hparams.model_spec.auxiliary_input_columns)
    consumed_auxiliary_columns = set(columns) & auxiliary_columns
    if consumed_auxiliary_columns:
        raise ValueError(
            f"{usage} consumes auxiliary input columns: "
            f"{sorted(consumed_auxiliary_columns)}"
        )

    typed_columns = set(hparams.categorical_columns) | set(hparams.real_columns)
    untyped_columns = set(columns) - typed_columns
    if untyped_columns:
        raise ValueError(
            f"{usage} references columns that must be declared in "
            f"categorical_columns or real_columns: {sorted(untyped_columns)}"
        )


def _validate_direct_embed_config(
    hparams: Any,
    usage: str,
    columns: list[str],
    config: Any,
    layout: Optional[Any],
) -> None:
    _ = layout
    feature_embedding_dims = config.feature_embedding_dims
    if feature_embedding_dims is not None and set(feature_embedding_dims) != set(
        columns
    ):
        raise ValueError(
            f"{usage} feature_embedding_dims must contain exactly its input "
            f"columns. Expected {columns}, got {list(feature_embedding_dims)}"
        )

    if feature_embedding_dims is not None:
        embedding_dim = sum(feature_embedding_dims.values())
        if embedding_dim != config.output_dim:
            raise ValueError(
                f"{usage} feature_embedding_dims sum ({embedding_dim}) must "
                f"equal output_dim ({config.output_dim})"
            )
        return

    categorical_columns, real_columns = _split_columns(
        columns, hparams.categorical_columns, hparams.real_columns
    )
    if categorical_columns and real_columns:
        raise ValueError(
            f"{usage} must configure feature_embedding_dims when both real "
            "and categorical variables are present."
        )
    if real_columns and config.output_dim < len(real_columns):
        raise ValueError(
            f"{usage} output_dim ({config.output_dim}) must be at least the "
            f"number of real variables ({len(real_columns)})."
        )
    if categorical_columns and config.output_dim % len(categorical_columns) != 0:
        raise ValueError(
            f"{usage} output_dim ({config.output_dim}) must be a multiple of "
            f"the number of categorical variables ({len(categorical_columns)}: "
            f"{categorical_columns})."
        )


def _validate_pass_through_config(
    hparams: Any,
    usage: str,
    columns: list[str],
    config: Any,
    layout: Optional[Any],
) -> None:
    _ = config, layout
    categorical_columns, real_columns = _split_columns(
        columns, hparams.categorical_columns, hparams.real_columns
    )
    if categorical_columns:
        raise ValueError(
            f"{usage} type 'pass_through' only supports real columns; "
            f"got categorical columns {categorical_columns}"
        )
    if not real_columns:
        raise ValueError(f"{usage} type 'pass_through' requires real columns")


def _validate_structured_config(
    hparams: Any,
    usage: str,
    columns: list[str],
    config: Any,
    layout: Optional[Any],
) -> None:
    _ = hparams, columns
    if layout is None:
        raise ValueError(f"{usage} requires a resolved feature layout")
    result_dim = config.output_dim
    unknown_axes = [
        axis for axis in config.axis_embeddings.axes if axis not in layout.axes
    ]
    if unknown_axes:
        raise ValueError(
            "Structured ingestion axis_embeddings references unavailable axes: "
            f"{unknown_axes}"
        )
    if (
        config.axis_embeddings.type == "rope"
        and (config.cell_dim or result_dim) % 2 != 0
    ):
        raise ValueError(
            "Structured ingestion axis_embeddings type 'rope' requires an even "
            "cell_dim/output_dim"
        )

    shape = AxisShape(tuple(layout.axes), config.cell_dim or result_dim)
    for block in config.processing_blocks:
        shape = STRUCTURED_BLOCK_HANDLERS[block.type].resolve(block, shape)

    if shape.channel_dim != result_dim:
        raise ValueError(
            "Structured ingestion processing_blocks must produce output_dim "
            f"{result_dim}, got {shape.channel_dim}"
        )


def _validate_temporal_conv_config(
    hparams: Any,
    usage: str,
    columns: list[str],
    config: Any,
    layout: Optional[Any],
) -> None:
    if config.base_ingestion == "direct_embed":
        _validate_direct_embed_config(hparams, usage, columns, config, layout)
    else:
        _validate_pass_through_config(hparams, usage, columns, config, layout)


def _validate_noop(
    hparams: Any,
    usage: str,
    columns: list[str],
    config: Any,
    layout: Optional[Any],
) -> None:
    _ = hparams, usage, columns, config, layout


def resolve_ingestion_plan(hparams: Any) -> IngestionPlan:
    """Lower the public union/dictionary syntax into one validated plan."""
    model_spec = hparams.model_spec
    ingestion_spec = model_spec.ingestion_spec
    if ingestion_spec is None:
        raise ValueError("model_spec.ingestion_spec must be configured")

    is_composite = isinstance(ingestion_spec, dict)
    if is_composite:
        if not ingestion_spec:
            raise ValueError(
                "model_spec.ingestion_spec must define at least one named ingestion"
            )
        branch_items = list(ingestion_spec.items())
        ingestion_merge = model_spec.ingestion_merge
        if ingestion_merge is None:
            raise ValueError(
                "model_spec.ingestion_merge must be configured for multiple ingestions"
            )
        merge_type: Optional[str] = ingestion_merge.type
    else:
        branch_items = [(None, ingestion_spec)]
        merge_type = None

    branches = []
    used_columns: dict[str, str] = {}
    consumed_columns: set[str] = set()
    for branch_name, config in branch_items:
        usage = (
            "model_spec.ingestion_spec"
            if branch_name is None
            else f"Composite ingestion branch {branch_name!r}"
        )
        handler = INGESTION_HANDLERS.get(config.type)
        if handler is None:
            raise ValueError(f"Unknown ingestion type: {config.type}")
        columns, layout = handler.resolve_columns(
            hparams,
            config,
            not is_composite,
        )
        if not columns:
            raise ValueError(f"{usage} must resolve to at least one input column")
        _validate_ingestion_columns(hparams, usage, columns)
        handler.validate(hparams, usage, columns, config, layout)
        categorical_columns, real_columns = _split_columns(
            columns, hparams.categorical_columns, hparams.real_columns
        )

        if is_composite and not model_spec.allow_shared_ingestion_columns:
            overlapping_columns = [col for col in columns if col in used_columns]
            if overlapping_columns:
                raise ValueError(
                    "Ingestion branches cannot share columns unless "
                    "allow_shared_ingestion_columns is true: "
                    f"{sorted(overlapping_columns)}"
                )
            for column in columns:
                used_columns[column] = str(branch_name)

        consumed_columns.update(columns)
        branches.append(
            ResolvedIngestionBranch(
                name=branch_name,
                config=config,
                columns=tuple(columns),
                categorical_columns=tuple(categorical_columns),
                real_columns=tuple(real_columns),
                width=config.output_dim,
                usage=usage,
                layout=layout,
            )
        )

    auxiliary_columns = set(model_spec.auxiliary_input_columns)
    unused_columns = set(hparams.input_columns) - consumed_columns
    unexpected_unused_columns = unused_columns - auxiliary_columns
    if model_spec.allow_unused_input_columns:
        if unused_columns:
            logger.warning(
                "model_spec.ingestion_spec does not consume every input column; "
                f"unused columns: {sorted(unused_columns)}"
            )
    elif unexpected_unused_columns:
        raise ValueError(
            "model_spec.ingestion_spec must consume every input column; unused "
            f"columns: {sorted(unexpected_unused_columns)}"
        )

    transformer_input_width = model_spec.dim_model - int(
        model_spec.positional_encoding == "range_concat"
    )
    return IngestionPlan(
        branches=tuple(branches),
        merge_type=merge_type,
        is_composite=is_composite,
        transformer_input_width=transformer_input_width,
    )


def _common_branch_kwargs(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> dict[str, Any]:
    return {
        "categorical_columns": list(branch.categorical_columns),
        "real_columns": list(branch.real_columns),
        "n_classes": context.hparams.n_classes,
        "context_length": context.hparams.window_view.context_length,
        "add_ingestion_position": context.add_ingestion_position,
        "dropout": context.hparams.training_spec.dropout,
    }


def _build_direct_embed_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    feature_embedding_dims = _feature_dims_for_columns(
        branch.config, list(branch.columns)
    )
    return DirectEmbedFeatureIngestion(
        categorical_columns=list(branch.categorical_columns),
        real_columns=list(branch.real_columns),
        n_classes=context.hparams.n_classes,
        context_length=context.hparams.window_view.context_length,
        embedding_size=None if feature_embedding_dims is not None else branch.width,
        feature_embedding_dims=feature_embedding_dims,
        add_ingestion_position=context.add_ingestion_position,
        dropout=context.hparams.training_spec.dropout,
        embedding_dim=branch.width,
        device_max_concat_length=context.device_max_concat_length,
    )


def _build_pass_through_module(
    branch: ResolvedIngestionBranch,
    context: IngestionBuildContext,
    *,
    projection_dim: int,
) -> PassThroughFeatureIngestion:
    return PassThroughFeatureIngestion(
        real_columns=list(branch.real_columns),
        context_length=context.hparams.window_view.context_length,
        add_ingestion_position=context.add_ingestion_position,
        dropout=context.hparams.training_spec.dropout,
        projection_dim=projection_dim,
        direct_real_dtype_provider=context.direct_real_dtype_provider,
        device_max_concat_length=context.device_max_concat_length,
    )


def _build_pass_through_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    return _build_pass_through_module(branch, context, projection_dim=branch.width)


def _build_temporal_conv_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    if branch.config.base_ingestion == "direct_embed":
        base_ingestion = _build_direct_embed_handler(branch, context)
        base_width = branch.width
    else:
        base_width = len(branch.real_columns)
        base_ingestion = _build_pass_through_module(
            branch, context, projection_dim=base_width
        )
    return TemporalConvFeatureIngestion(
        base_ingestion=base_ingestion,
        base_ingestion_width=base_width,
        channels=branch.width,
        kernel_size=branch.config.kernel_size,
        dilation_schedule=branch.config.dilation_schedule,
        causal=branch.config.causal,
        activation_fn=branch.config.activation_fn,
        dropout=branch.config.dropout,
        post_conv_norm=branch.config.post_conv_norm,
        orientation=branch.config.orientation,
        context_length=context.hparams.window_view.context_length,
    )


def _build_feature_pool_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    return FeaturePoolFeatureIngestion(
        columns=list(branch.columns),
        token_dim=branch.width,
        **_common_branch_kwargs(branch, context),
    )


def _build_grouped_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    return GroupedFeatureIngestion(
        groups=branch.config.groups,
        token_dim=branch.width,
        **_common_branch_kwargs(branch, context),
    )


def _build_siamese_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    return SiameseFeatureIngestion(
        columns=list(branch.columns),
        token_dim=branch.width,
        **_common_branch_kwargs(branch, context),
    )


def _build_structured_handler(
    branch: ResolvedIngestionBranch, context: IngestionBuildContext
) -> BaseFeatureIngestion:
    return StructuredFeatureIngestion(
        layout=branch.layout,
        result_dim=branch.width,
        cell_dim=branch.config.cell_dim,
        axis_embeddings=branch.config.axis_embeddings,
        processing_blocks=branch.config.processing_blocks,
        **_common_branch_kwargs(branch, context),
    )


INGESTION_HANDLERS: dict[str, IngestionHandler] = {
    "direct_embed": IngestionHandler(
        _flat_columns_for_config,
        _validate_direct_embed_config,
        _build_direct_embed_handler,
    ),
    "pass_through": IngestionHandler(
        _flat_columns_for_config,
        _validate_pass_through_config,
        _build_pass_through_handler,
    ),
    "temporal_conv": IngestionHandler(
        _flat_columns_for_config,
        _validate_temporal_conv_config,
        _build_temporal_conv_handler,
    ),
    "feature_pool": IngestionHandler(
        _flat_columns_for_config,
        _validate_noop,
        _build_feature_pool_handler,
    ),
    "grouped": IngestionHandler(
        _grouped_columns_for_config,
        _validate_noop,
        _build_grouped_handler,
    ),
    "siamese": IngestionHandler(
        _flat_columns_for_config,
        _validate_noop,
        _build_siamese_handler,
    ),
    "structured": IngestionHandler(
        _structured_columns_for_config,
        _validate_structured_config,
        _build_structured_handler,
    ),
}


def compile_feature_ingestion(
    *,
    hparams: Any,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> BuiltIngestion:
    plan = resolve_ingestion_plan(hparams)
    context = IngestionBuildContext(
        hparams=hparams,
        direct_real_dtype_provider=direct_real_dtype_provider,
        device_max_concat_length=device_max_concat_length,
        add_ingestion_position=_add_ingestion_position_encoding(hparams.model_spec),
    )
    built_branches = {
        branch.name: INGESTION_HANDLERS[branch.config.type].build(branch, context)
        for branch in plan.branches
    }

    if not plan.is_composite:
        branch = plan.branches[0]
        return BuiltIngestion(built_branches[None], branch.width)

    named_branches = {str(name): module for name, module in built_branches.items()}
    branch_widths = {str(branch.name): branch.width for branch in plan.branches}
    return BuiltIngestion(
        CompositeFeatureIngestion(
            branches=named_branches,
            branch_widths=branch_widths,
            merge_type=cast(str, plan.merge_type),
            merge_dim=plan.transformer_input_width,
        ),
        plan.transformer_input_width,
    )


def build_feature_ingestion(
    *,
    hparams: Any,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> BaseFeatureIngestion:
    """Compatibility wrapper returning only the compiled runtime module."""
    return compile_feature_ingestion(
        hparams=hparams,
        direct_real_dtype_provider=direct_real_dtype_provider,
        device_max_concat_length=device_max_concat_length,
    ).module
