import math
from collections.abc import Callable
from itertools import product
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import ModuleDict

EMBEDDING_INDEX_DTYPES = (torch.int32, torch.int64)
NARROW_EMBEDDING_INDEX_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
)
WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES = (torch.uint32, torch.uint64)


def _smallest_embedding_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in EMBEDDING_INDEX_DTYPES:
        return dtype
    if dtype in NARROW_EMBEDDING_INDEX_DTYPES:
        return torch.int32
    if dtype in WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES:
        return torch.int64
    raise TypeError(f"Embedding indices must use an integer dtype, got {dtype}.")


def embedding_safe_indices(indices: Tensor) -> Tensor:
    target_dtype = _smallest_embedding_safe_dtype(indices.dtype)
    if indices.dtype == target_dtype:
        return indices
    return indices.to(dtype=target_dtype)


def get_feature_embedding_dims(
    embedding_size: int,
    categorical_columns: list[str],
    real_columns: list[str],
) -> dict[str, int]:
    if not (len(categorical_columns) + len(real_columns)) > 0:
        raise ValueError("No columns found")

    if len(categorical_columns) == 0 and len(real_columns) > 0:
        if embedding_size < len(real_columns):
            raise ValueError(
                f"initial_embedding_dim ({embedding_size}) is smaller than the "
                f"number of real input columns ({len(real_columns)}). "
                "Cannot allocate at least 1 dimension per column."
            )

        feature_embedding_dims = {col: 1 for col in real_columns}
        column_index = dict(enumerate(real_columns))

        remaining_dims = embedding_size - len(real_columns)
        for i in range(remaining_dims):
            j = i % len(real_columns)
            feature_embedding_dims[column_index[j]] += 1

        if sum(feature_embedding_dims.values()) != embedding_size:
            raise ValueError(
                "Auto-calculated embedding dimensions "
                f"({sum(feature_embedding_dims.values())}) do not sum to "
                f"initial_embedding_dim ({embedding_size})."
            )
    elif len(real_columns) == 0 and len(categorical_columns) > 0:
        if embedding_size < len(categorical_columns):
            raise ValueError(
                f"initial_embedding_dim ({embedding_size}) is smaller than the "
                f"number of categorical columns ({len(categorical_columns)}). "
                "Resulting embedding dimension would be 0."
            )

        if (embedding_size % len(categorical_columns)) != 0:
            raise ValueError(
                f"initial_embedding_dim ({embedding_size}) must be divisible by "
                f"n_categorical ({len(categorical_columns)})"
            )
        dim_model_comp = embedding_size // len(categorical_columns)
        feature_embedding_dims = {col: dim_model_comp for col in categorical_columns}
    else:
        raise ValueError(
            "If both real and categorical variables are present, "
            "feature_embedding_dims config value must be set"
        )

    return feature_embedding_dims


class BaseFeatureFrontend(nn.Module):
    output_dim: int
    INIT_STD = 0.02

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def initialize_weights(self) -> None:
        return None


class FlatFeatureFrontend(BaseFeatureFrontend):
    """The original sequifier per-column embedding path."""

    def __init__(
        self,
        *,
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        embedding_size: int,
        feature_embedding_dims: Optional[dict[str, int]],
        use_rope: bool,
        dropout: float,
        output_dim: Optional[int] = None,
        direct_real_dtype_provider: Optional[Callable[[], torch.dtype]] = None,
        device_max_concat_length: int = 12,
    ):
        super().__init__()
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.n_classes = n_classes
        self.context_length = context_length
        self.use_rope = use_rope
        self.drop = nn.Dropout(dropout)
        self.device_max_concat_length = device_max_concat_length
        self.direct_real_dtype_provider = direct_real_dtype_provider

        if feature_embedding_dims is not None:
            self.feature_embedding_dims = feature_embedding_dims
        else:
            self.feature_embedding_dims = get_feature_embedding_dims(
                embedding_size, categorical_columns, real_columns
            )

        self.input_dim = sum(self.feature_embedding_dims.values())
        self.embedding_size = self.input_dim
        self.output_dim = output_dim or self.input_dim

        self.encoder = ModuleDict()
        self.real_columns_with_embedding = []
        self.real_columns_direct = []
        for col in self.real_columns:
            if self.feature_embedding_dims[col] > 1:
                self.encoder[col] = nn.Linear(1, self.feature_embedding_dims[col])
                self.real_columns_with_embedding.append(col)
            else:
                if self.feature_embedding_dims[col] != 1:
                    raise ValueError(
                        f"Real column {col} without embedding must have "
                        "feature_embedding_dims=1"
                    )
                self.real_columns_direct.append(col)

        for col in self.categorical_columns:
            self.encoder[col] = nn.Embedding(
                self.n_classes[col], self.feature_embedding_dims[col]
            )

        if not self.use_rope:
            self.pos_encoder = ModuleDict()
            for col in self.real_columns + self.categorical_columns:
                self.pos_encoder[col] = nn.Embedding(
                    self.context_length, self.feature_embedding_dims[col]
                )
        else:
            self.pos_encoder = None

        if self.output_dim != self.input_dim:
            self.joint_embedding_layer = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.joint_embedding_layer = None

    def initialize_weights(self) -> None:
        for col in self.categorical_columns:
            self.encoder[col].weight.data.normal_(mean=0.0, std=self.INIT_STD)

        if self.pos_encoder is not None:
            for col_name in self.pos_encoder:
                self.pos_encoder[col_name].weight.data.normal_(
                    mean=0.0, std=self.INIT_STD
                )

        if self.joint_embedding_layer is not None:
            self.joint_embedding_layer.weight.data.normal_(mean=0.0, std=self.INIT_STD)
            if self.joint_embedding_layer.bias is not None:
                self.joint_embedding_layer.bias.data.zero_()

    def _recursive_concat(self, srcs: list[Tensor]) -> Tensor:
        if len(srcs) <= self.device_max_concat_length:
            return torch.cat(srcs, 2)

        srcs_inner = []
        for start in range(0, len(srcs), self.device_max_concat_length):
            src = self._recursive_concat(
                srcs[start : start + self.device_max_concat_length]
            )
            srcs_inner.append(src)
        return self._recursive_concat(srcs_inner)

    def _position_encoding(self, col: str, batch_size: int, device: torch.device):
        pos = torch.arange(0, self.context_length, dtype=torch.long, device=device)
        pos = pos.repeat(batch_size, 1)
        return self.pos_encoder[col](pos)  # type: ignore[index]

    def _with_position(self, col: str, src_t: Tensor) -> Tensor:
        if self.use_rope:
            return self.drop(src_t)
        src_p = self._position_encoding(col, src_t.shape[0], src_t.device)
        return self.drop(src_t + src_p)

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        srcs = []
        scale = math.sqrt(self.embedding_size)
        for col in self.categorical_columns:
            src_t = self.encoder[col](embedding_safe_indices(src[col])) * scale
            srcs.append(self._with_position(col, src_t))

        for col in self.real_columns:
            if col in self.real_columns_direct:
                target_dtype = (
                    self.direct_real_dtype_provider()
                    if self.direct_real_dtype_provider is not None
                    else src[col].dtype
                )
                src_t = src[col].unsqueeze(2).to(dtype=target_dtype) * scale
            else:
                layer = self.encoder[col]
                inp = src[col][:, :, None].to(dtype=layer.weight.dtype)
                src_t = layer(inp) * scale

            srcs.append(self._with_position(col, src_t))

        output = self._recursive_concat(srcs)
        if self.joint_embedding_layer is not None:
            output = self.joint_embedding_layer(output)
        return output


class _ColumnTokenFrontend(BaseFeatureFrontend):
    def __init__(
        self,
        *,
        columns: list[str],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        output_dim: int,
        use_rope: bool,
        dropout: float,
    ):
        super().__init__()
        self.columns = columns
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.n_classes = n_classes
        self.context_length = context_length
        self.output_dim = output_dim
        self.use_rope = use_rope
        self.drop = nn.Dropout(dropout)

        self.encoder = ModuleDict()
        for col in self.categorical_columns:
            self.encoder[col] = nn.Embedding(self.n_classes[col], self.output_dim)
        for col in self.real_columns:
            self.encoder[col] = nn.Linear(1, self.output_dim)

        if not self.use_rope:
            self.pos_encoder = nn.Embedding(self.context_length, self.output_dim)
        else:
            self.pos_encoder = None

    def initialize_weights(self) -> None:
        for col in self.categorical_columns:
            self.encoder[col].weight.data.normal_(mean=0.0, std=self.INIT_STD)
        if self.pos_encoder is not None:
            self.pos_encoder.weight.data.normal_(mean=0.0, std=self.INIT_STD)

    def _encode_column(self, col: str, src: dict[str, Tensor]) -> Tensor:
        if col in self.categorical_columns:
            return self.encoder[col](embedding_safe_indices(src[col]))

        layer = self.encoder[col]
        return layer(src[col][:, :, None].to(dtype=layer.weight.dtype))

    def _with_position(self, x: Tensor) -> Tensor:
        if self.use_rope:
            return self.drop(x)
        pos = torch.arange(0, self.context_length, dtype=torch.long, device=x.device)
        pos = pos.repeat(x.shape[0], 1)
        return self.drop(x + self.pos_encoder(pos))  # type: ignore[operator]


class FeatureTokenFrontend(_ColumnTokenFrontend):
    """Encode each feature as a token and pool feature tokens per time step."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.feature_embedding = nn.Parameter(
            torch.zeros(len(self.columns), self.output_dim)
        )

    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.feature_embedding.data.normal_(mean=0.0, std=self.INIT_STD)

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        encoded = [self._encode_column(col, src) for col in self.columns]
        tokens = torch.stack(encoded, dim=2)
        feature_embedding = self.feature_embedding.to(dtype=tokens.dtype)
        tokens = tokens + feature_embedding[None, None, :, :]
        return self._with_position(tokens.mean(dim=2))


class GroupedFeatureFrontend(BaseFeatureFrontend):
    """Encode configured column groups and average the group representations."""

    def __init__(
        self,
        *,
        groups: dict[str, list[str]],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        output_dim: int,
        use_rope: bool,
        dropout: float,
    ):
        super().__init__()
        self.groups = groups
        self.output_dim = output_dim
        self.group_frontends = ModuleDict()
        categorical_set = set(categorical_columns)
        real_set = set(real_columns)
        for group_name, group_columns in self.groups.items():
            self.group_frontends[group_name] = FeatureTokenFrontend(
                columns=group_columns,
                categorical_columns=[
                    col for col in group_columns if col in categorical_set
                ],
                real_columns=[col for col in group_columns if col in real_set],
                n_classes=n_classes,
                context_length=context_length,
                output_dim=output_dim,
                use_rope=use_rope,
                dropout=dropout,
            )

    def initialize_weights(self) -> None:
        for frontend in self.group_frontends.values():
            frontend.initialize_weights()

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        outputs = [
            frontend(src, metadata) for frontend in self.group_frontends.values()
        ]
        return torch.stack(outputs, dim=0).mean(dim=0)


class SiameseFeatureFrontend(BaseFeatureFrontend):
    """Apply shared encoders across branch columns and pool their outputs."""

    def __init__(
        self,
        *,
        columns: list[str],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        output_dim: int,
        use_rope: bool,
        dropout: float,
    ):
        super().__init__()
        self.columns = columns
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.context_length = context_length
        self.output_dim = output_dim
        self.use_rope = use_rope
        self.drop = nn.Dropout(dropout)

        if categorical_columns:
            self.categorical_encoder = nn.Embedding(
                max(n_classes[col] for col in categorical_columns), output_dim
            )
        else:
            self.categorical_encoder = None
        if real_columns:
            self.real_encoder = nn.Linear(1, output_dim)
        else:
            self.real_encoder = None

        if not self.use_rope:
            self.pos_encoder = nn.Embedding(self.context_length, self.output_dim)
        else:
            self.pos_encoder = None

    def initialize_weights(self) -> None:
        if self.categorical_encoder is not None:
            self.categorical_encoder.weight.data.normal_(mean=0.0, std=self.INIT_STD)
        if self.pos_encoder is not None:
            self.pos_encoder.weight.data.normal_(mean=0.0, std=self.INIT_STD)

    def _with_position(self, x: Tensor) -> Tensor:
        if self.use_rope:
            return self.drop(x)
        pos = torch.arange(0, self.context_length, dtype=torch.long, device=x.device)
        pos = pos.repeat(x.shape[0], 1)
        return self.drop(x + self.pos_encoder(pos))  # type: ignore[operator]

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        encoded = []
        for col in self.columns:
            if col in self.categorical_columns:
                encoded.append(
                    self.categorical_encoder(embedding_safe_indices(src[col]))  # type: ignore
                )
            else:
                encoded.append(
                    self.real_encoder(
                        src[col][:, :, None].to(dtype=self.real_encoder.weight.dtype)  # type: ignore
                    )
                )
        return self._with_position(torch.stack(encoded, dim=2).mean(dim=2))


class StructuredFeatureFrontend(_ColumnTokenFrontend):
    """Compile a dense_axes layout into an ordered cell tensor."""

    def __init__(
        self,
        *,
        layout: Any,
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        output_dim: int,
        use_rope: bool,
        dropout: float,
    ):
        self.layout = layout
        self.axis_order = layout.axis_order
        self.axis_sizes = [len(layout.axes[axis]) for axis in self.axis_order]
        self.expected_dense_shape = tuple(self.axis_sizes)
        self.coordinate_to_index = {
            tuple(coordinates): index
            for index, coordinates in enumerate(
                product(*(layout.axes[axis] for axis in self.axis_order))
            )
        }
        coordinate_to_column = {
            tuple(coordinates[axis] for axis in self.axis_order): column
            for column, coordinates in layout.columns.items()
        }
        self.ordered_columns = [
            coordinate_to_column[coordinates]
            for coordinates in product(*(layout.axes[axis] for axis in self.axis_order))
        ]
        super().__init__(
            columns=self.ordered_columns,
            categorical_columns=categorical_columns,
            real_columns=real_columns,
            n_classes=n_classes,
            context_length=context_length,
            output_dim=output_dim,
            use_rope=use_rope,
            dropout=dropout,
        )

    def _dense_cells(self, src: dict[str, Tensor]) -> Tensor:
        encoded = [self._encode_column(col, src) for col in self.ordered_columns]
        cells = torch.stack(encoded, dim=2)
        return cells.reshape(
            cells.shape[0],
            cells.shape[1],
            *self.expected_dense_shape,
            self.output_dim,
        )

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        dense_cells = self._dense_cells(src)
        axis_dims = tuple(range(2, 2 + len(self.axis_sizes)))
        return self._with_position(dense_cells.mean(dim=axis_dims))


class ConvFeatureFrontend(StructuredFeatureFrontend):
    """Apply a lightweight convolution across dense layout cells."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.cell_conv = nn.Conv1d(
            self.output_dim, self.output_dim, kernel_size=3, padding=1
        )

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        encoded = [self._encode_column(col, src) for col in self.ordered_columns]
        cells = torch.stack(encoded, dim=2)
        batch_size, context_length, cell_count, _ = cells.shape
        conv_input = cells.reshape(
            batch_size * context_length, cell_count, self.output_dim
        )
        conv_input = conv_input.transpose(1, 2)
        conv_output = self.cell_conv(conv_input).transpose(1, 2)
        output = conv_output.mean(dim=1).reshape(
            batch_size, context_length, self.output_dim
        )
        return self._with_position(output)


class PatchFeatureFrontend(StructuredFeatureFrontend):
    """Apply a same-length temporal patch projection to structured features."""

    def __init__(self, *, patch_size: int, stride: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.patch_projection = nn.Conv1d(
            self.output_dim,
            self.output_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.patch_size // 2,
        )

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        output = super().forward(src, metadata)
        patch_input = output.transpose(1, 2)
        patch_output = self.patch_projection(patch_input).transpose(1, 2)
        return patch_output[:, : output.shape[1], :]


class FrontendMerge(nn.Module):
    def __init__(self, merge_type: str, branch_dims: dict[str, int], output_dim: int):
        super().__init__()
        self.merge_type = merge_type
        self.branch_names = list(branch_dims)
        self.branch_dims = branch_dims
        self.output_dim = output_dim

        if self.merge_type == "concat":
            input_dim = sum(self.branch_dims.values())
            self.concat_projection = (
                nn.Linear(input_dim, self.output_dim)
                if input_dim != self.output_dim
                else nn.Identity()
            )
        else:
            self.branch_projections = ModuleDict(
                {
                    name: (
                        nn.Linear(branch_dim, self.output_dim)
                        if branch_dim != self.output_dim
                        else nn.Identity()
                    )
                    for name, branch_dim in self.branch_dims.items()
                }
            )
            if self.merge_type == "gated":
                self.gate = nn.Linear(
                    len(self.branch_names) * self.output_dim, len(self.branch_names)
                )
            elif self.merge_type == "attention":
                self.query = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, branch_outputs: dict[str, Tensor]) -> Tensor:
        if self.merge_type == "concat":
            merged = torch.cat(
                [branch_outputs[name] for name in self.branch_names],
                dim=-1,
            )
            return self.concat_projection(merged)

        projected = [
            self.branch_projections[name](branch_outputs[name])
            for name in self.branch_names
        ]
        stacked = torch.stack(projected, dim=2)
        if self.merge_type == "sum":
            return stacked.sum(dim=2)

        if self.merge_type == "gated":
            gate_input = torch.cat(projected, dim=-1)
            weights = torch.softmax(self.gate(gate_input), dim=-1)
            return (stacked * weights[:, :, :, None]).sum(dim=2)

        query = self.query.to(dtype=stacked.dtype)
        scores = (stacked * query[None, None, None, :]).sum(dim=-1)
        scores = scores / math.sqrt(self.output_dim)
        weights = torch.softmax(scores, dim=-1)
        return (stacked * weights[:, :, :, None]).sum(dim=2)


class CompositeFeatureFrontend(BaseFeatureFrontend):
    def __init__(
        self,
        *,
        branches: dict[str, BaseFeatureFrontend],
        merge_type: str,
        output_dim: int,
    ):
        super().__init__()
        self.branches = ModuleDict(branches)
        self.output_dim = output_dim
        self.merge = FrontendMerge(
            merge_type,
            {name: branch.output_dim for name, branch in branches.items()},
            output_dim,
        )

    def initialize_weights(self) -> None:
        for branch in self.branches.values():
            branch.initialize_weights()

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        branch_outputs = {
            name: branch(src, metadata) for name, branch in self.branches.items()
        }
        return self.merge(branch_outputs)


def build_feature_frontend(
    *,
    hparams: Any,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> BaseFeatureFrontend:
    model_spec = hparams.model_spec
    frontend_spec = model_spec.frontend
    use_rope = model_spec.positional_encoding == "rope"

    if frontend_spec.type == "flat":
        return _build_flat_frontend(
            hparams=hparams,
            columns=hparams.input_columns,
            frontend_spec=frontend_spec,
            use_top_level_joint=True,
            direct_real_dtype_provider=direct_real_dtype_provider,
            device_max_concat_length=device_max_concat_length,
        )

    branches = {}
    for branch_name, branch_spec in frontend_spec.branches.items():
        branches[branch_name] = _build_branch_frontend(
            hparams=hparams,
            branch_spec=branch_spec,
            use_rope=use_rope,
            direct_real_dtype_provider=direct_real_dtype_provider,
            device_max_concat_length=device_max_concat_length,
        )

    return CompositeFeatureFrontend(
        branches=branches,
        merge_type=frontend_spec.merge.type,
        output_dim=frontend_spec.merge.output_dim,
    )


def _split_columns(
    columns: list[str], categorical_columns: list[str], real_columns: list[str]
) -> tuple[list[str], list[str]]:
    categorical_set = set(categorical_columns)
    real_set = set(real_columns)
    return (
        [col for col in columns if col in categorical_set],
        [col for col in columns if col in real_set],
    )


def _feature_dims_for_columns(
    hparams: Any, columns: list[str]
) -> Optional[dict[str, int]]:
    feature_embedding_dims = hparams.model_spec.feature_embedding_dims
    if feature_embedding_dims is None:
        return None
    return {col: feature_embedding_dims[col] for col in columns}


def _build_flat_frontend(
    *,
    hparams: Any,
    columns: list[str],
    frontend_spec: Any,
    use_top_level_joint: bool,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> FlatFeatureFrontend:
    categorical_columns, real_columns = _split_columns(
        columns, hparams.categorical_columns, hparams.real_columns
    )
    feature_embedding_dims = _feature_dims_for_columns(hparams, columns)
    output_dim = frontend_spec.output_dim
    if use_top_level_joint and output_dim is None:
        output_dim = hparams.model_spec.joint_embedding_dim

    embedding_size = (
        hparams.model_spec.initial_embedding_dim
        if use_top_level_joint
        else frontend_spec.output_dim or hparams.model_spec.initial_embedding_dim
    )
    return FlatFeatureFrontend(
        categorical_columns=categorical_columns,
        real_columns=real_columns,
        n_classes=hparams.n_classes,
        context_length=hparams.window_view.context_length,
        embedding_size=embedding_size,
        feature_embedding_dims=feature_embedding_dims,
        use_rope=hparams.model_spec.positional_encoding == "rope",
        dropout=hparams.training_spec.dropout,
        output_dim=output_dim,
        direct_real_dtype_provider=direct_real_dtype_provider,
        device_max_concat_length=device_max_concat_length,
    )


def _layout_columns(hparams: Any, layout_name: str) -> list[str]:
    return list(hparams.feature_layout.layouts[layout_name].columns)


def _build_branch_frontend(
    *,
    hparams: Any,
    branch_spec: Any,
    use_rope: bool,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> BaseFeatureFrontend:
    frontend = branch_spec.frontend
    if frontend.type in {"structured", "conv", "patch"}:
        columns = _layout_columns(hparams, frontend.layout)
    elif frontend.type == "grouped":
        columns = [
            column
            for group_columns in frontend.groups.values()
            for column in group_columns
        ]
    else:
        columns = branch_spec.columns

    categorical_columns, real_columns = _split_columns(
        columns, hparams.categorical_columns, hparams.real_columns
    )

    common_kwargs = {
        "categorical_columns": categorical_columns,
        "real_columns": real_columns,
        "n_classes": hparams.n_classes,
        "context_length": hparams.window_view.context_length,
        "use_rope": use_rope,
        "dropout": hparams.training_spec.dropout,
    }

    if frontend.type == "flat":
        return _build_flat_frontend(
            hparams=hparams,
            columns=columns,
            frontend_spec=frontend,
            use_top_level_joint=False,
            direct_real_dtype_provider=direct_real_dtype_provider,
            device_max_concat_length=device_max_concat_length,
        )

    if frontend.type == "feature_token":
        return FeatureTokenFrontend(
            columns=columns,
            output_dim=frontend.output_dim,
            **common_kwargs,
        )

    if frontend.type == "grouped":
        return GroupedFeatureFrontend(
            groups=frontend.groups,
            output_dim=frontend.output_dim,
            **common_kwargs,
        )

    if frontend.type == "siamese":
        return SiameseFeatureFrontend(
            columns=columns,
            output_dim=frontend.output_dim,
            **common_kwargs,
        )

    layout = hparams.feature_layout.layouts[frontend.layout]
    if frontend.type == "structured":
        return StructuredFeatureFrontend(
            layout=layout,
            output_dim=frontend.output_dim,
            **common_kwargs,
        )

    if frontend.type == "conv":
        return ConvFeatureFrontend(
            layout=layout,
            output_dim=frontend.output_dim,
            **common_kwargs,
        )

    return PatchFeatureFrontend(
        layout=layout,
        output_dim=frontend.output_dim,
        patch_size=frontend.patch_size,
        stride=frontend.stride,
        **common_kwargs,
    )
