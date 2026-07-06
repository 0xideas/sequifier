import math
from collections.abc import Callable
from itertools import product
from typing import Any, Optional

import torch
import torch.nn.functional as F
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
                f"embedding_size ({embedding_size}) is smaller than the "
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
                f"embedding_size ({embedding_size})."
            )
    elif len(real_columns) == 0 and len(categorical_columns) > 0:
        if embedding_size < len(categorical_columns):
            raise ValueError(
                f"embedding_size ({embedding_size}) is smaller than the "
                f"number of categorical columns ({len(categorical_columns)}). "
                "Resulting embedding dimension would be 0."
            )

        if (embedding_size % len(categorical_columns)) != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by "
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


class BaseFeatureIngestion(nn.Module):
    output_dim: int
    INIT_STD = 0.02

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def initialize_weights(self) -> None:
        return None


class DirectEmbedFeatureIngestion(BaseFeatureIngestion):
    """The original sequifier per-column embedding path."""

    def __init__(
        self,
        *,
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        embedding_size: Optional[int],
        feature_embedding_dims: Optional[dict[str, int]],
        use_rope: bool,
        dropout: float,
        output_dim: Optional[int] = None,
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

        if feature_embedding_dims is not None:
            self.feature_embedding_dims = feature_embedding_dims
        else:
            if embedding_size is None:
                raise ValueError(
                    "direct_embed ingestion requires output_dim when "
                    "feature_embedding_dims is not configured"
                )
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
            self.encoder[col] = nn.Linear(1, self.feature_embedding_dims[col])
            self.real_columns_with_embedding.append(col)

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
            self.output_projection_layer = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.output_projection_layer = None

    def initialize_weights(self) -> None:
        for col in self.categorical_columns:
            self.encoder[col].weight.data.normal_(mean=0.0, std=self.INIT_STD)

        if self.pos_encoder is not None:
            for col_name in self.pos_encoder:
                self.pos_encoder[col_name].weight.data.normal_(
                    mean=0.0, std=self.INIT_STD
                )

        if self.output_projection_layer is not None:
            self.output_projection_layer.weight.data.normal_(
                mean=0.0, std=self.INIT_STD
            )
            if self.output_projection_layer.bias is not None:
                self.output_projection_layer.bias.data.zero_()

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
            layer = self.encoder[col]
            inp = src[col][:, :, None].to(dtype=layer.weight.dtype)
            src_t = layer(inp) * scale

            srcs.append(self._with_position(col, src_t))

        output = self._recursive_concat(srcs)
        if self.output_projection_layer is not None:
            output = self.output_projection_layer(output)
        return output


class PassThroughFeatureIngestion(BaseFeatureIngestion):
    """Pass real-valued columns through without per-feature encoders."""

    def __init__(
        self,
        *,
        real_columns: list[str],
        context_length: int,
        use_rope: bool,
        dropout: float,
        output_dim: Optional[int] = None,
        direct_real_dtype_provider: Optional[Callable[[], torch.dtype]] = None,
        device_max_concat_length: int = 12,
    ):
        super().__init__()
        if not real_columns:
            raise ValueError("pass_through ingestion requires at least one real column")

        self.real_columns = real_columns
        self.real_columns_direct = list(real_columns)
        self.context_length = context_length
        self.use_rope = use_rope
        self.drop = nn.Dropout(dropout)
        self.input_dim = len(real_columns)
        self.embedding_size = self.input_dim
        self.output_dim = output_dim or self.input_dim
        self.direct_real_dtype_provider = direct_real_dtype_provider
        self.device_max_concat_length = device_max_concat_length

        if not self.use_rope:
            self.pos_encoder = ModuleDict()
            for col in self.real_columns:
                self.pos_encoder[col] = nn.Embedding(self.context_length, 1)
        else:
            self.pos_encoder = None

        if self.output_dim != self.input_dim:
            self.output_projection_layer = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.output_projection_layer = None

    def initialize_weights(self) -> None:
        if self.pos_encoder is not None:
            for col_name in self.pos_encoder:
                self.pos_encoder[col_name].weight.data.normal_(
                    mean=0.0, std=self.INIT_STD
                )

        if self.output_projection_layer is not None:
            self.output_projection_layer.weight.data.normal_(
                mean=0.0, std=self.INIT_STD
            )
            if self.output_projection_layer.bias is not None:
                self.output_projection_layer.bias.data.zero_()

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

    def _target_dtype(self, src: dict[str, Tensor]) -> torch.dtype:
        if self.output_projection_layer is not None:
            return self.output_projection_layer.weight.dtype
        if self.direct_real_dtype_provider is not None:
            return self.direct_real_dtype_provider()
        return src[self.real_columns[0]].dtype

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
        target_dtype = self._target_dtype(src)
        for col in self.real_columns:
            src_t = src[col].unsqueeze(2).to(dtype=target_dtype) * scale
            srcs.append(self._with_position(col, src_t))

        output = self._recursive_concat(srcs)
        if self.output_projection_layer is not None:
            output = self.output_projection_layer(
                output.to(dtype=self.output_projection_layer.weight.dtype)
            )
        return output


class TemporalConvFeatureIngestion(BaseFeatureIngestion):
    """Apply Conv1D over time after flat-column encoding."""

    def __init__(
        self,
        *,
        base_ingestion: DirectEmbedFeatureIngestion,
        output_dim: int,
        kernel_size: int,
        dilation: int,
        num_layers: int,
        causal: bool,
        activation_fn: str,
        dropout: float,
    ):
        super().__init__()
        self.base_ingestion = base_ingestion
        self.output_dim = output_dim
        if self.base_ingestion.output_dim != self.output_dim:
            raise ValueError(
                "temporal_conv base ingestion output_dim must match output_dim"
            )
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    self.output_dim,
                    self.output_dim,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                )
                for _ in range(num_layers)
            ]
        )
        self.activation = self._activation(activation_fn)
        self.drop = nn.Dropout(dropout)

    def initialize_weights(self) -> None:
        self.base_ingestion.initialize_weights()

    @staticmethod
    def _activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unknown temporal_conv activation_fn: {name}")

    def _temporal_padding(self) -> tuple[int, int]:
        padding = (self.kernel_size - 1) * self.dilation
        if self.causal:
            return padding, 0
        return padding // 2, padding // 2

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        output = self.base_ingestion(src, metadata)
        for layer in self.layers:
            conv_input = output.transpose(1, 2)
            conv_input = F.pad(conv_input, self._temporal_padding())
            output = layer(conv_input).transpose(1, 2)
            output = self.drop(self.activation(output))
        return output


class _ColumnTokenIngestion(BaseFeatureIngestion):
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


class FeaturePoolFeatureIngestion(_ColumnTokenIngestion):
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


class GroupedFeatureIngestion(BaseFeatureIngestion):
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
        self.group_ingestions = ModuleDict()
        categorical_set = set(categorical_columns)
        real_set = set(real_columns)
        for group_name, group_columns in self.groups.items():
            self.group_ingestions[group_name] = FeaturePoolFeatureIngestion(
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
        for ingestion in self.group_ingestions.values():
            ingestion.initialize_weights()

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        outputs = [
            ingestion(src, metadata) for ingestion in self.group_ingestions.values()
        ]
        return torch.stack(outputs, dim=0).mean(dim=0)


class SiameseFeatureIngestion(BaseFeatureIngestion):
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


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _module_key(indices: tuple[int, ...]) -> str:
    if not indices:
        return "shared"
    return "_".join(str(index) for index in indices)


def _rotate_half_last_dim(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _AxisProjectionBlock(nn.Module):
    """Project one or more cartesian axes into the channel dimension."""

    def __init__(
        self,
        *,
        axes: list[str],
        unshared_axes: list[str],
        output_dim: int,
        active_axes: list[str],
        axis_sizes: dict[str, int],
        input_dim: int,
    ):
        super().__init__()
        self.axes = axes
        self.unshared_axes = unshared_axes
        self.output_dim = output_dim
        self.active_axes = active_axes
        self.output_axes = [axis for axis in active_axes if axis not in axes]
        self.axis_sizes = axis_sizes
        self.input_dim = input_dim

        input_features = (
            _product([axis_sizes[axis] for axis in self.axes]) * self.input_dim
        )
        self.unshared_indices = list(
            product(*(range(axis_sizes[axis]) for axis in self.unshared_axes))
        ) or [()]
        self.layers = ModuleDict(
            {
                _module_key(indices): nn.Linear(input_features, self.output_dim)
                for indices in self.unshared_indices
            }
        )

    def _apply_shared(
        self,
        x: Tensor,
        active_axes: list[str],
        layer: nn.Linear,
    ) -> Tensor:
        keep_axes = [axis for axis in active_axes if axis not in self.axes]
        permute_dims = (
            [0, 1]
            + [2 + active_axes.index(axis) for axis in keep_axes]
            + [2 + active_axes.index(axis) for axis in self.axes]
            + [x.ndim - 1]
        )
        x = x.permute(*permute_dims)
        leading_shape = x.shape[: 2 + len(keep_axes)]
        x = x.reshape(-1, layer.in_features)
        x = layer(x)
        return x.reshape(*leading_shape, self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if not self.unshared_axes:
            return self._apply_shared(
                x,
                self.active_axes,
                self.layers["shared"],
            )

        output_shape = (
            list(x.shape[:2])
            + [self.axis_sizes[axis] for axis in self.output_axes]
            + [self.output_dim]
        )
        output = x.new_zeros(output_shape)
        remaining_axes = [
            axis for axis in self.active_axes if axis not in self.unshared_axes
        ]

        for indices in self.unshared_indices:
            index_by_axis = dict(zip(self.unshared_axes, indices))
            input_index = (
                [slice(None), slice(None)]
                + [
                    index_by_axis[axis] if axis in index_by_axis else slice(None)
                    for axis in self.active_axes
                ]
                + [slice(None)]
            )
            output_index = (
                [slice(None), slice(None)]
                + [
                    index_by_axis[axis] if axis in index_by_axis else slice(None)
                    for axis in self.output_axes
                ]
                + [slice(None)]
            )
            output[tuple(output_index)] = self._apply_shared(
                x[tuple(input_index)],
                remaining_axes,
                self.layers[_module_key(indices)],
            )

        return output


class _AxisConvBlock(nn.Module):
    """Apply a native convolution over one to three cartesian axes."""

    CONV_CLASSES = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    def __init__(
        self,
        *,
        axes: list[str],
        unshared_axes: list[str],
        output_dim: int,
        kernel_size: int,
        active_axes: list[str],
        axis_sizes: dict[str, int],
        input_dim: int,
    ):
        super().__init__()
        self.axes = axes
        self.unshared_axes = unshared_axes
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.active_axes = active_axes
        self.output_axes = list(active_axes)
        self.axis_sizes = axis_sizes
        self.input_dim = input_dim

        conv_class = self.CONV_CLASSES[len(self.axes)]
        self.unshared_indices = list(
            product(*(range(axis_sizes[axis]) for axis in self.unshared_axes))
        ) or [()]
        self.layers = ModuleDict(
            {
                _module_key(indices): conv_class(
                    self.input_dim,
                    self.output_dim,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
                for indices in self.unshared_indices
            }
        )

    def _apply_shared(
        self,
        x: Tensor,
        active_axes: list[str],
        layer: nn.Module,
    ) -> Tensor:
        other_axes = [axis for axis in active_axes if axis not in self.axes]
        sweep_axes = [axis for axis in active_axes if axis in self.axes]
        permute_dims = (
            [0, 1]
            + [2 + active_axes.index(axis) for axis in other_axes]
            + [x.ndim - 1]
            + [2 + active_axes.index(axis) for axis in sweep_axes]
        )
        x = x.permute(*permute_dims)
        leading_shape = x.shape[: 2 + len(other_axes)]
        sweep_shape = [self.axis_sizes[axis] for axis in sweep_axes]
        x = x.reshape(-1, self.input_dim, *sweep_shape)
        x = layer(x)
        x = x.reshape(*leading_shape, self.output_dim, *sweep_shape)

        axis_to_dim = {axis: 2 + index for index, axis in enumerate(other_axes)} | {
            axis: 2 + len(other_axes) + 1 + index
            for index, axis in enumerate(sweep_axes)
        }
        channel_dim = 2 + len(other_axes)
        permute_back = (
            [0, 1] + [axis_to_dim[axis] for axis in active_axes] + [channel_dim]
        )
        return x.permute(*permute_back)

    def forward(self, x: Tensor) -> Tensor:
        if not self.unshared_axes:
            return self._apply_shared(
                x,
                self.active_axes,
                self.layers["shared"],
            )

        output_shape = (
            list(x.shape[:2])
            + [self.axis_sizes[axis] for axis in self.output_axes]
            + [self.output_dim]
        )
        output = x.new_zeros(output_shape)
        remaining_axes = [
            axis for axis in self.active_axes if axis not in self.unshared_axes
        ]

        for indices in self.unshared_indices:
            index_by_axis = dict(zip(self.unshared_axes, indices))
            input_index = (
                [slice(None), slice(None)]
                + [
                    index_by_axis[axis] if axis in index_by_axis else slice(None)
                    for axis in self.active_axes
                ]
                + [slice(None)]
            )
            output_index = (
                [slice(None), slice(None)]
                + [
                    index_by_axis[axis] if axis in index_by_axis else slice(None)
                    for axis in self.output_axes
                ]
                + [slice(None)]
            )
            output[tuple(output_index)] = self._apply_shared(
                x[tuple(input_index)],
                remaining_axes,
                self.layers[_module_key(indices)],
            )

        return output


class _AxisAttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        n_head: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        output, _ = self.attention(x, x, x, need_weights=False)
        return output


class _AxisAttentionBlock(nn.Module):
    """Apply self-attention over one or more cartesian axes."""

    def __init__(
        self,
        *,
        axes: list[str],
        unshared_axes: list[str],
        output_dim: int,
        n_head: int,
        dropout: float,
        active_axes: list[str],
        axis_sizes: dict[str, int],
        input_dim: int,
    ):
        super().__init__()
        self.axes = axes
        self.unshared_axes = unshared_axes
        self.output_dim = output_dim
        self.n_head = n_head
        self.active_axes = active_axes
        self.output_axes = list(active_axes)
        self.axis_sizes = axis_sizes
        self.input_dim = input_dim

        self.unshared_indices = list(
            product(*(range(axis_sizes[axis]) for axis in self.unshared_axes))
        ) or [()]
        self.layers = ModuleDict(
            {
                _module_key(indices): _AxisAttentionLayer(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    n_head=self.n_head,
                    dropout=dropout,
                )
                for indices in self.unshared_indices
            }
        )

    def _apply_shared(
        self,
        x: Tensor,
        active_axes: list[str],
        layer: nn.Module,
    ) -> Tensor:
        other_axes = [axis for axis in active_axes if axis not in self.axes]
        attend_axes = [axis for axis in active_axes if axis in self.axes]
        permute_dims = (
            [0, 1]
            + [2 + active_axes.index(axis) for axis in other_axes]
            + [2 + active_axes.index(axis) for axis in attend_axes]
            + [x.ndim - 1]
        )
        x = x.permute(*permute_dims)
        leading_shape = x.shape[: 2 + len(other_axes)]
        attend_shape = [self.axis_sizes[axis] for axis in attend_axes]
        x = x.reshape(-1, _product(attend_shape), self.input_dim)
        x = layer(x)
        x = x.reshape(*leading_shape, *attend_shape, self.output_dim)

        axis_to_dim = {axis: 2 + index for index, axis in enumerate(other_axes)} | {
            axis: 2 + len(other_axes) + index for index, axis in enumerate(attend_axes)
        }
        channel_dim = x.ndim - 1
        permute_back = (
            [0, 1] + [axis_to_dim[axis] for axis in active_axes] + [channel_dim]
        )
        return x.permute(*permute_back)

    def forward(self, x: Tensor) -> Tensor:
        if not self.unshared_axes:
            return self._apply_shared(
                x,
                self.active_axes,
                self.layers["shared"],
            )

        output_shape = (
            list(x.shape[:2])
            + [self.axis_sizes[axis] for axis in self.output_axes]
            + [self.output_dim]
        )
        output = x.new_zeros(output_shape)
        remaining_axes = [
            axis for axis in self.active_axes if axis not in self.unshared_axes
        ]

        for indices in self.unshared_indices:
            index_by_axis = dict(zip(self.unshared_axes, indices))
            input_index = (
                [slice(None), slice(None)]
                + [
                    index_by_axis[axis] if axis in index_by_axis else slice(None)
                    for axis in self.active_axes
                ]
                + [slice(None)]
            )
            output_index = (
                [slice(None), slice(None)]
                + [
                    index_by_axis[axis] if axis in index_by_axis else slice(None)
                    for axis in self.output_axes
                ]
                + [slice(None)]
            )
            output[tuple(output_index)] = self._apply_shared(
                x[tuple(input_index)],
                remaining_axes,
                self.layers[_module_key(indices)],
            )

        return output


class _AxisPoolBlock(nn.Module):
    """Reduce one or more cartesian axes."""

    def __init__(
        self,
        *,
        axes: list[str],
        mode: str,
        active_axes: list[str],
    ):
        super().__init__()
        self.axes = axes
        self.mode = mode
        self.active_axes = active_axes
        self.output_axes = [axis for axis in active_axes if axis not in axes]

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(2 + self.active_axes.index(axis) for axis in self.axes)
        if self.mode == "mean":
            return x.mean(dim=dims)
        if self.mode == "sum":
            return x.sum(dim=dims)
        return torch.amax(x, dim=dims)


class StructuredFeatureIngestion(_ColumnTokenIngestion):
    """Compile a cartesian layout into an ordered cell tensor."""

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
        cell_dim: Optional[int] = None,
        axis_embeddings: Optional[Any] = None,
        processing_blocks: Optional[list[Any]] = None,
    ):
        self.layout = layout
        self.axis_order = layout.axis_order
        self.axis_size_by_name = {
            axis: len(layout.axes[axis]) for axis in self.axis_order
        }
        self.axis_sizes = [len(layout.axes[axis]) for axis in self.axis_order]
        self.expected_dense_shape = tuple(self.axis_sizes)
        self.cell_dim = cell_dim or output_dim
        self.axis_embeddings_config = axis_embeddings
        self.processing_blocks = processing_blocks or []
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
            output_dim=self.cell_dim,
            use_rope=use_rope,
            dropout=dropout,
        )
        self.output_dim = output_dim
        if not self.use_rope:
            self.pos_encoder = nn.Embedding(self.context_length, self.output_dim)

        self.axis_embedding_type = (
            "none"
            if self.axis_embeddings_config is None
            else self.axis_embeddings_config.type
        )
        self.axis_embedding_axes = (
            []
            if self.axis_embeddings_config is None
            else list(self.axis_embeddings_config.axes)
        )
        self.axis_embedding_theta = (
            10000.0
            if self.axis_embeddings_config is None
            else self.axis_embeddings_config.rope_theta
        )
        if self.axis_embedding_type == "rope" and self.cell_dim % 2 != 0:
            raise ValueError("Axis RoPE requires an even cell_dim")

        self.axis_embedding_layers = nn.ModuleList()
        if self.axis_embedding_type == "learned":
            for axis in self.axis_embedding_axes:
                self.axis_embedding_layers.append(
                    nn.Embedding(self.axis_size_by_name[axis], self.cell_dim)
                )

        self.axis_blocks = nn.ModuleList()
        active_axes = list(self.axis_order)
        channel_dim = self.cell_dim
        for block in self.processing_blocks:
            if block.type == "axis_projection":
                compiled_block = _AxisProjectionBlock(
                    axes=block.axes,
                    unshared_axes=block.unshared_axes,
                    output_dim=block.output_dim,
                    active_axes=list(active_axes),
                    axis_sizes=self.axis_size_by_name,
                    input_dim=channel_dim,
                )
                active_axes = compiled_block.output_axes
                channel_dim = block.output_dim
            elif block.type == "axis_conv":
                compiled_block = _AxisConvBlock(
                    axes=block.axes,
                    unshared_axes=block.unshared_axes,
                    output_dim=block.output_dim,
                    kernel_size=block.kernel_size,
                    active_axes=list(active_axes),
                    axis_sizes=self.axis_size_by_name,
                    input_dim=channel_dim,
                )
                channel_dim = block.output_dim
            elif block.type == "axis_attention":
                compiled_block = _AxisAttentionBlock(
                    axes=block.axes,
                    unshared_axes=block.unshared_axes,
                    output_dim=block.output_dim,
                    n_head=block.n_head,
                    dropout=block.dropout,
                    active_axes=list(active_axes),
                    axis_sizes=self.axis_size_by_name,
                    input_dim=channel_dim,
                )
                channel_dim = block.output_dim
            else:
                compiled_block = _AxisPoolBlock(
                    axes=block.axes,
                    mode=block.mode,
                    active_axes=list(active_axes),
                )
                active_axes = compiled_block.output_axes

            self.axis_blocks.append(compiled_block)

        self.active_axes_after_blocks = active_axes

    def initialize_weights(self) -> None:
        super().initialize_weights()
        for layer in self.axis_embedding_layers:
            layer.weight.data.normal_(mean=0.0, std=self.INIT_STD)

    def _dense_cells(self, src: dict[str, Tensor]) -> Tensor:
        encoded = [self._encode_column(col, src) for col in self.ordered_columns]
        cells = torch.stack(encoded, dim=2)
        return cells.reshape(
            cells.shape[0],
            cells.shape[1],
            *self.expected_dense_shape,
            self.cell_dim,
        )

    def _axis_broadcast_shape(self, x: Tensor, axis_name: str) -> list[int]:
        axis_idx = self.axis_order.index(axis_name)
        target_dim = axis_idx + 2
        axis_size = self.axis_size_by_name[axis_name]
        broadcast_shape = [1] * (x.ndim - 1) + [self.cell_dim]
        broadcast_shape[target_dim] = axis_size
        return broadcast_shape

    def _apply_learned_axis_embeddings(self, dense_cells: Tensor) -> Tensor:
        output = dense_cells
        for axis_name, embedding_layer in zip(
            self.axis_embedding_axes, self.axis_embedding_layers
        ):
            axis_size = self.axis_size_by_name[axis_name]
            indices = torch.arange(axis_size, device=output.device)
            embeddings = embedding_layer(indices).to(dtype=output.dtype)
            output = output + embeddings.view(
                *self._axis_broadcast_shape(output, axis_name)
            )
        return output

    def _axis_rope_cos_sin(self, x: Tensor, axis_name: str) -> tuple[Tensor, Tensor]:
        axis_size = self.axis_size_by_name[axis_name]
        compute_dtype = torch.float32
        positions = torch.arange(axis_size, device=x.device, dtype=compute_dtype)
        inv_freq = 1.0 / (
            self.axis_embedding_theta
            ** (
                torch.arange(0, self.cell_dim, 2, device=x.device, dtype=compute_dtype)
                / self.cell_dim
            )
        )
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        broadcast_shape = self._axis_broadcast_shape(x, axis_name)
        cos = emb.cos().to(dtype=x.dtype).view(*broadcast_shape)
        sin = emb.sin().to(dtype=x.dtype).view(*broadcast_shape)
        return cos, sin

    def _apply_rope_axis_embeddings(self, dense_cells: Tensor) -> Tensor:
        output = dense_cells
        for axis_name in self.axis_embedding_axes:
            cos, sin = self._axis_rope_cos_sin(output, axis_name)
            output = (output * cos) + (_rotate_half_last_dim(output) * sin)
        return output

    def _apply_axis_embeddings(self, dense_cells: Tensor) -> Tensor:
        if self.axis_embedding_type == "none":
            return dense_cells
        if self.axis_embedding_type == "learned":
            return self._apply_learned_axis_embeddings(dense_cells)
        if self.axis_embedding_type == "rope":
            return self._apply_rope_axis_embeddings(dense_cells)
        raise ValueError(f"Unknown axis embedding type: {self.axis_embedding_type}")

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        dense_cells = self._apply_axis_embeddings(self._dense_cells(src))
        if not self.axis_blocks:
            axis_dims = tuple(range(2, 2 + len(self.axis_sizes)))
            return self._with_position(dense_cells.mean(dim=axis_dims))

        output = dense_cells
        for block in self.axis_blocks:
            output = block(output)

        if self.active_axes_after_blocks:
            axis_dims = tuple(range(2, 2 + len(self.active_axes_after_blocks)))
            output = output.mean(dim=axis_dims)

        return self._with_position(output)


class IngestionMerge(nn.Module):
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


class CompositeFeatureIngestion(BaseFeatureIngestion):
    def __init__(
        self,
        *,
        branches: dict[str, BaseFeatureIngestion],
        merge_type: str,
        output_dim: int,
    ):
        super().__init__()
        self.branches = ModuleDict(branches)
        self.output_dim = output_dim
        self.merge = IngestionMerge(
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


def build_feature_ingestion(
    *,
    hparams: Any,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> BaseFeatureIngestion:
    model_spec = hparams.model_spec
    ingestion_layer_config = model_spec.ingestion_layer_config
    use_rope = model_spec.positional_encoding == "rope"

    if isinstance(ingestion_layer_config, dict):
        branches = {}
        branch_default_output_dim = (
            model_spec.dim_model if len(ingestion_layer_config) == 1 else None
        )
        for branch_name, branch_config in ingestion_layer_config.items():
            branches[branch_name] = _build_branch_ingestion(
                hparams=hparams,
                branch_config=branch_config,
                use_rope=use_rope,
                default_output_dim=branch_default_output_dim,
                direct_real_dtype_provider=direct_real_dtype_provider,
                device_max_concat_length=device_max_concat_length,
            )

        ingestion_merge = model_spec.ingestion_merge
        if ingestion_merge is None:
            raise ValueError("ingestion_merge must be configured for multiple streams")

        return CompositeFeatureIngestion(
            branches=branches,
            merge_type=ingestion_merge.type,
            output_dim=model_spec.dim_model,
        )

    if ingestion_layer_config.type == "direct_embed":
        return _build_direct_embed_ingestion(
            hparams=hparams,
            columns=ingestion_layer_config.columns or hparams.input_columns,
            ingestion_config=ingestion_layer_config,
            default_output_dim=model_spec.dim_model,
            device_max_concat_length=device_max_concat_length,
        )

    if ingestion_layer_config.type == "pass_through":
        return _build_pass_through_ingestion(
            hparams=hparams,
            columns=ingestion_layer_config.columns or hparams.input_columns,
            ingestion_config=ingestion_layer_config,
            default_output_dim=model_spec.dim_model,
            direct_real_dtype_provider=direct_real_dtype_provider,
            device_max_concat_length=device_max_concat_length,
        )

    return _build_branch_ingestion(
        hparams=hparams,
        branch_config=ingestion_layer_config,
        use_rope=use_rope,
        default_output_dim=model_spec.dim_model,
        direct_real_dtype_provider=direct_real_dtype_provider,
        device_max_concat_length=device_max_concat_length,
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
    ingestion_config: Any, columns: list[str]
) -> Optional[dict[str, int]]:
    feature_embedding_dims = ingestion_config.feature_embedding_dims
    if feature_embedding_dims is None:
        return None
    return {col: feature_embedding_dims[col] for col in columns}


def _resolve_required_output_dim(
    configured_output_dim: Optional[int],
    default_output_dim: Optional[int],
    *,
    usage: str,
) -> int:
    if configured_output_dim is not None:
        return configured_output_dim
    if default_output_dim is not None:
        return default_output_dim
    raise ValueError(f"{usage} must configure output_dim")


def _build_direct_embed_ingestion(
    *,
    hparams: Any,
    columns: list[str],
    ingestion_config: Any,
    default_output_dim: Optional[int],
    device_max_concat_length: int,
) -> DirectEmbedFeatureIngestion:
    categorical_columns, real_columns = _split_columns(
        columns, hparams.categorical_columns, hparams.real_columns
    )
    feature_embedding_dims = _feature_dims_for_columns(ingestion_config, columns)
    output_dim = ingestion_config.output_dim or default_output_dim
    embedding_size = None if feature_embedding_dims is not None else output_dim
    return DirectEmbedFeatureIngestion(
        categorical_columns=categorical_columns,
        real_columns=real_columns,
        n_classes=hparams.n_classes,
        context_length=hparams.window_view.context_length,
        embedding_size=embedding_size,
        feature_embedding_dims=feature_embedding_dims,
        use_rope=hparams.model_spec.positional_encoding == "rope",
        dropout=hparams.training_spec.dropout,
        output_dim=output_dim,
        device_max_concat_length=device_max_concat_length,
    )


def _build_pass_through_ingestion(
    *,
    hparams: Any,
    columns: list[str],
    ingestion_config: Any,
    default_output_dim: Optional[int],
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> PassThroughFeatureIngestion:
    categorical_columns, real_columns = _split_columns(
        columns, hparams.categorical_columns, hparams.real_columns
    )
    if categorical_columns:
        raise ValueError(
            "pass_through ingestion only supports real columns, "
            f"got categorical columns: {categorical_columns}"
        )
    return PassThroughFeatureIngestion(
        real_columns=real_columns,
        context_length=hparams.window_view.context_length,
        use_rope=hparams.model_spec.positional_encoding == "rope",
        dropout=hparams.training_spec.dropout,
        output_dim=ingestion_config.output_dim or default_output_dim,
        direct_real_dtype_provider=direct_real_dtype_provider,
        device_max_concat_length=device_max_concat_length,
    )


def _layout_columns(hparams: Any, layout_name: str) -> list[str]:
    return list(hparams.feature_layout.layouts[layout_name].columns)


def _build_branch_ingestion(
    *,
    hparams: Any,
    branch_config: Any,
    use_rope: bool,
    default_output_dim: Optional[int],
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
) -> BaseFeatureIngestion:
    if branch_config.type == "structured":
        columns = _layout_columns(hparams, branch_config.layout)
    elif branch_config.type == "grouped":
        columns = [
            column
            for group_columns in branch_config.groups.values()
            for column in group_columns
        ]
    else:
        columns = branch_config.columns

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

    if branch_config.type == "direct_embed":
        return _build_direct_embed_ingestion(
            hparams=hparams,
            columns=columns,
            ingestion_config=branch_config,
            default_output_dim=default_output_dim,
            device_max_concat_length=device_max_concat_length,
        )

    if branch_config.type == "pass_through":
        return _build_pass_through_ingestion(
            hparams=hparams,
            columns=columns,
            ingestion_config=branch_config,
            default_output_dim=default_output_dim,
            direct_real_dtype_provider=direct_real_dtype_provider,
            device_max_concat_length=device_max_concat_length,
        )

    if branch_config.type == "temporal_conv":
        output_dim = _resolve_required_output_dim(
            branch_config.output_dim,
            default_output_dim,
            usage="temporal_conv ingestion",
        )
        base_ingestion = _build_direct_embed_ingestion(
            hparams=hparams,
            columns=columns,
            ingestion_config=branch_config,
            default_output_dim=output_dim,
            device_max_concat_length=device_max_concat_length,
        )
        return TemporalConvFeatureIngestion(
            base_ingestion=base_ingestion,
            output_dim=output_dim,
            kernel_size=branch_config.kernel_size,
            dilation=branch_config.dilation,
            num_layers=branch_config.num_layers,
            causal=branch_config.causal,
            activation_fn=branch_config.activation_fn,
            dropout=branch_config.dropout,
        )

    if branch_config.type == "feature_pool":
        output_dim = _resolve_required_output_dim(
            branch_config.output_dim,
            default_output_dim,
            usage="feature_pool ingestion",
        )
        return FeaturePoolFeatureIngestion(
            columns=columns,
            output_dim=output_dim,
            **common_kwargs,
        )

    if branch_config.type == "grouped":
        output_dim = _resolve_required_output_dim(
            branch_config.output_dim,
            default_output_dim,
            usage="grouped ingestion",
        )
        return GroupedFeatureIngestion(
            groups=branch_config.groups,
            output_dim=output_dim,
            **common_kwargs,
        )

    if branch_config.type == "siamese":
        output_dim = _resolve_required_output_dim(
            branch_config.output_dim,
            default_output_dim,
            usage="siamese ingestion",
        )
        return SiameseFeatureIngestion(
            columns=columns,
            output_dim=output_dim,
            **common_kwargs,
        )

    if branch_config.type == "structured":
        layout = hparams.feature_layout.layouts[branch_config.layout]
        output_dim = _resolve_required_output_dim(
            branch_config.output_dim,
            default_output_dim,
            usage="structured ingestion",
        )
        return StructuredFeatureIngestion(
            layout=layout,
            output_dim=output_dim,
            cell_dim=branch_config.cell_dim,
            axis_embeddings=branch_config.axis_embeddings,
            processing_blocks=branch_config.processing_blocks,
            **common_kwargs,
        )

    raise ValueError(f"Unknown ingestion type: {branch_config.type}")
