import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import ModuleDict

from sequifier.model.dtypes import (
    cast_floating_to_dtype,
    cast_floating_to_module_dtype,
    module_param_dtype,
)
from sequifier.model.layers import RMSNorm
from sequifier.typechecking import beartype, conditional_beartype

EMBEDDING_INDEX_DTYPES = (torch.int32, torch.int64)
NARROW_EMBEDDING_INDEX_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
)
WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES = (torch.uint32, torch.uint64)


@conditional_beartype
def _smallest_embedding_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in EMBEDDING_INDEX_DTYPES:
        return dtype
    if dtype in NARROW_EMBEDDING_INDEX_DTYPES:
        return torch.int32
    if dtype in WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES:
        return torch.int64
    raise TypeError(f"Embedding indices must use an integer dtype, got {dtype}.")


@conditional_beartype
def embedding_safe_indices(indices: Tensor) -> Tensor:
    target_dtype = _smallest_embedding_safe_dtype(indices.dtype)
    if indices.dtype == target_dtype:
        return indices
    return indices.to(dtype=target_dtype)


@beartype
def _validate_module_dict_key(key: str, usage: str) -> None:
    if key == "":
        raise ValueError(f"{usage} cannot be empty")
    if "." in key:
        raise ValueError(f"{usage} cannot contain '.'")


@beartype
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
    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @staticmethod
    @conditional_beartype
    def _scale_embedding(x: Tensor, embedding_dim: int) -> Tensor:
        return x * math.sqrt(embedding_dim)


class RealFeatureProjection(nn.Linear):
    """Project one standardized real-valued feature into an embedding space."""

    @beartype
    def __init__(self, embedding_dim: int):
        super().__init__(1, embedding_dim)


class DirectEmbedFeatureIngestion(BaseFeatureIngestion):
    """The original sequifier per-column embedding path."""

    @beartype
    def __init__(
        self,
        *,
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        embedding_size: Optional[int],
        feature_embedding_dims: Optional[dict[str, int]],
        add_ingestion_position: bool,
        dropout: float,
        embedding_dim: Optional[int] = None,
        device_max_concat_length: int = 12,
    ):
        super().__init__()
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.n_classes = n_classes
        self.context_length = context_length
        self.add_ingestion_position = add_ingestion_position
        self.drop = nn.Dropout(dropout)
        self.device_max_concat_length = device_max_concat_length

        if feature_embedding_dims is not None:
            self.feature_embedding_dims = feature_embedding_dims
        else:
            if embedding_size is None:
                raise ValueError(
                    "direct_embed ingestion requires an embedding dimension when "
                    "feature_embedding_dims is not configured"
                )
            self.feature_embedding_dims = get_feature_embedding_dims(
                embedding_size, categorical_columns, real_columns
            )

        self.input_dim = sum(self.feature_embedding_dims.values())
        self.embedding_size = self.input_dim
        self.embedding_dim = embedding_dim or self.input_dim

        self.encoder = ModuleDict()
        self.real_columns_with_embedding = []
        self.real_columns_direct = []
        for col in self.real_columns:
            self.encoder[col] = RealFeatureProjection(self.feature_embedding_dims[col])
            self.real_columns_with_embedding.append(col)

        for col in self.categorical_columns:
            self.encoder[col] = nn.Embedding(
                self.n_classes[col], self.feature_embedding_dims[col]
            )

        if self.add_ingestion_position:
            self.pos_encoder = ModuleDict()
            for col in self.real_columns + self.categorical_columns:
                self.pos_encoder[col] = nn.Embedding(
                    self.context_length, self.feature_embedding_dims[col]
                )
        else:
            self.pos_encoder = None

        if self.embedding_dim != self.input_dim:
            self.output_projection_layer = nn.Linear(self.input_dim, self.embedding_dim)
        else:
            self.output_projection_layer = None

    @conditional_beartype
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

    @conditional_beartype
    def _with_position(self, col: str, src_t: Tensor) -> Tensor:
        if not self.add_ingestion_position:
            return self.drop(src_t)
        src_p = self._position_encoding(col, src_t.shape[0], src_t.device)
        src_p = cast_floating_to_dtype(src_p, src_t.dtype)
        return self.drop(src_t + src_p)

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        srcs = []
        for col in self.categorical_columns:
            embedding = cast(nn.Embedding, self.encoder[col])
            src_t = self._scale_embedding(
                embedding(embedding_safe_indices(src[col])), self.embedding_dim
            )
            srcs.append(self._with_position(col, src_t))

        for col in self.real_columns:
            layer = cast(nn.Linear, self.encoder[col])
            inp = src[col][:, :, None].to(dtype=layer.weight.dtype)
            src_t = layer(inp)

            srcs.append(self._with_position(col, src_t))

        output = self._recursive_concat(srcs)
        if self.output_projection_layer is not None:
            output = self.output_projection_layer(
                cast_floating_to_module_dtype(output, self.output_projection_layer)
            )
        return output


class PassThroughFeatureIngestion(BaseFeatureIngestion):
    """Pass real-valued columns through without per-feature encoders."""

    @beartype
    def __init__(
        self,
        *,
        real_columns: list[str],
        context_length: int,
        add_ingestion_position: bool,
        dropout: float,
        projection_dim: Optional[int] = None,
        direct_real_dtype_provider: Optional[Callable[[], torch.dtype]] = None,
        device_max_concat_length: int = 12,
    ):
        super().__init__()
        if not real_columns:
            raise ValueError("pass_through ingestion requires at least one real column")

        self.real_columns = real_columns
        self.real_columns_direct = list(real_columns)
        self.context_length = context_length
        self.add_ingestion_position = add_ingestion_position
        self.drop = nn.Dropout(dropout)
        self.input_dim = len(real_columns)
        self.embedding_size = self.input_dim
        self.projection_dim = projection_dim or self.input_dim
        self.direct_real_dtype_provider = direct_real_dtype_provider
        self.device_max_concat_length = device_max_concat_length

        if self.add_ingestion_position:
            self.pos_encoder = ModuleDict()
            for col in self.real_columns:
                self.pos_encoder[col] = nn.Embedding(self.context_length, 1)
        else:
            self.pos_encoder = None

        if self.projection_dim != self.input_dim:
            self.output_projection_layer = nn.Linear(
                self.input_dim, self.projection_dim
            )
        else:
            self.output_projection_layer = None

    @conditional_beartype
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

    @conditional_beartype
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

    @conditional_beartype
    def _with_position(self, col: str, src_t: Tensor) -> Tensor:
        if not self.add_ingestion_position:
            return self.drop(src_t)
        src_p = self._position_encoding(col, src_t.shape[0], src_t.device)
        src_p = cast_floating_to_dtype(src_p, src_t.dtype)
        return self.drop(src_t + src_p)

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        srcs = []
        target_dtype = self._target_dtype(src)
        for col in self.real_columns:
            src_t = src[col].unsqueeze(2).to(dtype=target_dtype)
            srcs.append(self._with_position(col, src_t))

        output = self._recursive_concat(srcs)
        if self.output_projection_layer is not None:
            output = self.output_projection_layer(
                output.to(dtype=self.output_projection_layer.weight.dtype)
            )
        return output


class TemporalConvFeatureIngestion(BaseFeatureIngestion):
    """Apply Conv1D over time after flat-column encoding."""

    @beartype
    def __init__(
        self,
        *,
        base_ingestion: nn.Module,
        base_ingestion_width: int,
        channels: int,
        kernel_size: int,
        dilation_schedule: list[int],
        causal: bool,
        activation_fn: str,
        dropout: float,
        post_conv_norm: str,
        orientation: str,
        context_length: int,
    ):
        super().__init__()
        self.base_ingestion = base_ingestion
        self.input_dim = base_ingestion_width
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_schedule = list(dilation_schedule)
        if not self.dilation_schedule:
            raise ValueError(
                "temporal_conv dilation schedule must contain at least one value"
            )
        self.num_layers = len(self.dilation_schedule)
        self.causal = causal
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    self.input_dim if layer_idx == 0 else self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                )
                for layer_idx, dilation in enumerate(self.dilation_schedule)
            ]
        )
        self.activation = self._activation(activation_fn)
        self.drop = nn.Dropout(dropout)
        self.orientation = orientation
        norm_dim = (
            context_length if self.orientation == "within_column" else self.channels
        )
        self.post_conv_norm = self._norm(post_conv_norm, norm_dim)

    @staticmethod
    @conditional_beartype
    def _activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unknown temporal_conv activation_fn: {name}")

    @staticmethod
    @conditional_beartype
    def _norm(name: str, output_dim: int) -> nn.Module:
        if name == "layer_norm":
            return nn.LayerNorm(output_dim, eps=1e-3)
        if name == "rmsnorm":
            return RMSNorm(output_dim)
        if name == "none":
            return nn.Identity()
        raise ValueError(f"Unknown temporal_conv post_conv_norm: {name}")

    @conditional_beartype
    def _apply_post_conv_norm(self, output: Tensor) -> Tensor:
        if self.orientation == "within_column":
            output = output.transpose(1, 2)
            output = self.post_conv_norm(
                cast_floating_to_module_dtype(output, self.post_conv_norm)
            )
            return output.transpose(1, 2)
        if self.orientation == "within_item_position":
            return self.post_conv_norm(
                cast_floating_to_module_dtype(output, self.post_conv_norm)
            )
        raise ValueError(
            f"Unknown temporal_conv normalization orientation: {self.orientation}"
        )

    @conditional_beartype
    def _temporal_padding(self, dilation: int) -> tuple[int, int]:
        padding = (self.kernel_size - 1) * dilation
        if self.causal:
            return padding, 0
        return padding // 2, padding // 2

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        output = self.base_ingestion(src, metadata)
        for layer, dilation in zip(self.layers, self.dilation_schedule):
            conv_input = output.transpose(1, 2).to(dtype=layer.weight.dtype)
            conv_input = F.pad(conv_input, self._temporal_padding(dilation))
            output = layer(conv_input).transpose(1, 2)
            output = self.drop(self.activation(output))
        return self._apply_post_conv_norm(output)


class _ColumnTokenIngestion(BaseFeatureIngestion):
    @beartype
    def __init__(
        self,
        *,
        columns: list[str],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        token_dim: int,
        add_ingestion_position: bool,
        dropout: float,
    ):
        super().__init__()
        self.columns = columns
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.n_classes = n_classes
        self.context_length = context_length
        self.token_dim = token_dim
        self.add_ingestion_position = add_ingestion_position
        self.drop = nn.Dropout(dropout)

        self.encoder = ModuleDict()
        for col in self.categorical_columns:
            self.encoder[col] = nn.Embedding(self.n_classes[col], self.token_dim)
        for col in self.real_columns:
            self.encoder[col] = RealFeatureProjection(self.token_dim)

        if self.add_ingestion_position:
            self.pos_encoder = nn.Embedding(self.context_length, self.token_dim)
        else:
            self.pos_encoder = None

    @conditional_beartype
    def _encode_column(self, col: str, src: dict[str, Tensor]) -> Tensor:
        if col in self.categorical_columns:
            embedding = cast(nn.Embedding, self.encoder[col])
            return self._scale_embedding(
                embedding(embedding_safe_indices(src[col])), self.token_dim
            )

        layer = cast(nn.Linear, self.encoder[col])
        return layer(src[col][:, :, None].to(dtype=layer.weight.dtype))

    @conditional_beartype
    def _with_position(self, x: Tensor) -> Tensor:
        if not self.add_ingestion_position:
            return self.drop(x)
        pos = torch.arange(0, self.context_length, dtype=torch.long, device=x.device)
        pos = pos.repeat(x.shape[0], 1)
        pos_embedding = self.pos_encoder(pos)  # type: ignore[operator]
        pos_embedding = cast_floating_to_dtype(pos_embedding, x.dtype)
        return self.drop(x + pos_embedding)


class FeaturePoolFeatureIngestion(_ColumnTokenIngestion):
    """Encode each feature as a token and pool feature tokens per time step."""

    @beartype
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.feature_embedding = nn.Parameter(
            torch.zeros(len(self.columns), self.token_dim)
        )

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        encoded = [self._encode_column(col, src) for col in self.columns]
        tokens = torch.stack(encoded, dim=2)
        feature_embedding = self.feature_embedding.to(dtype=tokens.dtype)
        tokens = (
            tokens
            + self._scale_embedding(feature_embedding, self.token_dim)[None, None, :, :]
        )
        return self._with_position(tokens.mean(dim=2))


class GroupedFeatureIngestion(BaseFeatureIngestion):
    """Encode configured column groups and average the group representations."""

    @beartype
    def __init__(
        self,
        *,
        groups: dict[str, list[str]],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        token_dim: int,
        add_ingestion_position: bool,
        dropout: float,
    ):
        super().__init__()
        self.groups = groups
        self.token_dim = token_dim
        self.group_ingestions = ModuleDict()
        categorical_set = set(categorical_columns)
        real_set = set(real_columns)
        for group_name, group_columns in self.groups.items():
            _validate_module_dict_key(
                group_name, f"grouped ingestion group {group_name!r}"
            )
            self.group_ingestions[group_name] = FeaturePoolFeatureIngestion(
                columns=group_columns,
                categorical_columns=[
                    col for col in group_columns if col in categorical_set
                ],
                real_columns=[col for col in group_columns if col in real_set],
                n_classes=n_classes,
                context_length=context_length,
                token_dim=token_dim,
                add_ingestion_position=add_ingestion_position,
                dropout=dropout,
            )

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        outputs = [
            ingestion(src, metadata) for ingestion in self.group_ingestions.values()
        ]
        return torch.stack(outputs, dim=0).mean(dim=0)


class SiameseFeatureIngestion(BaseFeatureIngestion):
    """Apply shared encoders across branch columns and pool their outputs."""

    @beartype
    def __init__(
        self,
        *,
        columns: list[str],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        token_dim: int,
        add_ingestion_position: bool,
        dropout: float,
    ):
        super().__init__()
        self.columns = columns
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.context_length = context_length
        self.token_dim = token_dim
        self.add_ingestion_position = add_ingestion_position
        self.drop = nn.Dropout(dropout)

        if categorical_columns:
            self.categorical_encoder = nn.Embedding(
                max(n_classes[col] for col in categorical_columns), token_dim
            )
        else:
            self.categorical_encoder = None
        if real_columns:
            self.real_encoder = RealFeatureProjection(token_dim)
        else:
            self.real_encoder = None

        if self.add_ingestion_position:
            self.pos_encoder = nn.Embedding(self.context_length, self.token_dim)
        else:
            self.pos_encoder = None

    @conditional_beartype
    def _with_position(self, x: Tensor) -> Tensor:
        if not self.add_ingestion_position:
            return self.drop(x)
        pos = torch.arange(0, self.context_length, dtype=torch.long, device=x.device)
        pos = pos.repeat(x.shape[0], 1)
        pos_embedding = self.pos_encoder(pos)  # type: ignore[operator]
        pos_embedding = cast_floating_to_dtype(pos_embedding, x.dtype)
        return self.drop(x + pos_embedding)

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        encoded = []
        for col in self.columns:
            if col in self.categorical_columns:
                encoded.append(
                    self._scale_embedding(
                        self.categorical_encoder(  # type: ignore[operator]
                            embedding_safe_indices(src[col])
                        ),
                        self.token_dim,
                    )
                )
            else:
                encoded.append(
                    self.real_encoder(
                        src[col][:, :, None].to(dtype=self.real_encoder.weight.dtype)  # type: ignore
                    )
                )
        output = torch.stack(encoded, dim=2).mean(dim=2)
        return self._with_position(output)


@conditional_beartype
def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


@conditional_beartype
def _module_key(indices: tuple[int, ...]) -> str:
    if not indices:
        return "shared"
    return "_".join(str(index) for index in indices)


@conditional_beartype
def _rotate_half_last_dim(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _AxisProjectionBlock(nn.Module):
    """Project one or more cartesian axes into the channel dimension."""

    @beartype
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

    @conditional_beartype
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
        x = cast_floating_to_module_dtype(x, layer)
        x = layer(x)
        return x.reshape(*leading_shape, self.output_dim)

    @conditional_beartype
    def forward(self, x: Tensor) -> Tensor:
        if not self.unshared_axes:
            return self._apply_shared(
                x,
                self.active_axes,
                cast(nn.Linear, self.layers["shared"]),
            )

        output_shape = (
            list(x.shape[:2])
            + [self.axis_sizes[axis] for axis in self.output_axes]
            + [self.output_dim]
        )
        first_layer = cast(nn.Linear, next(iter(self.layers.values())))
        output = x.new_zeros(output_shape, dtype=first_layer.weight.dtype)
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
                cast(nn.Linear, self.layers[_module_key(indices)]),
            )

        return output


class _AxisConvBlock(nn.Module):
    """Apply a native convolution over one to three cartesian axes."""

    CONV_CLASSES = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    @beartype
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

    @conditional_beartype
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
        conv_layer = cast(nn.Conv1d | nn.Conv2d | nn.Conv3d, layer)
        x = x.reshape(-1, self.input_dim, *sweep_shape)
        x = conv_layer(x.to(dtype=conv_layer.weight.dtype))
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

    @conditional_beartype
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
        first_layer = cast(
            nn.Conv1d | nn.Conv2d | nn.Conv3d, next(iter(self.layers.values()))
        )
        output = x.new_zeros(output_shape, dtype=first_layer.weight.dtype)
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
    @beartype
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

    @conditional_beartype
    def forward(self, x: Tensor) -> Tensor:
        x = cast_floating_to_module_dtype(x, self.input_projection)
        x = self.input_projection(x)
        x = cast_floating_to_module_dtype(x, self.attention)
        output, _ = self.attention(x, x, x, need_weights=False)
        return output


class _AxisAttentionBlock(nn.Module):
    """Apply self-attention over one or more cartesian axes."""

    @beartype
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

    @conditional_beartype
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

    @conditional_beartype
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
        first_layer = cast(_AxisAttentionLayer, next(iter(self.layers.values())))
        output_dtype = module_param_dtype(first_layer) or x.dtype
        output = x.new_zeros(output_shape, dtype=output_dtype)
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

    @beartype
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

    @conditional_beartype
    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(2 + self.active_axes.index(axis) for axis in self.axes)
        if self.mode == "mean":
            return x.mean(dim=dims)
        if self.mode == "sum":
            return x.sum(dim=dims)
        return torch.amax(x, dim=dims)


@dataclass(frozen=True)
class AxisShape:
    active_axes: tuple[str, ...]
    channel_dim: int


@dataclass(frozen=True)
class StructuredBlockHandler:
    resolve: Callable[[Any, AxisShape], AxisShape]
    build: Callable[[Any, AxisShape, dict[str, int]], nn.Module]


@beartype
def _validate_block_axes(block: Any, shape: AxisShape) -> None:
    unknown_axes = [axis for axis in block.axes if axis not in shape.active_axes]
    if unknown_axes:
        raise ValueError(
            "Structured ingestion block references unavailable axes: " f"{unknown_axes}"
        )
    if block.type not in {"axis_projection", "axis_conv", "axis_attention"}:
        return
    available_unshared_axes = [
        axis for axis in shape.active_axes if axis not in block.axes
    ]
    invalid_unshared_axes = [
        axis for axis in block.unshared_axes if axis not in available_unshared_axes
    ]
    if invalid_unshared_axes:
        raise ValueError(
            "Structured ingestion block unshared_axes must be a subset of "
            f"non-swept active axes: {invalid_unshared_axes}"
        )


@beartype
def _resolve_axis_projection(block: Any, shape: AxisShape) -> AxisShape:
    _validate_block_axes(block, shape)
    return AxisShape(
        tuple(axis for axis in shape.active_axes if axis not in block.axes),
        block.output_dim,
    )


@beartype
def _resolve_axis_preserving(block: Any, shape: AxisShape) -> AxisShape:
    _validate_block_axes(block, shape)
    return AxisShape(shape.active_axes, block.output_dim)


@beartype
def _resolve_axis_pool(block: Any, shape: AxisShape) -> AxisShape:
    _validate_block_axes(block, shape)
    return AxisShape(
        tuple(axis for axis in shape.active_axes if axis not in block.axes),
        shape.channel_dim,
    )


@beartype
def _build_axis_projection(
    block: Any, shape: AxisShape, axis_sizes: dict[str, int]
) -> nn.Module:
    return _AxisProjectionBlock(
        axes=block.axes,
        unshared_axes=block.unshared_axes,
        output_dim=block.output_dim,
        active_axes=list(shape.active_axes),
        axis_sizes=axis_sizes,
        input_dim=shape.channel_dim,
    )


@beartype
def _build_axis_conv(
    block: Any, shape: AxisShape, axis_sizes: dict[str, int]
) -> nn.Module:
    return _AxisConvBlock(
        axes=block.axes,
        unshared_axes=block.unshared_axes,
        output_dim=block.output_dim,
        kernel_size=block.kernel_size,
        active_axes=list(shape.active_axes),
        axis_sizes=axis_sizes,
        input_dim=shape.channel_dim,
    )


@beartype
def _build_axis_attention(
    block: Any, shape: AxisShape, axis_sizes: dict[str, int]
) -> nn.Module:
    return _AxisAttentionBlock(
        axes=block.axes,
        unshared_axes=block.unshared_axes,
        output_dim=block.output_dim,
        n_head=block.n_head,
        dropout=block.dropout,
        active_axes=list(shape.active_axes),
        axis_sizes=axis_sizes,
        input_dim=shape.channel_dim,
    )


@beartype
def _build_axis_pool(
    block: Any, shape: AxisShape, axis_sizes: dict[str, int]
) -> nn.Module:
    return _AxisPoolBlock(
        axes=block.axes,
        mode=block.mode,
        active_axes=list(shape.active_axes),
    )


STRUCTURED_BLOCK_HANDLERS: dict[str, StructuredBlockHandler] = {
    "axis_projection": StructuredBlockHandler(
        _resolve_axis_projection, _build_axis_projection
    ),
    "axis_conv": StructuredBlockHandler(_resolve_axis_preserving, _build_axis_conv),
    "axis_attention": StructuredBlockHandler(
        _resolve_axis_preserving, _build_axis_attention
    ),
    "axis_pool": StructuredBlockHandler(_resolve_axis_pool, _build_axis_pool),
}


class StructuredFeatureIngestion(_ColumnTokenIngestion):
    """Compile a cartesian layout into an ordered cell tensor."""

    @beartype
    def __init__(
        self,
        *,
        layout: Any,
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        context_length: int,
        result_dim: int,
        add_ingestion_position: bool,
        dropout: float,
        cell_dim: Optional[int] = None,
        axis_embeddings: Optional[Any] = None,
        processing_blocks: Optional[list[Any]] = None,
    ):
        self.layout = layout
        self.axis_names = list(layout.axes)
        self.axis_size_by_name = {
            axis: len(layout.axes[axis]) for axis in self.axis_names
        }
        self.axis_sizes = [len(layout.axes[axis]) for axis in self.axis_names]
        self.expected_dense_shape = tuple(self.axis_sizes)
        self.cell_dim = cell_dim or result_dim
        self.axis_embeddings_config = axis_embeddings
        self.processing_blocks = processing_blocks or []
        self.coordinate_to_index = {
            tuple(coordinates): index
            for index, coordinates in enumerate(
                product(*(layout.axes[axis] for axis in self.axis_names))
            )
        }
        coordinate_to_column = {
            tuple(coordinates[axis] for axis in self.axis_names): column
            for column, coordinates in layout.columns.items()
        }
        self.ordered_columns = [
            coordinate_to_column[coordinates]
            for coordinates in product(*(layout.axes[axis] for axis in self.axis_names))
        ]
        super().__init__(
            columns=self.ordered_columns,
            categorical_columns=categorical_columns,
            real_columns=real_columns,
            n_classes=n_classes,
            context_length=context_length,
            token_dim=self.cell_dim,
            add_ingestion_position=add_ingestion_position,
            dropout=dropout,
        )
        self.result_dim = result_dim
        if self.add_ingestion_position:
            self.pos_encoder = nn.Embedding(self.context_length, self.result_dim)

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
        shape = AxisShape(tuple(self.axis_names), self.cell_dim)
        for block in self.processing_blocks:
            handler = STRUCTURED_BLOCK_HANDLERS[block.type]
            compiled_block = handler.build(block, shape, self.axis_size_by_name)
            self.axis_blocks.append(compiled_block)
            shape = handler.resolve(block, shape)

        self.active_axes_after_blocks = list(shape.active_axes)

    @conditional_beartype
    def _dense_cells(self, src: dict[str, Tensor]) -> Tensor:
        encoded = [self._encode_column(col, src) for col in self.ordered_columns]
        cells = torch.stack(encoded, dim=2)
        return cells.reshape(
            cells.shape[0],
            cells.shape[1],
            *self.expected_dense_shape,
            self.cell_dim,
        )

    @conditional_beartype
    def _axis_broadcast_shape(self, x: Tensor, axis_name: str) -> list[int]:
        axis_idx = self.axis_names.index(axis_name)
        target_dim = axis_idx + 2
        axis_size = self.axis_size_by_name[axis_name]
        broadcast_shape = [1] * (x.ndim - 1) + [self.cell_dim]
        broadcast_shape[target_dim] = axis_size
        return broadcast_shape

    @conditional_beartype
    def _apply_learned_axis_embeddings(self, dense_cells: Tensor) -> Tensor:
        output = dense_cells
        for axis_name, embedding_layer in zip(
            self.axis_embedding_axes, self.axis_embedding_layers
        ):
            axis_size = self.axis_size_by_name[axis_name]
            indices = torch.arange(axis_size, device=output.device)
            embeddings = self._scale_embedding(
                embedding_layer(indices).to(dtype=output.dtype), self.cell_dim
            )
            output = output + embeddings.view(
                *self._axis_broadcast_shape(output, axis_name)
            )
        return output

    @conditional_beartype
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

    @conditional_beartype
    def _apply_rope_axis_embeddings(self, dense_cells: Tensor) -> Tensor:
        output = dense_cells
        for axis_name in self.axis_embedding_axes:
            cos, sin = self._axis_rope_cos_sin(output, axis_name)
            output = (output * cos) + (_rotate_half_last_dim(output) * sin)
        return output

    @conditional_beartype
    def _apply_axis_embeddings(self, dense_cells: Tensor) -> Tensor:
        if self.axis_embedding_type == "none":
            return dense_cells
        if self.axis_embedding_type == "learned":
            return self._apply_learned_axis_embeddings(dense_cells)
        if self.axis_embedding_type == "rope":
            return self._apply_rope_axis_embeddings(dense_cells)
        raise ValueError(f"Unknown axis embedding type: {self.axis_embedding_type}")

    @conditional_beartype
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
    @beartype
    def __init__(self, merge_type: str, branch_dims: dict[str, int], merge_dim: int):
        super().__init__()
        self.merge_type = merge_type
        self.branch_names = list(branch_dims)
        self.branch_dims = branch_dims
        self.merge_dim = merge_dim

        if self.merge_type == "concat":
            input_dim = sum(self.branch_dims.values())
            self.concat_projection = (
                nn.Linear(input_dim, self.merge_dim)
                if input_dim != self.merge_dim
                else nn.Identity()
            )
        elif self.merge_type in {"sum", "gated", "attention"}:
            self.branch_projections = ModuleDict(
                {
                    name: (
                        nn.Linear(branch_dim, self.merge_dim)
                        if branch_dim != self.merge_dim
                        else nn.Identity()
                    )
                    for name, branch_dim in self.branch_dims.items()
                }
            )
            if self.merge_type == "gated":
                self.gate = nn.Linear(
                    len(self.branch_names) * self.merge_dim, len(self.branch_names)
                )
            elif self.merge_type == "attention":
                self.query = nn.Parameter(torch.zeros(self.merge_dim))
        else:
            raise ValueError(
                "merge_type must be one of 'concat', 'sum', 'gated', or "
                f"'attention', got {self.merge_type!r}"
            )

    @conditional_beartype
    def forward(self, branch_outputs: dict[str, Tensor]) -> Tensor:
        if self.merge_type == "concat":
            merged = torch.cat(
                [branch_outputs[name] for name in self.branch_names],
                dim=-1,
            )
            return self.concat_projection(
                cast_floating_to_module_dtype(merged, self.concat_projection)
            )

        projected = [
            self.branch_projections[name](
                cast_floating_to_module_dtype(
                    branch_outputs[name], self.branch_projections[name]
                )
            )
            for name in self.branch_names
        ]
        stacked = torch.stack(projected, dim=2)
        if self.merge_type == "sum":
            return stacked.sum(dim=2)

        if self.merge_type == "gated":
            gate_input = torch.cat(projected, dim=-1)
            weights = torch.softmax(
                self.gate(cast_floating_to_module_dtype(gate_input, self.gate)),
                dim=-1,
            )
            return (stacked * weights[:, :, :, None]).sum(dim=2)

        query = self.query.to(dtype=stacked.dtype)
        scores = (stacked * query[None, None, None, :]).sum(dim=-1)
        scores = scores / math.sqrt(self.merge_dim)
        weights = torch.softmax(scores, dim=-1)
        return (stacked * weights[:, :, :, None]).sum(dim=2)


class CompositeFeatureIngestion(BaseFeatureIngestion):
    @beartype
    def __init__(
        self,
        *,
        branches: dict[str, nn.Module],
        branch_widths: dict[str, int],
        merge_type: str,
        merge_dim: int,
    ):
        super().__init__()
        for branch_name in branches:
            _validate_module_dict_key(
                branch_name, f"Composite ingestion branch {branch_name!r}"
            )
        self.branches = ModuleDict(branches)
        self.merge = IngestionMerge(
            merge_type,
            branch_widths,
            merge_dim,
        )

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        branch_outputs = {
            name: branch(src, metadata) for name, branch in self.branches.items()
        }
        return self.merge(branch_outputs)


@beartype
def _split_columns(
    columns: list[str], categorical_columns: list[str], real_columns: list[str]
) -> tuple[list[str], list[str]]:
    categorical_set = set(categorical_columns)
    real_set = set(real_columns)
    typed_columns = categorical_set | real_set
    dropped_columns = [col for col in columns if col not in typed_columns]
    if dropped_columns:
        raise ValueError(
            "Ingestion columns must be declared in categorical_columns or "
            f"real_columns; would drop columns: {dropped_columns}"
        )
    return (
        [col for col in columns if col in categorical_set],
        [col for col in columns if col in real_set],
    )


@beartype
def _feature_dims_for_columns(
    ingestion_config: Any, columns: list[str]
) -> Optional[dict[str, int]]:
    feature_embedding_dims = ingestion_config.feature_embedding_dims
    if feature_embedding_dims is None:
        return None
    return {col: feature_embedding_dims[col] for col in columns}


@beartype
def resolve_ingestion_plan(hparams: Any):
    from sequifier.model.ingestion_compiler import resolve_ingestion_plan as resolve

    return resolve(hparams)


@beartype
def compile_feature_ingestion(
    *,
    hparams: Any,
    direct_real_dtype_provider: Callable[[], torch.dtype],
    device_max_concat_length: int,
):
    from sequifier.model.ingestion_compiler import compile_feature_ingestion as compile

    return compile(
        hparams=hparams,
        direct_real_dtype_provider=direct_real_dtype_provider,
        device_max_concat_length=device_max_concat_length,
    )


@beartype
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
