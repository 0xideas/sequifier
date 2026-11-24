import os
import warnings
from typing import Any, Literal, Optional, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.command_ownership import (
    CommandRunSettings,
    PreprocessingRunSettings,
)
from sequifier.config.composition import (
    load_composed_yaml_config,
    merge_config_fragments,
)
from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.config.naming import ConfigModel
from sequifier.config.sequence_import_config import (
    PreparePolicyConfig,
    SequenceImportConfig,
)
from sequifier.config.splits import SequencePreprocessingSettings
from sequifier.config.splits import WindowSplit as WindowSplit
from sequifier.helpers import (
    canonicalize_polars_dtype_name,
    is_float_dtype_name,
    is_integer_dtype_name,
    try_catch_excess_keys,
)
from sequifier.typechecking import beartype


@beartype
def load_preprocessor_config(
    config_path: str, args_config: dict
) -> "PreprocessorModel | SequenceImportConfig | PreparePolicyConfig | PrepareInferenceConfig | ExportMetadataConfig":
    """Load preprocessing YAML plus CLI overrides."""
    config_values = load_composed_yaml_config(config_path)

    from sequifier.config.command_ownership import apply_run_overrides

    config_values, args_config = apply_run_overrides(config_values, args_config)
    config_values = merge_config_fragments((config_values, args_config))

    if config_values.get("operation") == "import_sequences":
        return SequenceImportConfig.model_validate(config_values)
    if config_values.get("operation") == "prepare_policy":
        return try_catch_excess_keys(config_path, PreparePolicyConfig, config_values)
    if config_values.get("operation") == "export_metadata":
        return try_catch_excess_keys(config_path, ExportMetadataConfig, config_values)
    if config_values.get("operation") == "prepare_inference":
        return try_catch_excess_keys(config_path, PrepareInferenceConfig, config_values)
    return try_catch_excess_keys(
        config_path,
        PreprocessorModel,
        config_values,
    )


class ExportMetadataConfig(ConfigModel):
    operation: Literal["export_metadata"]
    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    model_path: str
    model_interface: str | None = None
    output_path: str


class PrepareInferenceConfig(ConfigModel):
    operation: Literal["prepare_inference"]
    run: PreprocessingRunSettings = Field(default_factory=PreprocessingRunSettings)
    preprocessing_data_path: str
    metadata_config_path: str
    output_path: str
    read_format: Literal["csv", "parquet"] | None = None
    static_data_path: str | None = None
    passthrough_columns: list[str] = Field(default_factory=list)
    image_paths: dict[str, str] = Field(default_factory=dict)
    preconditioning_image_paths: dict[str, str] | None = None
    mask_column: str | None = None


class CardinalityHashingModel(ConfigModel):
    """Hash non-retained categorical values into a bounded set of buckets."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_buckets: int = Field(
        ge=1,
    )
    seed: int = 0


class CardinalityLimitModel(ConfigModel):
    """Bound the vocabulary produced for one categorical column."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    min_count: Optional[int] = Field(
        default=None,
        ge=1,
    )
    max_categories: Optional[int] = Field(
        default=None,
        ge=1,
    )
    hashing: Optional[CardinalityHashingModel] = None

    @model_validator(mode="after")
    def validate_options(self) -> "CardinalityLimitModel":
        if self.min_count is not None and self.max_categories is not None:
            raise ValueError(
                "max_categories and min_count are mutually exclusive cardinality limits"
            )
        if (
            self.min_count is None
            and self.max_categories is None
            and self.hashing is None
        ):
            raise ValueError(
                "cardinality configuration requires max_categories, min_count, or hashing"
            )
        return self


class PreprocessorModel(SequencePreprocessingSettings):
    """Top-level preprocessing config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    depth_layouts: DepthLayoutRegistryModel = Field(
        default_factory=DepthLayoutRegistryModel
    )
    run: PreprocessingRunSettings = Field(default_factory=PreprocessingRunSettings)
    preprocessing_data_path: str
    static_data_path: Optional[str] = None
    image_sources: dict[str, ImageSourceConfig] = Field(default_factory=dict)
    time_alignment_column: str | None = None
    preconditioning_image: PreconditioningImageConfig | None = None
    static_categorical_columns: list[str] = Field(default_factory=list)
    static_real_columns: list[str] = Field(default_factory=list)

    read_format: str = "csv"
    write_format: str = "parquet"
    sequence_view: Literal["decisions", "observations"] = "decisions"
    indexed_output_path: str | None = Field(default=None, min_length=1)
    indexed_passthrough: list[str] = Field(default_factory=list)
    merge_output: bool = True
    allow_sequence_splitting: bool = False
    selected_columns: Optional[list[str]] = None
    column_data_types: Optional[dict[str, str]] = None
    normalize_real_columns: bool = True
    normalize_on_all_data: bool = False

    max_rows: Optional[int] = None
    batches_per_file: int = 1024
    process_by_file: bool = True
    reprocess: bool = False

    use_precomputed_maps: Optional[list[str]] = None
    cardinality_config: dict[str, CardinalityLimitModel] = Field(default_factory=dict)
    metadata_config_path: Optional[str] = None
    mask_column: Optional[str] = None
    curriculum_column: Optional[Union[str, list[str]]] = None

    @property
    def project_root(self):
        return self.run.project_root

    @property
    def seed(self):
        return self.run.reproducibility.seed

    @property
    def num_workers(self):
        return self.run.execution.data_loading.num_workers

    @model_validator(mode="after")
    def validate_resources(self):
        from sequifier.io.sample_order import curriculum_columns

        if self.indexed_passthrough and self.indexed_output_path is None:
            raise ValueError("indexed_passthrough requires indexed_output_path")
        passthrough = set(self.indexed_passthrough)
        reserved = {
            "",
            "sequenceId",
            "itemPosition",
            self.mask_column,
            self.split_column,
            self.time_alignment_column,
            *curriculum_columns(self.curriculum_column),
            *(layout.position_column for _, layout in self.depth_layouts.items()),
            *self.depth_layouts.deep_columns,
        }
        if len(passthrough) != len(self.indexed_passthrough) or reserved & passthrough:
            raise ValueError("Indexed passthrough must name distinct auxiliary columns")
        if conflicts := passthrough.intersection(self.selected_columns or ()):
            raise ValueError(
                f"Columns cannot be selected features and passthrough: {sorted(conflicts)}"
            )
        return self

    @field_validator("image_sources")
    @classmethod
    def validate_image_source_names(cls, sources):
        if any(not name.isidentifier() for name in sources):
            raise ValueError("Image view names must be identifiers")
        return sources

    @field_validator("time_alignment_column")
    @classmethod
    def validate_time_alignment_column(cls, column):
        if column in {"sequenceId", "itemPosition"}:
            raise ValueError("time_alignment_column must name a separate input column")
        return column

    @model_validator(mode="after")
    def validate_static_source(self):
        if self.static_data_path is None:
            if (self.static_categorical_columns or self.static_real_columns) and not (
                self.read_format == "sequence"
                or self.preprocessing_data_path.lower().endswith(".json")
            ):
                raise ValueError("Static column roles require static_data_path")
            return self
        path = self.static_data_path
        if not os.path.isfile(path) or not path.lower().endswith((".csv", ".parquet")):
            raise ValueError(
                "static_data_path must name an existing CSV or Parquet file"
            )
        if set(self.static_categorical_columns) & set(self.static_real_columns):
            raise ValueError("Static categorical and real columns must be disjoint")
        if len(self.static_categorical_columns) != len(
            set(self.static_categorical_columns)
        ) or len(self.static_real_columns) != len(set(self.static_real_columns)):
            raise ValueError("Static column roles must not contain duplicates")
        if any(
            not name or name == "sequenceId"
            for name in self.static_categorical_columns + self.static_real_columns
        ):
            raise ValueError(
                "Static column roles cannot name sequenceId or an empty column"
            )
        return self

    @field_validator("curriculum_column")
    @classmethod
    def validate_curriculum_columns(cls, value):
        columns = [value] if isinstance(value, str) else value
        if columns is not None and (
            not columns
            or any(not column for column in columns)
            or any(
                column.startswith("__sequifier_curriculum_value_") for column in columns
            )
            or len(columns) != len(set(columns))
        ):
            raise ValueError("curriculum_column must contain unique, non-empty names")
        return value

    @field_validator("preprocessing_data_path", "static_data_path", mode="before")
    @classmethod
    def resolve_input_path(cls, value, info):
        run = info.data.get("run")
        if value is not None and run is not None and run.project_root != ".":
            from sequifier.helpers import normalize_path

            return normalize_path(str(value), run.project_root)
        return value

    @field_validator("preprocessing_data_path")
    @classmethod
    @beartype
    def validate_preprocessing_data_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
        return v

    @field_validator("read_format")
    @classmethod
    @beartype
    def validate_read_format(cls, v: str) -> str:
        supported_formats = ["csv", "parquet", "sequence"]
        if v not in supported_formats:
            raise ValueError(
                f"Currently only {', '.join(supported_formats)} are supported "
                "for preprocessing input"
            )
        return v

    @field_validator("write_format")
    @classmethod
    @beartype
    def validate_write_format(cls, v: str) -> str:
        supported_formats = ["csv", "parquet", "pt"]
        if v not in supported_formats:
            raise ValueError(
                f"Currently only {', '.join(supported_formats)} are supported "
                "for preprocessing output"
            )
        return v

    @field_validator("merge_output")
    @classmethod
    @beartype
    def validate_format2(cls, v: bool, info: Any):
        write_format = info.data.get("write_format")

        if write_format == "pt" and v is True:
            raise ValueError(
                "With write_format 'pt', merge_output must be set to False"
            )

        if (
            write_format == "parquet"
            and v is True
            and not info.data.get("indexed_output_path")
        ):
            warnings.warn(
                "Training on distributed data in parquet format takes significantly more CPU per GPU than with 'pt'. Inferring on distributed data in parquet is less efficient than with 'pt'"
            )

        # Allow "parquet" to have merge_output = False
        if write_format not in ["pt", "parquet"] and v is False:
            raise ValueError(
                f"With write_format '{write_format}', merge_output must be set to True. "
                "Only 'pt' and 'parquet' formats support uncombined (split) output."
            )

        return v

    @model_validator(mode="after")
    def validate_output_format(self):
        # Field validators do not run for an omitted, defaulted merge_output.
        if self.write_format == "pt" and self.merge_output:
            raise ValueError(
                "With write_format 'pt', merge_output must be set to False"
            )
        return self

    @field_validator("batches_per_file")
    @classmethod
    @beartype
    def validate_batches_per_file(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batches_per_file must be a positive integer")
        return v

    @field_validator("column_data_types")
    @classmethod
    @beartype
    def validate_column_types(
        cls, v: Optional[dict[str, str]], info: Any
    ) -> Optional[dict[str, str]]:
        if not v:
            return None

        normalized = {
            column: (
                dtype
                if dtype.startswith(("Date", "Datetime"))
                else canonicalize_polars_dtype_name(dtype)
            )
            for column, dtype in v.items()
        }
        selected_columns = info.data.get("selected_columns")
        if selected_columns is not None:
            missing_columns = [
                column for column in selected_columns if column not in normalized
            ]
            if missing_columns:
                raise ValueError(
                    "column_data_types must include every selected column. "
                    f"Missing: {missing_columns}"
                )

        return normalized

    @model_validator(mode="after")
    @beartype
    def validate_mask_column_requires_metadata(self) -> "PreprocessorModel":
        categorical = set(self.categorical_columns or [])
        real = set(self.real_columns or [])
        declared = categorical | real
        cardinality_columns = set(self.cardinality_config)
        if cardinality_columns & real:
            raise ValueError(
                "cardinality_config may only reference categorical columns. "
                f"Real columns: {sorted(cardinality_columns & real)}"
            )
        if categorical and cardinality_columns - categorical:
            raise ValueError(
                "When categorical_columns is set, cardinality_config columns must "
                f"be categorical. Invalid: {sorted(cardinality_columns - categorical)}"
            )
        if self.selected_columns is not None:
            unselected_cardinality = cardinality_columns - set(self.selected_columns)
            if unselected_cardinality:
                raise ValueError(
                    "cardinality_config columns must be selected columns. "
                    f"Not selected: {sorted(unselected_cardinality)}"
                )
        precomputed_overlap = cardinality_columns & set(self.use_precomputed_maps or [])
        if precomputed_overlap:
            raise ValueError(
                "cardinality_config cannot be combined with precomputed maps "
                "(use_precomputed_maps) for the same columns: "
                f"{sorted(precomputed_overlap)}"
            )
        if self.selected_columns is not None:
            unknown = declared - set(self.selected_columns)
            if unknown:
                raise ValueError(
                    "categorical_columns and real_columns must be selected columns. "
                    f"Not selected: {sorted(unknown)}"
                )

        if self.column_data_types is not None:
            missing_types = declared - set(self.column_data_types)
            if missing_types:
                raise ValueError(
                    "column_data_types must include every explicitly classified "
                    f"column. Missing: {sorted(missing_types)}"
                )
            categorical_with_non_integer_types = sorted(
                column
                for column in categorical
                if not is_integer_dtype_name(self.column_data_types[column])
            )
            if categorical_with_non_integer_types:
                raise ValueError(
                    "Categorical columns require integer column_data_types. "
                    f"Invalid: {categorical_with_non_integer_types}"
                )
            real_with_non_float_types = sorted(
                column
                for column in real
                if not is_float_dtype_name(self.column_data_types[column])
            )
            if real_with_non_float_types:
                raise ValueError(
                    "Real columns require floating-point column_data_types. "
                    f"Invalid: {real_with_non_float_types}"
                )

        self.validate_split_method()
        if self.split_method == "value_cutoff":
            if self.split_column == self.mask_column:
                raise ValueError("split_column cannot also be mask_column")
            curriculum_columns = (
                [self.curriculum_column]
                if isinstance(self.curriculum_column, str)
                else self.curriculum_column or []
            )
            if self.split_column in curriculum_columns:
                raise ValueError("split_column cannot also be a curriculum_column")
            temporal_columns = {
                column
                for column, dtype in (self.column_data_types or {}).items()
                if dtype.startswith(("Date", "Datetime"))
            }
            if temporal_columns - {self.split_column, self.time_alignment_column}:
                raise ValueError(
                    "Only split_column or time_alignment_column may have a timestamp column_data_type"
                )
        elif any(
            column != self.time_alignment_column
            and dtype.startswith(("Date", "Datetime"))
            for column, dtype in (self.column_data_types or {}).items()
        ):
            raise ValueError(
                "Timestamp columns require time_alignment_column or value_cutoff split_column"
            )
        if self.depth_layouts:
            if self.split_method == "value_cutoff":
                raise ValueError(
                    "value_cutoff splitting is not supported with depth_layouts"
                )
            if (
                self.write_format != "pt"
                or self.merge_output
                or self.mask_column is not None
            ):
                raise ValueError(
                    "Depth preprocessing requires write_format: pt, merge_output: false, and no mask_column"
                )
            if len(self.depth_layouts.root) != 1:
                raise ValueError("Raw preprocessing currently accepts one depth layout")
            if self.selected_columns is not None and not set(
                self.depth_layouts.deep_columns
            ) <= set(self.selected_columns):
                raise ValueError("selected_columns must include every depth feature")
            positions = {
                layout.position_column for _, layout in self.depth_layouts.items()
            }
            if positions.intersection(
                self.selected_columns or []
            ) or positions.intersection(self.column_data_types or {}):
                raise ValueError(
                    "Depth position columns are coordinates, not selected features or column_data_types"
                )
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")
        if self.mask_column in ("sequenceId", "itemPosition"):
            raise ValueError("mask_column cannot be sequenceId or itemPosition")
        curriculum_columns = (
            [self.curriculum_column]
            if isinstance(self.curriculum_column, str)
            else self.curriculum_column or []
        )
        if set(curriculum_columns) & {"sequenceId", "itemPosition"}:
            raise ValueError("curriculum_column cannot be sequenceId or itemPosition")
        if curriculum_columns and (
            self.mask_column in curriculum_columns
            or set(curriculum_columns)
            & {layout.position_column for _, layout in self.depth_layouts.items()}
            or set(curriculum_columns) & set(self.depth_layouts.deep_columns)
        ):
            raise ValueError(
                "curriculum_column cannot also be a mask or depth feature/position "
                "column"
            )
        self.validate_window_placement()
        if self.prediction_aligned_splits and self.allow_sequence_splitting:
            raise ValueError(
                "Aligned placement requires allow_sequence_splitting: false"
            )
        return self
