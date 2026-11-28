"""Explicit sequence import with the same source and authored task contracts."""

import json
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import AliasChoices, Field, field_validator, model_validator

from sequifier.config.cardinality import CardinalityLimitModel
from sequifier.config.column_names import (
    validate_column_names,
    validate_policy_column_types,
)
from sequifier.config.command_ownership import (
    CommandRunSettings,
    PreprocessingRunSettings,
)
from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.config.policy_config import (
    PolicyBinding,
    PolicyConfigBase,
    ResolvedPolicyBinding,
)
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.splits import SequencePreprocessingSettings

CommandRun = TypeVar("CommandRun", bound=CommandRunSettings)


def select_policy_input(values, path, *, source_format=None):
    """Replace the configured source location while preserving feature bindings."""
    values.pop("preprocessing_data_path", None)
    source = values.setdefault("source", {})
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("preprocessing_data_path requires a nonempty path")
    run = values.get("run", {})
    root = (
        run.project_root
        if isinstance(run, CommandRunSettings)
        else run.get("project_root", ".")
    )
    location = Path(root) / path
    if location.is_dir() and values.get("operation") == "prepare_policy":
        from sequifier.preprocess import _folder_input_files

        read_format = values.get("read_format") or "csv"
        if read_format not in {"csv", "parquet"}:
            raise ValueError(
                "Policy preprocessing folders require read_format: csv or parquet"
            )
        static = values.get("static_data_path") or source.get("static_data_path")
        output = values.get("indexed_output_path") or values.get("output_path")
        paths = _folder_input_files(
            str(location),
            read_format,
            str(Path(root) / static) if static else None,
            exclude_directory=(Path(root) / output).resolve() if output else None,
        )
        if not paths:
            raise ValueError(f"No {read_format} inputs found in {location}")
        for key in ("split", "manifest_path"):
            source.pop(key, None)
        source["paths"] = [str(Path(file).resolve()) for file in paths]
        return
    if (
        source_format is None
        and location.is_file()
        and (
            values.get("read_format") == "sequence"
            or location.suffix.lower() not in {".csv", ".parquet", ".pt"}
        )
    ):
        with location.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(
                f"Expected a sequence manifest, indexed view or trajectory in {location}"
            )
        source_format = document.get("format")
    for key in ("paths", "split", "manifest_path"):
        source.pop(key, None)
    if source_format == "sequifier.sequence-data":
        source["manifest_path"] = str(path)
        for key in ("encoding", "episode_end", "format"):
            source.pop(key, None)
    else:
        source["paths"] = [str(path)]
        if source_format == "sequifier.sequence-view":
            from sequifier.io.indexed_sequences import IndexedView

            source["format"] = "prepared"
            source.setdefault(
                "metadata_config_path", str(IndexedView.load(location).metadata_path)
            )


class EpisodeSourceConfig(PolicyConfigBase, Generic[CommandRun]):
    run: CommandRun
    source: SequenceSourceConfig
    policy_binding: PolicyBinding
    initialize_from: str | None = None
    model_interface: str | None = None
    reprocess: bool = False

    @model_validator(mode="after")
    def validate_interface_source(self):
        self.policy_binding.validate_source_units(self.source.units)
        if self.model_interface is not None and not self.initialize_from:
            raise ValueError("model_interface requires initialize_from")
        return self

    @property
    def project_root(self):
        return self.run.project_root

    def resolved_source(self):
        from sequifier.trajectory.sources import resolve_source

        return resolve_source(self.source, self.project_root)

    def resolved_policy(self, source=None) -> ResolvedPolicyBinding:
        if self.initialize_from:
            from sequifier.artifacts.model_artifact import read_execution_artifact
            from sequifier.artifacts.model_config import (
                model_execution_from_model_config,
            )
            from sequifier.rl.attachment import bind_policy

            execution, name = model_execution_from_model_config(
                read_execution_artifact(
                    Path(self.project_root) / self.initialize_from
                ).model_config.values,
                interface_name=self.model_interface,
            )
            if self.model_interface is None and len(execution.interface_names) > 1:
                raise ValueError(
                    "Select model_interface when initialize_from has multiple interfaces"
                )
            return bind_policy(execution, name, self.policy_binding)
        observations = list(self.source.columns.observations)
        static = list(self.source.columns.static_inputs)
        if not observations:
            resolved = source or self.resolved_source()
            manifest = resolved.manifest or {}
            observations = list(manifest.get("observation_columns", ()))
            static = static or list(manifest.get("static_columns", ()))
        if not observations:
            raise ValueError(
                "sequence import requires source.columns.observations or initialize_from"
            )
        # Import validates and stores domain episodes, without constructing a model.
        return ResolvedPolicyBinding(
            **self.policy_binding.model_dump(),
            context_length=1,
            interface_name="default",
            observation_columns=observations,
            static_columns=static,
            action_order=list(self.policy_binding.actions),
        )


class SequenceImportConfig(EpisodeSourceConfig[CommandRunSettings]):
    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    output_path: str = Field(min_length=1)
    operation: Literal["import_sequences"]
    passthrough_columns: list[str] = Field(default_factory=list)
    episodes_per_shard: int = Field(default=64, gt=0)
    mask_column: str | None = Field(default=None, min_length=1)

    @field_validator("mask_column")
    @classmethod
    def validate_mask_column(cls, name):
        if name:
            validate_column_names([name], "mask_column")
        return name

    @field_validator("passthrough_columns")
    @classmethod
    def validate_passthrough_columns(cls, names):
        validate_column_names(names, "passthrough_columns", allowed=("__collection",))
        if not all(names) or len(names) != len(set(names)):
            raise ValueError("passthrough_columns must contain unique, non-empty names")
        return names


class PreparePolicyConfig(
    EpisodeSourceConfig[PreprocessingRunSettings], SequencePreprocessingSettings
):
    run: PreprocessingRunSettings = Field(default_factory=PreprocessingRunSettings)
    indexed_output_path: str = Field(
        min_length=1,
        validation_alias=AliasChoices("indexed_output_path", "output_path"),
    )
    operation: Literal["prepare_policy"]
    read_format: Literal["csv", "parquet", "sequence"] | None = None
    selected_columns: list[str] | None = None
    static_data_path: str | None = None
    static_categorical_columns: list[str] = Field(default_factory=list)
    static_real_columns: list[str] = Field(default_factory=list)
    depth_layouts: DepthLayoutRegistryModel = Field(
        default_factory=DepthLayoutRegistryModel
    )
    image_sources: dict[str, ImageSourceConfig] = Field(default_factory=dict)
    preconditioning_image: PreconditioningImageConfig | None = None
    time_alignment_column: str | None = None
    column_data_types: dict[str, str] | None = None
    curriculum_column: str | list[str] | None = None
    use_precomputed_maps: list[str] | None = None
    cardinality_config: dict[str, CardinalityLimitModel] = Field(default_factory=dict)
    normalize_on_all_data: bool = False

    normalize_real_columns: bool = True
    metadata_config_path: str | None = None
    mask_column: str | None = Field(default=None, min_length=1)
    indexed_passthrough: list[str] = Field(default_factory=list)

    @field_validator("column_data_types")
    @classmethod
    def validate_dtypes(cls, values, info):
        return validate_policy_column_types(
            values,
            temporal_columns=(
                info.data.get("split_column"),
                info.data.get("time_alignment_column"),
            ),
        )

    @model_validator(mode="before")
    @classmethod
    def resolve_input_path(cls, values):
        if not isinstance(values, dict):
            return values
        declared_source = values.get("source", {})
        source = (
            declared_source.model_dump()
            if isinstance(declared_source, SequenceSourceConfig)
            else dict(declared_source)
        )
        values = {**values, "source": source}
        if "preprocessing_data_path" in values:
            if (
                source.get("paths")
                or source.get("manifest_path")
                or source.get("split")
            ):
                raise ValueError("Select preprocessing_data_path or a source location")
            select_policy_input(values, values["preprocessing_data_path"])
        if values.get("static_data_path"):
            if source.get("static_data_path") not in (None, values["static_data_path"]):
                raise ValueError("Select one static_data_path")
            source["static_data_path"] = values["static_data_path"]
        return values

    def resolved_policy(self, source=None):
        if self.initialize_from:
            policy = super().resolved_policy(source)
            temporal = {
                name
                for name in policy.observation_columns
                if not name.startswith(
                    ("__image:", "__image_mask:", "depth_valid_mask:")
                )
            }
            if self.selected_columns is not None and temporal - set(
                self.selected_columns
            ):
                raise ValueError(
                    "selected_columns must include the initialized policy's observations"
                )
            static = set(self.static_categorical_columns + self.static_real_columns)
            if static and static != set(policy.static_columns) - {
                "__preconditioning_image"
            }:
                raise ValueError(
                    "Static column roles differ from the initialized policy"
                )
            return policy
        columns = self.source.columns
        static = self.static_categorical_columns + self.static_real_columns
        if (
            static
            and columns.static_inputs
            and set(static) != set(columns.static_inputs)
        ):
            raise ValueError(
                "Static column roles differ from source.columns.static_inputs"
            )
        observations = columns.observations
        if self.selected_columns is not None:
            manifest = (source or self.resolved_source()).manifest or {}
            known = observations or manifest.get("observation_columns", [])
            observations = (
                [name for name in self.selected_columns if name in known]
                if known
                else [
                    name
                    for name in self.selected_columns
                    if name not in self.policy_binding.actions
                ]
            )
        source_config = self.source.model_copy(
            update={
                "columns": columns.model_copy(
                    update={
                        "observations": [
                            name
                            for name in observations
                            if name != self.time_alignment_column
                            and not (
                                name == self.split_column
                                and (self.column_data_types or {})
                                .get(name, "")
                                .startswith(("Date", "Datetime"))
                            )
                        ],
                        "static_inputs": static or columns.static_inputs,
                    }
                )
            }
        )
        return EpisodeSourceConfig.resolved_policy(
            self.model_copy(update={"source": source_config}), source
        )

    @model_validator(mode="after")
    def validate_resources(self):
        from sequifier.io.sample_order import curriculum_columns

        for field in (
            "selected_columns",
            "column_data_types",
            "categorical_columns",
            "real_columns",
            "static_categorical_columns",
            "static_real_columns",
            "cardinality_config",
            "use_precomputed_maps",
        ):
            validate_column_names(getattr(self, field) or (), field)
        validate_column_names(
            self.indexed_passthrough, "indexed_passthrough", allowed=("__collection",)
        )
        if self.mask_column:
            validate_column_names([self.mask_column], "mask_column")
        if self.split_column:
            validate_column_names([self.split_column], "split_column")
        if self.time_alignment_column:
            validate_column_names([self.time_alignment_column], "time_alignment_column")
        if any(not name.isidentifier() for name in self.image_sources):
            raise ValueError("Image view names must be identifiers")
        curriculum = curriculum_columns(self.curriculum_column)
        validate_column_names(curriculum, "curriculum_column")
        if len(set(curriculum)) != len(curriculum) or set(curriculum) & {
            "",
            self.mask_column,
            self.split_column,
        }:
            raise ValueError("curriculum_column must name distinct auxiliary columns")
        static = self.static_categorical_columns + self.static_real_columns
        if len(set(static)) != len(static):
            raise ValueError("Static column roles must be disjoint and unique")
        observed = set(self.source.columns.observations)
        action_roles = {
            name: field.type
            for name, field in self.policy_binding.actions.items()
            if name not in observed and name not in (self.selected_columns or ())
        }
        action_roles.update(
            {
                name: self.policy_binding.actions[feedback.action].type
                for name, feedback in self.policy_binding.feedback.items()
                if feedback.action in self.policy_binding.actions
                and name not in observed
            }
        )
        real_columns = {
            *(self.real_columns or ()),
            *self.static_real_columns,
            *(name for name, role in action_roles.items() if role == "real"),
        }
        for name in real_columns:
            if (
                self.column_data_types
                and name in self.column_data_types
                and self.column_data_types[name] != "Float32"
            ):
                raise ValueError(
                    f"column_data_types.{name}: policy real columns require Float32"
                )
        names = self.indexed_passthrough
        if len(set(names)) != len(names) or set(names) & {
            "",
            "sequenceId",
            "itemPosition",
            self.mask_column,
        }:
            raise ValueError("Indexed passthrough must name distinct auxiliary columns")
        limited = set(self.cardinality_config)
        # Action names can also denote independent observed features. Validate
        # cardinality against resolved observation/static columns during preparation.
        if limited & set(self.policy_binding.feedback):
            raise ValueError("cardinality_config cannot change feedback vocabularies")
        if limited & set(self.real_columns or ()) or (
            self.categorical_columns and limited - set(self.categorical_columns)
        ):
            raise ValueError("cardinality_config must select categorical features")
        self.validate_split_method()
        self.validate_window_placement()
        return self
