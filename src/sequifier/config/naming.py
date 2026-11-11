"""Canonical authored names and run-seed resolution.

Removed names are diagnosed, never translated into accepted aliases. Arbitrary
user dictionaries (columns, constructor arguments, and environment config) are
not inspected as configuration schemas.
"""

import secrets
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, model_validator

RENAMED_FIELDS = {
    "interface": "interfaces.<name>",
    "dataset": "dataset_training.<name>",
    "part": "parts.<name>",
    "policy_binding": "policy_bindings.<interface>",
    "dim_model": "model_dim",
    "n_heads": "num_heads",
    "n_kv_heads": "num_kv_heads",
    "embedding_size": "total_embedding_dim",
    "positional_encoding_scope": "position_encoding_scope",
    "post_conv_norm": "post_conv_normalization",
    "frame_pool": "frame_pooling",
    "image_frame_pool": "image_frame_pooling",
    "support": "support_length",
    "decoder_support": "decoder_support_length",
    "device_max_concat_length": "max_concat_tensors",
    "accumulation_steps": "accumulation_batches",
    "minibatch_size": "batch_size",
    "updates": "iterations",
    "groups_per_update": "groups_per_iteration",
    "rollout_length": "rollout_env_steps",
    "log_interval": "log_interval_batches",
    "early_stopping_epochs": "early_stopping_patience_epochs",
    "gradient_clip": "gradient_clip_norm",
    "value_clip": "value_clip_delta",
    "target_kl": "old_policy_target_kl",
    "kl_coefficient": "reference_kl_coefficient",
    "max_weight": "max_advantage_weight",
    "weight": "selection_weight",
    "save_interval_batches": "snapshot_interval_batches",
    "save_interval_minutes": "snapshot_interval_minutes",
    "save_latest_interval_minutes": "latest_checkpoint_interval_minutes",
    "save_interval_val_loss": "evaluate_on_snapshot",
    "calculate_validation_loss_on_initialization": "evaluate_before_training",
    "min_freq": "min_count",
    "top_k": "max_categories",
    "k": "max_categories",
    "buckets": "num_buckets",
    "column_types": "column_data_types",
    "standard_deviation": "std",
    "n_cores": "num_workers",
    "image_path": "image_paths",
    "preconditioning_image_path": "preconditioning_image_paths",
    "manifest": "manifest_path",
    "freezing_except": "freeze_except_groups",
    "model_type": "task: forecast | policy | embedding",
    "model_path": "model_paths (a list)",
    "autoregressive": "autoregressive_generation",
    "global_seed": "sampler_seed",
    "model_config_write_path": "trial_config_directory",
    "evaluation_script": "evaluation_script_path",
    "evaluation_inference_config": "evaluation_inference_config_path",
}


class ConfigModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def reject_removed_names(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        fields = cls.model_fields
        replacements = dict(RENAMED_FIELDS)
        for key in ("evaluation_interval", "checkpoint_interval"):
            unit = "iterations" if f"{key}_iterations" in fields else "epochs"
            replacements[key] = f"{key}_{unit}"
        if "direction" in fields:
            replacements["mode"] = "direction: min | max"
        if "pooling" in fields:
            replacements["mode"] = "pooling"
        if "hidden_dim" in fields:
            replacements["dim"] = "hidden_dim"
        if "max_depth_length" in fields:
            replacements["context_length"] = "max_depth_length"
        if "freeze_groups" in fields:
            replacements["freeze"] = "freeze_groups"
        elif "freeze_modules" in fields:
            replacements["freeze"] = "freeze_modules"
        if "paths" in fields and cls.__name__ == "PreconditioningImageConfig":
            replacements["path"] = "paths (a view-to-path mapping)"
        if "interface_metadata_paths" in fields:
            replacements["metadata_config_path"] = "interface_metadata_paths"
        if cls.__name__ == "PreprocessorModel":
            replacements["window_length"] = "window_lengths (one value per split)"
        if cls.__name__ in {"InferenceConfig", "ResolvedInferenceConfig"}:
            replacements["preprocessing_data_path"] = "metadata_config_path"
        if cls.__name__ == "SequifierConfig":
            replacements["evaluation"] = "training_plan.phases[].evaluation"
        for key in values:
            if key not in fields and key in replacements:
                replacement = replacements[key]
                target = replacement.split(":")[0].split(".")[0].split()[0]
                if target in fields:
                    raise ValueError(
                        f"{cls.__name__}.{key} was removed; use {replacement}"
                    )
        if cls.__name__ == "TrainingPlanModel" and "phases" not in values:
            raise ValueError("Declare training phases under training_plan.phases")
        if cls.__name__ == "ComponentSpec" and (extra := values.keys() - fields.keys()):
            raise ValueError(
                f"Put constructor arguments {sorted(extra)} under arguments"
            )
        if "_project_root" in cls.__private_attributes__ and "project_root" in values:
            raise ValueError(
                "Source project_root was removed; use the run's project_root"
            )
        return values


def resolve_run_seed(value: Any) -> int:
    if value == "random":
        return secrets.randbelow(1_000_000_000)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer or 'random'")
    return value


RunSeed = Annotated[int, BeforeValidator(resolve_run_seed)]
