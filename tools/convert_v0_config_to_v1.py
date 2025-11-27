import argparse

import yaml


# Helper to rename keys while preserving insertion order
def rename_key(d, old, new):
    if old not in d:
        return
    # Create a new dict to preserve order
    new_d = {}
    for k, v in d.items():
        if k == old:
            new_d[new] = v
        else:
            new_d[k] = v
    d.clear()
    d.update(new_d)


def convert_preprocess(config):
    rename_key(config, "project_path", "project_root")

    assert "group_proportions" in config, "group_proportions missing"
    rename_key(config, "group_proportions", "split_ratios")

    if "seq_step_sizes" in config:
        rename_key(config, "seq_step_sizes", "stride_by_split")

    # Rename and validate merge settings
    if "combine_into_single_file" in config:
        config.pop("combine_into_single_file")

    # Enforce new v1.0 logic for merge_output based on format
    write_format = config.get("write_format", "parquet")
    if write_format == "pt":
        config["merge_output"] = False
    else:
        config["merge_output"] = True

    if "seed" not in config:
        config["seed"] = 1010

    return config


def convert_model_spec(ms, is_hp_search=False):
    # Mapping old keys to new keys
    if "d_model" in ms:
        val = ms["d_model"]
        rename_key(ms, "d_model", "dim_model")
        ms["initial_embedding_dim"] = val

        if is_hp_search:
            ms["joint_embedding_dim"] = [None] * len(val)

    if "d_model_by_column" in ms:
        rename_key(ms, "d_model_by_column", "feature_embedding_dims")

    if "nhead" in ms:
        rename_key(ms, "nhead", "n_head")

    if "d_hid" in ms:
        rename_key(ms, "d_hid", "dim_feedforward")

    if "nlayers" in ms:
        rename_key(ms, "nlayers", "num_layers")

    if "inference_size" in ms:
        rename_key(ms, "inference_size", "prediction_length")
    elif is_hp_search:
        ms["prediction_length"] = 1

    # Add v1 defaults
    if is_hp_search:
        ms["activation_fn"] = ["swiglu"]
        ms["normalization"] = ["rmsnorm"]
        ms["positional_encoding"] = ["learned"]
        ms["attention_type"] = ["mha"]
        ms["norm_first"] = [True]
        ms["n_kv_heads"] = [None]
        ms["rope_theta"] = [10000.0]
    else:
        ms["activation_fn"] = "swiglu"
        ms["normalization"] = "rmsnorm"
        ms["positional_encoding"] = "learned"
        ms["attention_type"] = "mha"
        ms["norm_first"] = True
        ms["rope_theta"] = 10000.0

    return ms


def convert_training_spec(ts, is_hp_search=False):
    if "lr" in ts:
        rename_key(ts, "lr", "learning_rate")

    if "iter_save" in ts:
        rename_key(ts, "iter_save", "save_interval_epochs")

    if "scheduler_step_iter" in ts:
        rename_key(ts, "scheduler_step_iter", "scheduler_step_on")
    else:
        ts["scheduler_step_on"] = "epoch"

    if not is_hp_search:
        ts["enforce_determinism"] = False
        ts["max_ram_gb"] = 16

    return ts


def convert_train(config):
    rename_key(config, "project_path", "project_root")
    rename_key(config, "ddconfig_path", "metadata_config_path")
    rename_key(config, "selected_columns", "input_columns")

    if "seed" not in config:
        config["seed"] = 1010

    assert "model_spec" in config, "model_spec missing"
    config["model_spec"] = convert_model_spec(config["model_spec"])

    assert "training_spec" in config, "training_spec missing"
    config["training_spec"] = convert_training_spec(config["training_spec"])

    return config


def convert_infer(config):
    rename_key(config, "project_path", "project_root")
    rename_key(config, "ddconfig_path", "metadata_config_path")
    rename_key(config, "selected_columns", "input_columns")

    if "inference_size" in config:
        rename_key(config, "inference_size", "prediction_length")

    if "seed" not in config:
        config["seed"] = 1010

    config["enforce_determinism"] = False

    return config


def convert_hp_search(config):
    rename_key(config, "project_path", "project_root")
    rename_key(config, "ddconfig_path", "metadata_config_path")
    rename_key(config, "selected_columns", "input_columns")

    # Add new export flags
    config["export_generative_model"] = True
    config["export_embedding_model"] = False
    config["override_input"] = False

    assert (
        "model_hyperparameter_sampling" in config
    ), "model_hyperparameter_sampling missing"
    config["model_hyperparameter_sampling"] = convert_model_spec(
        config["model_hyperparameter_sampling"], is_hp_search=True
    )

    assert (
        "training_hyperparameter_sampling" in config
    ), "training_hyperparameter_sampling missing"
    config["training_hyperparameter_sampling"] = convert_training_spec(
        config["training_hyperparameter_sampling"], is_hp_search=True
    )

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_type", type=str)
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    assert args.config_type in [
        "preprocess",
        "train",
        "infer",
        "hyperparameter-search",
    ], "Invalid config-type"

    with open(args.file_path, "r") as f:
        config = yaml.safe_load(f)

    if args.config_type == "preprocess":
        new_config = convert_preprocess(config)
    elif args.config_type == "train":
        new_config = convert_train(config)
    elif args.config_type == "infer":
        new_config = convert_infer(config)
    elif args.config_type == "hyperparameter-search":
        new_config = convert_hp_search(config)
    else:
        new_config = {}

    with open(args.file_path, "w") as f:
        f.write(yaml.dump(new_config, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
