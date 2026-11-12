"""Assert authored interface facts against frozen artifact/source metadata."""

from pathlib import Path


def validate_input_contracts(config, resolved):
    if config.model is None:
        return
    from sequifier.config.metadata import load_dataset_metadata
    from sequifier.config.train_config import _assert_compatible, _part_signature

    for name, spec in config.model.interfaces.items():
        declaration = spec.input_contract
        datasets = [
            value
            for value in resolved.dataset_training.values()
            if value.model_interface == name
        ]
        if not datasets:
            continue
        if declaration.metadata_config_path:
            expected = load_dataset_metadata(
                str(Path(config.run.project_root) / declaration.metadata_config_path)
            )
            for dataset in datasets:
                for part in dataset.parts.values():
                    _assert_compatible(
                        _part_signature(expected, config.model.interfaces[name]),
                        _part_signature(part.metadata, config.model.interfaces[name]),
                        f"model.interfaces.{name}.input_contract",
                    )
        normalization = declaration.normalization
        if normalization is None:
            continue
        resolved.normalization_by_interface[name] = normalization
        for dataset in datasets:
            interface = dataset.interface
            expected = normalization.statistics
            if normalization.source == "identity":
                expected = {
                    column: {"mean": 0.0, "std": 1.0}
                    for column in interface.input_columns + interface.static_columns
                    if column in interface.real_columns
                }
            for column, values in expected.items():
                if column not in interface.real_columns:
                    raise ValueError(
                        f"{name}: normalization references unknown/non-real input {column!r}"
                    )
                actual = interface.selected_columns_statistics.get(column)
                if not interface.normalize_real_columns:
                    actual = {"mean": 0.0, "std": 1.0}
                if actual != values:
                    raise ValueError(
                        f"model.interfaces.{name}.input_contract.normalization.{column} differs from frozen metadata/artifact"
                    )
