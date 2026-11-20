"""Assert authored interface facts against frozen artifact/source metadata."""


def validate_input_contracts(config, resolved):
    if config.model is None:
        return
    for name, spec in config.model.interfaces.items():
        declaration = spec.input_contract
        normalization = declaration.normalization
        if normalization is None:
            continue
        resolved.normalization_by_interface[name] = normalization
        interface = resolved.interface(name)
        expected = normalization.statistics
        real_inputs = interface.real_columns + interface.static_real_columns
        if normalization.source == "identity":
            expected = {column: {"mean": 0.0, "std": 1.0} for column in real_inputs}
        for column, values in expected.items():
            if column not in real_inputs:
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
