"""Assert authored interface facts against frozen artifact/source metadata."""


def validate_input_contracts(config, resolved):
    if config.model is None:
        return
    for name, spec in config.model.interfaces.items():
        declaration = spec.input_contract
        if getattr(declaration, "cardinality_config", None) is not None:
            storage = resolved.execution.interface_storage.get(name)
            interface = resolved.interface(name)
            recorded = (
                storage.raw_input_contract(interface.column_data_types)
                if storage
                else {}
            )
            if declaration.cardinality_rules != recorded.get("cardinality_config", {}):
                raise ValueError(
                    f"{name}: cardinality_config differs from frozen input metadata"
                )
        normalization = declaration.normalization
        if normalization is None:
            continue
        resolved.normalization_by_interface[name] = normalization
        resolved.interface(name).validate_normalization(normalization)
