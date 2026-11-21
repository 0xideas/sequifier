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
        resolved.interface(name).validate_normalization(normalization)
