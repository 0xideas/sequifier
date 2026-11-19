"""Internal policy feature layout shared by fitting and model attachment."""

from dataclasses import dataclass, replace

from sequifier.config.policy_config import PolicyBinding, ResolvedPolicyBinding


@dataclass(frozen=True, kw_only=True)
class PolicyInputLayout:
    observation_columns: list[str]
    static_columns: list[str]
    categorical_columns: list[str]
    observation_n_classes: dict[str, int]
    static_n_classes: dict[str, int]
    id_maps: dict[str, dict[str | int, int]]

    @classmethod
    def select(
        cls,
        temporal,
        static,
        binding,
        categorical,
        n_classes,
        id_maps,
        *,
        features_only=False,
    ):
        observations = [
            name
            for name in temporal
            if name not in binding.previous_action_inputs
            and (name != "__policy_return" or not binding.return_conditioned)
        ]
        vocabulary_columns = observations if features_only else temporal
        columns = set(vocabulary_columns) | set(static)
        return cls(
            observation_columns=observations,
            static_columns=list(static),
            categorical_columns=list(categorical),
            observation_n_classes={
                k: n for k, n in n_classes.items() if k in vocabulary_columns
            },
            static_n_classes={k: n for k, n in n_classes.items() if k in static},
            id_maps={k: dict(v) for k, v in id_maps.items() if k in columns},
        )

    @classmethod
    def for_fitting(cls, temporal, static, binding, metadata):
        layout = cls.select(
            temporal,
            static,
            binding,
            [],
            metadata.n_classes,
            metadata.id_maps,
            features_only=True,
        )
        return replace(
            layout,
            categorical_columns=[
                name
                for name in layout.observation_columns + layout.static_columns
                if "int" in metadata.column_data_types[name].lower()
            ],
        )

    @property
    def categorical_n_classes(self):
        return {**self.observation_n_classes, **self.static_n_classes}

    def binding_values(self):
        return {
            "observation_columns": self.observation_columns,
            "static_columns": self.static_columns,
            "categorical_columns": self.categorical_columns,
            "observation_n_classes": self.observation_n_classes,
            "static_n_classes": self.static_n_classes,
            "id_maps": self.id_maps,
        }


@dataclass(frozen=True, kw_only=True)
class PolicyFittingSpec(PolicyInputLayout):
    """Raw episode features before an executable model schema exists."""

    binding: PolicyBinding
    context_length: int
    interface_name: str

    @property
    def actions(self):
        return self.binding.actions

    @property
    def action_columns(self):
        return list(self.actions)

    @property
    def return_conditioned(self):
        return self.binding.return_conditioned

    @property
    def returns(self):
        return self.binding.returns

    @property
    def input_encoding(self):
        return "raw"


PolicyEpisodeSpec = ResolvedPolicyBinding | PolicyFittingSpec
