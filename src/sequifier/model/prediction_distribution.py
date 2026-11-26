"""Scalar prediction distributions, independent of the learning objective."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class _LogScales(nn.ParameterDict):
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in self:
            key = prefix + name
            if key in state_dict and state_dict[key].ndim == 0:
                state_dict[key] = state_dict[key].reshape(1)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PredictionScales(nn.Module):
    """Gaussian scales owned and persisted by the model interface."""

    def __init__(self, fields):
        super().__init__()
        self.fields = dict(fields)
        self.log_scales = _LogScales(
            {
                name: nn.Parameter(torch.tensor([math.log(field.scale.initial)]))
                for name, field in sorted(fields.items())
                if field.scale.trainable
            }
        )

    def ensure_fields(self, fields):
        for name, field in sorted(fields.items()):
            if field.scale.trainable and name not in self.log_scales:
                self.log_scales[name] = nn.Parameter(
                    torch.tensor([math.log(field.scale.initial)])
                )
        self.fields.update(fields)

    def scale(self, name):
        if name not in self.fields:
            raise ValueError(
                f"{name}: scale: model requires a prediction_distributions entry"
            )
        spec = self.fields[name].scale
        if not spec.trainable:
            return spec.initial
        return (
            self.log_scales[name]
            .reshape(())
            .clamp(math.log(spec.min), math.log(spec.max))
            .exp()
        )


class RealPredictionDistribution:
    """A Gaussian, optionally mapped bijectively onto an open real domain.

    Values and densities use physical coordinates. Explicit point masses can
    represent endpoints; continuous samples retain transform latents for replay.
    """

    def __init__(self, location, scale, field=None, *, validate=True):
        self.dtype = torch.float32 if location.device.type == "mps" else torch.float64
        self.location = location.to(self.dtype)
        self.scale = torch.as_tensor(scale, device=location.device).to(self.dtype)
        self.field = (
            field
            if field is not None
            else SimpleNamespace(lower=None, upper=None, atoms={})
        )
        if validate and (
            not torch.isfinite(self.location).all()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError(
                "real prediction locations and positive scales must be finite"
            )
        self.normal = torch.distributions.Normal(
            self.location, self.scale, validate_args=None if validate else False
        )

    def _transform(self, latent):
        lower, upper = self.field.lower, self.field.upper
        if lower is not None and upper is not None:
            return lower + (upper - lower) * torch.sigmoid(latent)
        if lower is not None:
            return lower + F.softplus(latent)
        if upper is not None:
            return upper - F.softplus(latent)
        return latent

    def mode(self):
        # Deterministic location; a transformed location need not be its mode.
        return self._transform(self.location)

    @torch.no_grad()
    def sample(self, generator=None):
        """Sample a tensor, optionally with an independent generator per batch row.

        Only random draws are split by environment. Transforms and mixture
        selection remain batched, and the scalar-generator path is unchanged.
        """
        if isinstance(generator, (list, tuple)):
            if self.location.ndim == 0 or len(generator) != self.location.shape[0]:
                raise ValueError("one action RNG is required per leading batch row")
            if any(not isinstance(rng, torch.Generator) for rng in generator):
                raise TypeError("batched action RNGs must be torch.Generator instances")

        def draw(random, **options):
            if isinstance(generator, (list, tuple)):
                if not generator:
                    return torch.empty(
                        self.location.shape, device=self.location.device, **options
                    )
                return torch.cat(
                    [
                        random(
                            self.location[index : index + 1].shape,
                            device=rng.device,
                            generator=rng,
                            **options,
                        ).to(self.location.device)
                        for index, rng in enumerate(generator)
                    ]
                )
            device = generator.device if generator is not None else self.location.device
            return random(
                self.location.shape, device=device, generator=generator, **options
            ).to(self.location.device)

        noise = draw(torch.randn, dtype=self.dtype)
        choice = draw(torch.rand) if self.field.atoms else None
        value, self.sampled_latent, self.sampled_component = self.select_from_noise(
            noise, choice
        )
        return value

    def select_from_noise(self, noise, choice):
        """Apply the Gaussian transform and choose among any configured point masses."""
        latent = self.location + self.scale * noise
        value = self._transform(latent)
        component = torch.full_like(value, -1, dtype=torch.long)
        if self.field.atoms:
            cumulative = 0.0
            for index, (atom, probability) in enumerate(
                sorted(self.field.atoms.items())
            ):
                selected = (choice >= cumulative) & (choice < cumulative + probability)
                value = torch.where(selected, value.new_tensor(atom), value)
                component = torch.where(
                    selected,
                    component.new_tensor(index),
                    component,
                )
                cumulative += probability
        return value, latent, component

    def log_prob(self, value, latent=None, component=None):
        value = value.to(self.dtype)
        if not torch.isfinite(value).all():
            raise ValueError("real actions must be finite")
        atoms = torch.zeros_like(value, dtype=torch.bool)
        atom_score = torch.zeros_like(value)
        if component is not None and (
            component.shape != value.shape
            or (component < -2).any()
            or (component >= len(self.field.atoms)).any()
        ):
            raise ValueError("invalid real mixture component")
        for index, (atom, probability) in enumerate(sorted(self.field.atoms.items())):
            selected = value == atom
            if component is not None:
                if ((component == index) & ~selected).any():
                    raise ValueError(
                        "recorded mixture component differs from its action"
                    )
                selected &= (component == -2) | (component == index)
            atoms |= selected
            atom_score = torch.where(
                selected, value.new_tensor(math.log(probability)), atom_score
            )
        supplied = torch.zeros_like(atoms) if latent is None else ~torch.isnan(latent)
        if latent is not None:
            latent = latent.to(self.dtype)
            if (
                latent.shape != value.shape
                or not torch.isfinite(latent[supplied]).all()
            ):
                raise ValueError(
                    "real transform latents must be finite and match actions"
                )
            comparable = supplied & ~atoms
            # Portable FP64 actions can differ by a few last bits between
            # ORT and native sigmoid/softplus implementations.
            transformed = self.field.lower is not None or self.field.upper is not None
            tolerance = (
                8 * torch.finfo(self.dtype).eps
                if transformed and self.dtype == torch.float64
                else 0.0
            )
            if not torch.allclose(
                self._transform(latent[comparable]),
                value[comparable],
                rtol=tolerance,
                atol=tolerance,
            ):
                raise ValueError(
                    "recorded real latent does not reconstruct its executed action"
                )
        lower, upper = self.field.lower, self.field.upper
        invert = ~supplied & ~atoms
        if lower is not None and ((value < lower) | (invert & (value == lower))).any():
            raise ValueError(
                "real endpoints require a declared atom or a recorded transform latent"
            )
        if upper is not None and ((value > upper) | (invert & (value == upper))).any():
            raise ValueError(
                "real endpoints require a declared atom or a recorded transform latent"
            )
        safe = torch.where(invert, value, self._transform(torch.zeros_like(value)))
        if lower is not None and upper is not None:
            fraction = (safe - lower) / (upper - lower)
            inverse = fraction.log() - torch.log1p(-fraction)
        elif lower is not None or upper is not None:
            if lower is None:
                assert upper is not None
                distance = upper - safe
            else:
                distance = safe - lower
            inverse = distance + torch.log(-torch.expm1(-distance))
        else:
            inverse = safe
        z = torch.where(supplied, latent, inverse) if latent is not None else inverse
        z = torch.where(atoms, torch.zeros_like(z), z)
        return torch.where(atoms, atom_score, self.continuous_log_prob(z))

    def continuous_log_prob(self, z):
        lower, upper = self.field.lower, self.field.upper
        if lower is not None and upper is not None:
            jacobian = math.log(upper - lower) + F.logsigmoid(z) + F.logsigmoid(-z)
        elif lower is not None or upper is not None:
            jacobian = F.logsigmoid(z)
        else:
            jacobian = 0.0
        return (
            self.normal.log_prob(z)
            - jacobian
            + math.log1p(-sum(self.field.atoms.values()))
        )

    def entropy(self):
        if (
            self.field.lower is not None
            or self.field.upper is not None
            or self.field.atoms
        ):
            raise ValueError("transformed entropy requires a sampling estimator")
        return self.normal.entropy()
