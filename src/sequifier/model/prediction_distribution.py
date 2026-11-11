"""Scalar prediction distributions, independent of the learning objective."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class PredictionScales(nn.Module):
    """Gaussian scales owned and persisted by the model interface."""

    def __init__(self, fields):
        super().__init__()
        self.fields = dict(fields)
        self.log_scales = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(math.log(field.scale.initial)))
                for name, field in sorted(fields.items())
                if field.scale.trainable
            }
        )

    def ensure_fields(self, fields):
        for name, field in sorted(fields.items()):
            if field.scale.trainable and name not in self.log_scales:
                self.log_scales[name] = nn.Parameter(
                    torch.tensor(math.log(field.scale.initial))
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
        return self.log_scales[name].clamp(math.log(spec.min), math.log(spec.max)).exp()


class RealPredictionDistribution:
    """A Gaussian, optionally mapped bijectively onto an open real domain.

    Values and densities use physical coordinates. Explicit point masses can
    represent endpoints; continuous samples retain transform latents for replay.
    """

    def __init__(self, location, scale, field=None):
        self.dtype = torch.float32 if location.device.type == "mps" else torch.float64
        self.location = location.to(self.dtype)
        self.scale = torch.as_tensor(scale, device=location.device).to(self.dtype)
        self.field = (
            field
            if field is not None
            else SimpleNamespace(lower=None, upper=None, atoms={})
        )
        if (
            not torch.isfinite(self.location).all()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError(
                "real prediction locations and positive scales must be finite"
            )
        self.normal = torch.distributions.Normal(self.location, self.scale)

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
        device = generator.device if generator is not None else self.location.device
        noise = torch.randn(
            self.location.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        ).to(self.location.device)
        self.sampled_latent = self.location + self.scale * noise
        value = self._transform(self.sampled_latent)
        self.sampled_component = torch.full_like(value, -1, dtype=torch.long)
        if self.field.atoms:
            choice = torch.rand(
                self.location.shape, device=device, generator=generator
            ).to(value.device)
            cumulative = 0.0
            for index, (atom, probability) in enumerate(
                sorted(self.field.atoms.items())
            ):
                selected = (choice >= cumulative) & (choice < cumulative + probability)
                value = torch.where(selected, value.new_tensor(atom), value)
                self.sampled_component = torch.where(
                    selected,
                    self.sampled_component.new_tensor(index),
                    self.sampled_component,
                )
                cumulative += probability
        return value

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
            if not torch.equal(self._transform(latent[comparable]), value[comparable]):
                raise ValueError(
                    "recorded real latent does not reconstruct its executed action"
                )
        lower, upper = self.field.lower, self.field.upper
        invert = ~supplied & ~atoms
        if lower is not None and (value[invert] <= lower).any():
            raise ValueError(
                "real endpoints require a declared atom or a recorded transform latent"
            )
        if upper is not None and (value[invert] >= upper).any():
            raise ValueError(
                "real endpoints require a declared atom or a recorded transform latent"
            )
        safe = torch.where(invert, value, self._transform(torch.zeros_like(value)))
        if lower is not None and upper is not None:
            fraction = (safe - lower) / (upper - lower)
            inverse = fraction.log() - torch.log1p(-fraction)
        elif lower is not None or upper is not None:
            distance = safe - lower if lower is not None else upper - safe
            inverse = distance + torch.log(-torch.expm1(-distance))
        else:
            inverse = safe
        z = torch.where(supplied, latent, inverse) if latent is not None else inverse
        z = torch.where(atoms, torch.zeros_like(z), z)
        if lower is not None and upper is not None:
            jacobian = math.log(upper - lower) + F.logsigmoid(z) + F.logsigmoid(-z)
        elif lower is not None or upper is not None:
            jacobian = F.logsigmoid(z)
        else:
            jacobian = 0.0
        score = (
            self.normal.log_prob(z)
            - jacobian
            + math.log1p(-sum(self.field.atoms.values()))
        )
        return torch.where(atoms, atom_score, score)

    def entropy(self):
        if (
            self.field.lower is not None
            or self.field.upper is not None
            or self.field.atoms
        ):
            raise ValueError("transformed entropy requires a sampling estimator")
        return self.normal.entropy()
