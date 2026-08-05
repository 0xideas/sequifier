from torch import Tensor, nn

from sequifier.model.dtypes import cast_floating_to_module_dtype


class SequifierModel(nn.Module):
    """A Sequifier network with three explicit, checkpointable components."""

    ingestion: nn.Module
    ingestion_adapter: nn.Module
    backbone: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        ingestion: nn.Module | None = None,
        backbone: nn.Module | None = None,
        decoder: nn.Module | None = None,
        *,
        decoding_support: int = 1,
    ):
        super().__init__()
        components = (ingestion, backbone, decoder)
        if any(component is not None for component in components) and not all(
            component is not None for component in components
        ):
            raise ValueError(
                "ingestion, backbone, and decoder must be provided together."
            )
        if ingestion is not None and backbone is not None and decoder is not None:
            self.ingestion = ingestion
            self.ingestion_adapter = nn.Identity()
            self.backbone = backbone
            self.decoder = decoder
        if decoding_support <= 0:
            raise ValueError("decoding_support must be positive.")
        self.decoding_support = decoding_support

    def encode(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        hidden = self.ingestion(features, metadata)
        hidden = self.ingestion_adapter(
            cast_floating_to_module_dtype(hidden, self.ingestion_adapter)
        )
        return self.encode_ingested(hidden, metadata, attention_mask)

    def encode_ingested(
        self,
        hidden: Tensor,
        metadata: dict[str, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        expected_width = getattr(self.backbone, "input_dim", self.backbone.dim_model)
        if hidden.shape[-1] != expected_width:
            raise ValueError(
                "Adapted ingestion output width must equal backbone input_dim: "
                f"{hidden.shape[-1]} != {expected_width}."
            )
        valid_mask = metadata["attention_valid_mask"].bool()
        if valid_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"Invalid attention_valid_mask shape {tuple(valid_mask.shape)} for "
                f"ingestion output {tuple(hidden.shape)}."
            )
        hidden = hidden.masked_fill(~valid_mask[:, :, None], 0.0)
        hidden = self.backbone(hidden, attention_mask)
        return hidden.masked_fill(~valid_mask[:, :, None], 0.0)

    def decoder_input(self, hidden: Tensor) -> Tensor:
        """Return batch-first support windows for the target decoder."""
        if self.decoding_support == 1:
            return hidden
        if self.decoding_support > hidden.shape[1]:
            raise ValueError(
                f"decoding_support {self.decoding_support} exceeds sequence length "
                f"{hidden.shape[1]}."
            )
        windows = hidden.unfold(1, self.decoding_support, 1)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows.reshape(
            hidden.shape[0],
            hidden.shape[1] - self.decoding_support + 1,
            self.decoding_support * hidden.shape[2],
        )

    def predict(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.encode(features, metadata, attention_mask)
        return self.decoder(self.decoder_input(hidden))

    def forward(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
