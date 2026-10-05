"""Versioned canonical-ID hashing and candidate resolution."""

import math

import torch
from torch import Tensor, nn

HASH_CODEC_VERSION = 1
HASH_PRIME = 2_147_483_647


class HashCodeCollision(ValueError):
    def __init__(self, first_id: int, second_id: int, details: str):
        self.ids = (first_id, second_id)
        super().__init__(details)


def affine_multi_hash_codes(
    ids: Tensor, multipliers: Tensor, offsets: Tensor, num_buckets: int
) -> Tensor:
    ids = ids.to(torch.int64)
    return torch.stack(
        [
            ((ids * multipliers[i] + offsets[i]).remainder(HASH_PRIME)).remainder(
                num_buckets
            )
            for i in range(multipliers.numel())
        ],
        dim=-1,
    )


def multi_hash_codes(
    ids: Tensor, multipliers: Tensor, offsets: Tensor, num_buckets: int
) -> Tensor:
    ids = ids.to(torch.int64)
    codes = affine_multi_hash_codes(ids, multipliers, offsets, num_buckets)
    capacity = num_buckets ** multipliers.numel()
    if capacity <= 3:
        raise ValueError(
            "Multi-hash special-token isolation requires more than three complete codes"
        )

    def reserved_code(value: int) -> Tensor:
        parts = []
        for _ in range(multipliers.numel()):
            parts.append(value % num_buckets)
            value //= num_buckets
        return torch.tensor(
            tuple(reversed(parts)), device=ids.device, dtype=torch.int64
        )

    # Reserve complete codes 0, 1, 2 for special IDs. An ordinary ID whose
    # affine tuple reaches a reserved code uses tuple 3 instead.
    conflicts = torch.zeros_like(ids, dtype=torch.bool)
    for special_id in range(3):
        conflicts = conflicts | (codes == reserved_code(special_id)).all(dim=-1)
    codes = torch.where(conflicts.unsqueeze(-1), reserved_code(3), codes)
    for special_id in range(3):
        codes = torch.where(
            (ids == special_id).unsqueeze(-1), reserved_code(special_id), codes
        )
    return codes


def qr_codes(ids: Tensor, num_buckets: int) -> Tensor:
    ids = ids.to(torch.int64)
    return torch.stack(
        (ids.remainder(num_buckets), ids.div(num_buckets, rounding_mode="floor")),
        dim=-1,
    )


class CategoricalHashCodec(nn.Module):
    def __init__(self, config, n_classes: int, decoder_ids: list[int] | None = None):
        super().__init__()
        self.kind = config.type
        self.num_buckets = config.num_buckets
        self.num_hashes = config.num_hashes
        self.seed = getattr(config, "seed", 0)
        self.version = HASH_CODEC_VERSION
        self.widths = (
            [config.num_buckets] * config.num_hashes
            if config.type == "multi_hash"
            else [config.num_buckets, max(1, math.ceil(n_classes / config.num_buckets))]
        )
        if self.kind == "multi_hash":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            self.register_buffer(
                "multipliers",
                torch.randint(
                    1,
                    HASH_PRIME,
                    (self.num_hashes,),
                    generator=generator,
                    dtype=torch.int64,
                ),
            )
            self.register_buffer(
                "offsets",
                torch.randint(
                    0,
                    HASH_PRIME,
                    (self.num_hashes,),
                    generator=generator,
                    dtype=torch.int64,
                ),
            )
        else:
            self.register_buffer("multipliers", torch.empty(0, dtype=torch.int64))
            self.register_buffer("offsets", torch.empty(0, dtype=torch.int64))
        all_codes = self.encode(torch.arange(n_classes, dtype=torch.int64))
        special_count = min(3, n_classes)
        special_codes = {
            tuple(row): special_id
            for special_id, row in enumerate(all_codes[:special_count].tolist())
        }
        for global_id, row in enumerate(
            all_codes[special_count:].tolist(), special_count
        ):
            if tuple(row) in special_codes:
                raise HashCodeCollision(
                    special_codes[tuple(row)],
                    global_id,
                    f"Hash code for canonical ID {global_id} collides with a special token; "
                    f"num_buckets={self.num_buckets}, num_hashes={self.num_hashes}, seed={self.seed}.",
                )
        if decoder_ids is not None:
            codebook = all_codes[decoder_ids]
            seen: dict[tuple[int, ...], int] = {}
            for global_id, row in zip(decoder_ids, codebook.tolist()):
                code = tuple(row)
                if code in seen:
                    raise HashCodeCollision(
                        seen[code],
                        global_id,
                        f"Hash target collision between canonical IDs {seen[code]} and {global_id}; "
                        f"num_buckets={self.num_buckets}, num_hashes={self.num_hashes}, seed={self.seed}.",
                    )
                seen[code] = global_id
            self.register_buffer("codebook", codebook)
            self.register_buffer(
                "decoder_ids", torch.tensor(decoder_ids, dtype=torch.int64)
            )
        else:
            self.register_buffer(
                "codebook", torch.empty((0, self.num_hashes), dtype=torch.int64)
            )
            self.register_buffer("decoder_ids", torch.empty(0, dtype=torch.int64))

    def encode(self, ids: Tensor) -> Tensor:
        ids = ids.to(torch.int64)
        if self.kind == "qr":
            return qr_codes(ids, self.num_buckets)
        return multi_hash_codes(ids, self.multipliers, self.offsets, self.num_buckets)

    def resolve(self, components: tuple[Tensor, ...]) -> Tensor:
        scores = None
        for index, logits in enumerate(components):
            selected = torch.log_softmax(logits.float(), dim=-1).index_select(
                -1, self.codebook[:, index]
            )
            scores = selected if scores is None else scores + selected
        if scores is None:
            raise RuntimeError("A hash resolver requires component logits")
        return scores

    def contract(self) -> dict:
        return {
            "version": self.version,
            "type": self.kind,
            "num_buckets": self.num_buckets,
            "num_hashes": self.num_hashes,
            "seed": self.seed,
            "widths": self.widths,
            "multipliers": self.multipliers.tolist(),
            "offsets": self.offsets.tolist(),
            "codebook": self.codebook.tolist(),
            "decoder_ids": self.decoder_ids.tolist(),
            "special_token_policy": "reserve_complete_codes_0_1_2_v1"
            if self.kind == "multi_hash"
            else "qr_injective_v1",
        }
