from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SpecialTokenIds:
    unknown: int = 0
    other: int = 1
    mask: int = 2

    @property
    def user_start(self) -> int:
        return max(asdict(self).values()) + 1

    @property
    def labels_by_id(self) -> dict[int, str]:
        return {
            self.unknown: "[unknown]",
            self.other: "[other]",
            self.mask: "[mask]",
        }

    @property
    def ids_by_label(self) -> dict[str, int]:
        return {label: id_ for id_, label in self.labels_by_id.items()}


SPECIAL_TOKEN_IDS = SpecialTokenIds()
SPECIAL_TOKEN_LABELS = frozenset(SPECIAL_TOKEN_IDS.ids_by_label.keys())
SPECIAL_TOKEN_ID_VALUES = frozenset(SPECIAL_TOKEN_IDS.labels_by_id.keys())


def validate_special_token_ids(
    special_token_ids: Mapping[str, Any] | None,
    source: str = "metadata",
) -> dict[str, int]:
    """Validate persisted special token IDs against runtime constants."""
    expected = SPECIAL_TOKEN_IDS.ids_by_label
    assert special_token_ids is not None
    try:
        normalized = {str(label): int(id_) for label, id_ in special_token_ids.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} special_token_ids must be a mapping from token labels to integer IDs."
        ) from exc

    if normalized != expected:
        raise ValueError(
            f"{source} special_token_ids must match the sequifier runtime constants. "
            f"Expected {expected}, found {dict(special_token_ids)}."
        )

    return normalized
