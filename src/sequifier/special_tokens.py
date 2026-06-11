from dataclasses import asdict, dataclass


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
