from typing import Any

import torch.distributed as dist


def broadcast_initial_state(
    selected_on_rank_zero: dict[str, Any] | None, rank: int
) -> dict[str, Any]:
    values = [selected_on_rank_zero if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    source = values[0]
    if not isinstance(source, dict):
        raise RuntimeError("Rank 0 did not broadcast an initial state source.")

    identity = (source.get("kind"), source.get("revision_id"), source.get("path"))
    identities: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(identities, identity)
    if any(other != identity for other in identities):
        raise RuntimeError(f"Ranks selected different initial states: {identities}")
    return source


def broadcast_publication_result(
    result_on_rank_zero: dict[str, Any] | None, rank: int
) -> dict[str, Any]:
    values = [result_on_rank_zero if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    result = values[0]
    if not isinstance(result, dict):
        raise RuntimeError("Rank 0 did not broadcast backbone publication status.")
    return result


def verify_loaded_revision(parent_revision_id: str | None) -> None:
    loaded_revisions: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(loaded_revisions, parent_revision_id)
    if any(revision != parent_revision_id for revision in loaded_revisions):
        raise RuntimeError(
            f"Ranks loaded different backbone revisions: {loaded_revisions}"
        )
