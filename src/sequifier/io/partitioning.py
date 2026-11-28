"""Leakage-safe connected components and deterministic held-out partitions."""

import hashlib
import heapq
import random
from collections import Counter

import polars as pl

from sequifier.helpers import assign_sequence_to_split


def split_components(records, leakage_keys, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    return partition_components(
        records, leakage_keys, [1 - validation_fraction, validation_fraction], seed
    )


def partition_components(records, leakage_keys, fractions, seed):
    records = list(records)
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, record in enumerate(records):
        for key in leakage_keys(record):
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(root(index), []).append(record)
    components = list(groups.values())
    if len(components) < len(fractions):
        raise ValueError(
            "splitting requires at least two independent components"
            if len(fractions) == 2
            else "splitting requires an independent component per split"
        )
    random.Random(seed).shuffle(components)
    total = len(components)
    partitions = []
    for index in reversed(range(1, len(fractions))):
        count = min(len(components) - index, max(1, int(total * fractions[index])))
        partitions.append(components[:count])
        components = components[count:]
    partitions.append(components)
    return tuple(
        [item for group in groups for item in group] for groups in reversed(partitions)
    )


def table_content_keys(table):
    """Identify equal recorded sequences independently of their IDs and shard order."""
    fields = sorted(
        set(table.collect_schema().names())
        - {"sequenceId", "sequenceName", "groupId", "__collection"}
    )
    # itemPosition (and any depth coordinate) is part of each row. Summing
    # independent row hashes makes this a bounded, order-independent reduction.
    if not fields:
        return {
            seq: () for seq in table.select("sequenceId").unique().collect().to_series()
        }
    row = pl.struct(fields)
    summaries = (
        table.group_by("sequenceId")
        .agg(
            pl.len().alias("length"),
            row.hash(seed=0).sum().alias("first"),
            row.hash(seed=1).sum().alias("second"),
        )
        .collect(engine="streaming")
    )
    schema_key = tuple(fields)
    return {
        seq: (schema_key, length, first, second)
        for seq, length, first, second in summaries.iter_rows()
    }


def sequence_assignments(sequence_ids, group_ids, content_keys, fractions, seed):
    """Assign the same population before fitting features in any learning mode."""
    groups = {
        seq: [("group", group)] for seq, group in group_ids.items() if group is not None
    }
    overrides = sequence_split_overrides(sequence_ids, groups, fractions, seed)
    assignments = {
        seq: overrides[seq]
        if seq in overrides
        else assign_sequence_to_split(seq, fractions, seed)
        for seq in sequence_ids
    }
    owners = {}
    for seq, content in content_keys.items():
        if owners.setdefault(content, assignments[seq]) != assignments[seq]:
            for member, key in content_keys.items():
                groups.setdefault(member, []).append(("content", key))
            return sequence_split_overrides(sequence_ids, groups, fractions, seed)
    return assignments


def sequence_split_overrides(sequence_ids, leakage_keys, fractions, seed):
    """Use supervised ID hashing unless group constraints require a component split."""
    if leakage_keys:
        partitions = partition_components(
            sorted(sequence_ids), lambda seq: leakage_keys.get(seq, ()), fractions, seed
        )
        return {seq: split for split, ids in enumerate(partitions) for seq in ids}

    # Keep only overrides needed to give every named split a sequence.
    unique_sequence_ids = sorted(set(sequence_ids))
    if len(unique_sequence_ids) < len(fractions):
        raise ValueError(
            "between_sequence splitting needs at least as many sequences as requested "
            f"splits; found {len(unique_sequence_ids)} sequences for "
            f"{len(fractions)} splits."
        )

    counts: Counter[int] = Counter()
    candidates: dict[int, list[tuple[int, int]]] = {
        split: [] for split in range(len(fractions))
    }
    candidate_limit = len(fractions)
    for sequence_id in unique_sequence_ids:
        split = assign_sequence_to_split(sequence_id, fractions, seed)
        counts[split] += 1
        digest_value = int.from_bytes(
            hashlib.sha256(f"{seed}:{sequence_id}".encode("utf-8")).digest(),
            byteorder="big",
            signed=False,
        )
        candidate = (digest_value, sequence_id)
        if len(candidates[split]) < candidate_limit:
            heapq.heappush(candidates[split], candidate)
        elif candidate > candidates[split][0]:
            heapq.heapreplace(candidates[split], candidate)

    missing_splits = [split for split in range(len(fractions)) if counts[split] == 0]
    overrides = {}
    for missing_split in missing_splits:
        donor_split = max(
            (split for split, count in counts.items() if count > 1),
            key=lambda split: (counts[split], fractions[split], -split),
        )
        _, sequence_to_move = heapq.nlargest(1, candidates[donor_split])[0]
        candidates[donor_split] = [
            candidate
            for candidate in candidates[donor_split]
            if candidate[1] != sequence_to_move
        ]
        heapq.heapify(candidates[donor_split])
        overrides[sequence_to_move] = missing_split
        counts[donor_split] -= 1
        counts[missing_split] += 1

    return overrides
