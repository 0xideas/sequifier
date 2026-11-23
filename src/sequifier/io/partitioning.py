"""Leakage-safe connected components and deterministic held-out partitions."""

import random


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
