"""Leakage-safe connected components and deterministic held-out partitions."""

import random


def split_components(records, leakage_keys, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
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
    if len(components) < 2:
        raise ValueError("splitting requires at least two independent components")
    random.Random(seed).shuffle(components)
    count = min(len(components) - 1, max(1, int(len(components) * validation_fraction)))
    return tuple(
        [record for group in groups for record in group]
        for groups in (components[count:], components[:count])
    )
