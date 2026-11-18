"""Count-bounded LRU storage for decoded episodes and decision labels."""

from collections import OrderedDict

from sequifier.io.memory_budget import resident_bytes


class EpisodeCache:
    def __init__(self, capacity, storage=None):
        self.capacity = capacity
        self.storage = storage
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key, load):
        if key not in self.entries:
            value = load(key)
            size = resident_bytes(value) if self.storage is not None else 0
            self.entries[key] = value, size
            self.bytes += size
            if self.storage is not None:
                self.storage.resize(self.bytes)
            if len(self.entries) > self.capacity:
                _, (_, evicted) = self.entries.popitem(last=False)
                self.bytes -= evicted
                if self.storage is not None:
                    self.storage.resize(self.bytes)
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        if self.storage is not None:
            self.storage.resize(0)
