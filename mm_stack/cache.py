from __future__ import annotations

from collections import OrderedDict


class QueryEmbeddingCache:
    def __init__(self, max_size: int = 128):
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict()

    def get(self, key: tuple[str, str, str]) -> list[float] | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: tuple[str, str, str], value: list[float]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
