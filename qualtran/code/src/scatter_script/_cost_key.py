from __future__ import annotations

from typing import Any


class CostKey:
    def __init__(self, name: str, params: dict[str, int] = None):
        self.name = name
        self.params = params or {}
        self._hash = hash((self.name, frozenset(self.params.items())))

    def __eq__(self, other: CostKey | Any) -> bool:
        if isinstance(other, CostKey):
            return self.name == other.name and self.params == other.params
        return NotImplemented

    def __lt__(self, other: CostKey) -> bool:
        if isinstance(other, CostKey):
            if self.name != other.name:
                return self.name < other.name
            for k in sorted(self.params.keys()):
                if k in other.params and other.params[k] != self.params[k]:
                    return self.params[k] < other.params[k]
            return False

        return NotImplemented

    def __getitem__(self, item):
        return self.params[item]

    def get(self, item, default: int) -> int:
        return self.params.get(item, default)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"CostKey({self.name!r}, {self.params!r})"

    def __str__(self) -> str:
        s = f"{self.name}"
        if self.params:
            s += "["
            s += ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
            s += "]"
        return s
