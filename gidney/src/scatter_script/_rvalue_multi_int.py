from __future__ import annotations

from typing import Sequence, Iterable, TYPE_CHECKING

import numpy as np

from scatter_script._util import slice_int

if TYPE_CHECKING:
    from scatter_script._quint import quint


class rvalue_multi_int:
    """Represents intermediate values computed from superposed integers."""

    def __init__(self, branch_vals: Iterable[int]):
        self.UNPHYSICAL_branch_vals = tuple(branch_vals)

    @staticmethod
    def from_value(
        other: int | quint | rvalue_multi_int, *, expected_count: int
    ) -> rvalue_multi_int:
        if isinstance(other, int):
            return rvalue_multi_int((other,) * expected_count)
        if isinstance(other, rvalue_multi_int):
            return other

        from scatter_script._quint import quint

        if isinstance(other, quint):
            return other.UNPHYSICAL_copy()

        if isinstance(
            other,
            (np.uint32, np.int32, np.uint64, np.int64, np.uint16, np.int16, np.uint8, np.int8),
        ):
            return rvalue_multi_int((int(other),) * expected_count)
        raise NotImplementedError(f"Don't know how to wrap {type(other)=} {other=}.")

    def __getitem__(self, item: int | slice) -> rvalue_multi_int:
        return rvalue_multi_int(slice_int(a, item) for a in self.UNPHYSICAL_branch_vals)

    @staticmethod
    def from_stacking(q_vals: Sequence[quint]) -> rvalue_multi_int:
        offsets = [0]
        for k in range(len(q_vals))[::-1]:
            offsets.append(offsets[-1] + len(q_vals[k]))
        offsets.pop()
        offsets = offsets[::-1]
        r_vals = [q_v.UNPHYSICAL_branch_vals for q_v in q_vals]
        return rvalue_multi_int(
            sum(r_vals[k2][k] << offsets[k2] for k2 in range(len(r_vals)))
            for k in range(len(q_vals[0]._buffer))
        )

    def __int__(self) -> int:
        unique_values = set(self.UNPHYSICAL_branch_vals)
        if len(unique_values) == 1:
            return next(iter(unique_values))
        raise NotImplementedError(f"Can't convert into a single int because {unique_values=}")

    def UNPHYSICAL_is_nonzero_in_any_branch(self) -> bool:
        return any(self.UNPHYSICAL_branch_vals)

    def _wrap(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int.from_value(other, expected_count=len(self.UNPHYSICAL_branch_vals))

    def __bool__(self) -> bool:
        unique_values = set(bool(e) for e in self.UNPHYSICAL_branch_vals)
        if len(unique_values) == 1:
            return next(iter(unique_values))
        raise NotImplementedError(f"Can't convert into a single bool because {unique_values=}")

    def __add__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a + b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __sub__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a - b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __rsub__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            b - a
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __and__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a & b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __or__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a | b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __xor__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a ^ b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    __rxor__ = __xor__

    def UNPHYSICAL_copy(self) -> rvalue_multi_int:
        return self

    def __mul__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a * b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __pow__(self, power: int | quint | rvalue_multi_int, modulo=None) -> rvalue_multi_int:
        return rvalue_multi_int(
            pow(a, b, modulo)
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(power).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __rpow__(self, other: int | quint | rvalue_multi_int, modulo=None) -> rvalue_multi_int:
        return rvalue_multi_int(
            pow(a, b, modulo)
            for a, b in zip(
                self._wrap(other).UNPHYSICAL_branch_vals, self.UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __mod__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a % b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    __rmul__ = __mul__
    __radd__ = __mul__

    def __lshift__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a << b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __rshift__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a >> b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __rmod__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            b % a
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __neg__(self) -> rvalue_multi_int:
        return rvalue_multi_int(-a for a in self.UNPHYSICAL_branch_vals)

    def __invert__(self) -> rvalue_multi_int:
        return rvalue_multi_int(~a for a in self.UNPHYSICAL_branch_vals)

    def __gt__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a > b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __lt__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a < b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __ge__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a >= b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __le__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a <= b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __eq__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a == b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __ne__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int(
            a != b
            for a, b in zip(
                self.UNPHYSICAL_branch_vals, self._wrap(other).UNPHYSICAL_branch_vals, strict=True
            )
        )

    def __repr__(self):
        return f"rvalue_multi_int({self.UNPHYSICAL_branch_vals!r})"
