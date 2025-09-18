from __future__ import annotations

from typing import Sequence

import numpy as np

from scatter_script._quint import quint
from scatter_script._rvalue_multi_int import rvalue_multi_int
from scatter_script._util import phase_correction_table_for_qrom_uncompute, ZeroArray, ConstArray


class LookupCmp:
    def __init__(self, a: quint | Lookup, b: quint | Lookup, cmp: str):
        self.a = a
        self.b = b
        self.cmp = cmp


class Lookup:
    def __init__(
        self,
        table: Sequence[int] | np.ndarray,
        *,
        index: quint | rvalue_multi_int | int | None = None,
        vent: np.ndarray | quint | Lookup | None = None,
    ):
        self.table = table
        self.index = index
        self.vent = vent

    def to_rval(self) -> rvalue_multi_int:
        assert self.index is not None
        return rvalue_multi_int(self.table[v] for v in self.index.peek())

    def phase_corrections_for_mx(self, mx: int) -> np.ndarray:
        return phase_correction_table_for_qrom_uncompute(table=self.table, mx=mx)

    def perform_venting(self, phases: np.ndarray) -> None:
        if isinstance(self.vent, quint):
            assert len(self.vent) == 1 and len(phases) == 2
            if phases[0] ^ phases[1]:
                self.vent.z()
            if phases[0]:
                self.vent._parent.global_phase_flip()
        elif isinstance(self.vent, Lookup):
            if isinstance(phases, ZeroArray):
                pass
            elif isinstance(phases, ConstArray):
                self.vent.table ^= phases.val
            else:
                self.vent.table ^= phases
        elif isinstance(self.vent, np.ndarray):
            if isinstance(phases, ZeroArray):
                pass
            elif isinstance(phases, ConstArray):
                self.vent ^= phases.val
            else:
                self.vent ^= phases
        else:
            raise NotImplementedError(f"{self.vent=}")

    def __len__(self) -> int:
        return len(self.table)

    def phase_flip_lookup(self) -> None:
        assert isinstance(self.index, quint)
        self.index._parent.phase_flip_by_lookup(Q_address=self.index, table=self.table)

    def venting_into(self, vent: np.ndarray | quint | Lookup) -> Lookup:
        return Lookup(self.table, index=self.index, vent=vent)

    def venting_into_new_table(self) -> Lookup:
        return self.venting_into(Lookup(np.zeros(self.shape, dtype=np.bool_)))

    @property
    def shape(self) -> tuple[int, ...]:
        if hasattr(self.table, "shape"):
            return self.table.shape
        return (len(self),)

    def __getitem__(self, item) -> Lookup | int:
        assert self.index is None
        if isinstance(item, int):
            result = self.table[item]
            if isinstance(result, int) or result.shape == ():
                return result
            return Lookup(
                result, index=self.index, vent=None if self.vent is None else self.vent[item]
            )
        if isinstance(item, (quint, rvalue_multi_int)):
            assert len(self.shape) == 1
            return Lookup(self.table, index=item, vent=self.vent)
        if isinstance(item, tuple):
            v = self
            for e in item:
                v = v[e]
            return v
        return NotImplemented

    def __neg__(self) -> Lookup:
        return Lookup(-self.table, index=self.index, vent=self.vent)

    def __sub__(self, other: int | np.ndarray) -> Lookup:
        return Lookup(self.table - other, index=self.index, vent=self.vent)

    def __xor__(self, other: int | np.ndarray) -> Lookup:
        return Lookup(self.table ^ other, index=self.index, vent=self.vent)

    def __ixor__(self, other: int | np.ndarray | ZeroArray | ConstArray) -> Lookup:
        if isinstance(other, ZeroArray):
            pass
        elif isinstance(other, ConstArray):
            self.table ^= other.val
        else:
            self.table ^= other
        return self

    def __isub__(self, other: int | np.ndarray | ZeroArray | ConstArray) -> Lookup:
        if isinstance(other, ZeroArray):
            pass
        elif isinstance(other, ConstArray):
            self.table -= other.val
        else:
            self.table -= other
        return self

    def __iadd__(self, other: int | np.ndarray) -> Lookup:
        if isinstance(other, ZeroArray):
            pass
        elif isinstance(other, ConstArray):
            self.table += other.val
        else:
            self.table += other
        return self

    def __rsub__(self, other: int) -> Lookup:
        return Lookup(other - self.table, index=self.index, vent=self.vent)

    def __mod__(self, other: int) -> Lookup:
        return Lookup(self.table % other, index=self.index, vent=self.vent)

    def __ge__(self, other: quint) -> LookupCmp:
        if isinstance(other, quint):
            return LookupCmp(self, other, ">=")
        return NotImplemented

    def __le__(self, other: quint) -> LookupCmp:
        if isinstance(other, quint):
            return LookupCmp(other, self, ">=")
        return NotImplemented

    def __gt__(self, other: quint) -> LookupCmp:
        if isinstance(other, quint):
            return LookupCmp(self, other, ">")
        return NotImplemented

    def __lt__(self, other: quint) -> LookupCmp:
        if isinstance(other, quint):
            return LookupCmp(other, self, ">")
        return NotImplemented
