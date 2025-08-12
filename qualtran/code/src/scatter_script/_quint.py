from __future__ import annotations

import random
from typing import Iterator, Sequence, Iterable, TYPE_CHECKING

import numpy as np

from scatter_script._cost_key import CostKey
from scatter_script._rvalue_multi_int import rvalue_multi_int
from scatter_script._util import slice_int, phase_correction_table_for_qrom_uncompute

if TYPE_CHECKING:
    from scatter_script._qpu import QPU
    from scatter_script._lookup import Lookup, LookupCmp


class quint:
    """Simulation state of a superposed integer."""

    def __init__(self, *, parent: QPU, buffer: list[int], offset: int, length: int, alloc_id: int):
        if not isinstance(buffer, list):
            raise ValueError(f"not isinstance({buffer=}, list)")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(f"not isinstance({offset=}, int) or offset < 0")
        if not isinstance(length, int) or length < 0:
            raise ValueError(f"not isinstance({length=}, int) or length < 0")
        self._buffer = buffer
        self._offset = offset
        self._length = length
        self._parent = parent
        self._alloc_id = alloc_id

    def UNPHYSICAL_is_nonzero_in_any_branch(self) -> bool:
        """Unphysically check if the register is non-zero in any branch."""
        return self.UNPHYSICAL_copy().UNPHYSICAL_is_nonzero_in_any_branch()

    @property
    def UNPHYSICAL_branch_vals(self) -> tuple[int, ...]:
        """Unphysically reads the value of the register in each branch."""
        s = self._offset
        m = ~(-1 << self._length)
        return tuple((v >> s) & m for v in self._buffer)

    def UNPHYSICAL_copy(self) -> rvalue_multi_int:
        """Unphysically makes a copy of the register's value in each branch."""
        return rvalue_multi_int(self.UNPHYSICAL_branch_vals)

    def z(self) -> None:
        self._parent.z(self)

    def cz(self, other: quint) -> None:
        if self._parent is not other._parent:
            raise NotImplementedError("self._parent is not other._parent")
        self._parent.cz(self, other)

    def phase_flip_if_ge(self, other: quint | rvalue_multi_int | int) -> None:
        if isinstance(other, quint):
            m = max(len(self), len(other))
            d = m - min(len(self), len(other))
            self._parent.cost_counters[CostKey("phase_flip_if_cmp", {"n": m})] += 1
            # Assumes using the adder from https://arxiv.org/abs/1709.06648
            # Pad the offset register up to the same size as the target, and use ancilla qubits to reduce Toffolis.
            self._parent._note_implicitly_used_temporary_qubits(d + m)
        else:
            self._parent.cost_counters[CostKey("phase_flip_if_cmp", {"n": -1})] += 1
            self._parent._note_implicitly_used_temporary_qubits(len(self) + max(0, len(self) - 1))

        v = (self >= other).UNPHYSICAL_branch_vals
        for k in range(len(v)):
            if v[k]:
                self._parent.branch_phases_u64[k] ^= 1 << 63

    def phase_flip_if_lt(self, other: quint | rvalue_multi_int | int) -> None:
        if isinstance(other, quint):
            m = max(len(self), len(other))
            d = m - min(len(self), len(other))
            self._parent.cost_counters[CostKey("phase_flip_if_cmp", {"n": m})] += 1
            # Assumes using the adder from https://arxiv.org/abs/1709.06648
            # Pad the offset register up to the same size as the target, and use ancilla qubits to reduce Toffolis.
            self._parent._note_implicitly_used_temporary_qubits(d + m)
        else:
            self._parent.cost_counters[CostKey("phase_flip_if_cmp", {"n": -1})] += 1
            self._parent._note_implicitly_used_temporary_qubits(len(self) + max(0, len(self) - 1))

        v = (self < other).UNPHYSICAL_branch_vals
        for k in range(len(v)):
            if v[k]:
                self._parent.branch_phases_u64[k] ^= 1 << 63

    def z_mask(self, mask: int) -> None:
        for k in range(len(self)):
            if slice_int(mask, k):
                self[k].z()

    def phaseflip_by_lookup(self, table: Sequence[bool]) -> None:
        self._parent.phase_flip_by_lookup(Q_address=self, table=table)

    def phase_correct_for_del_lookup(self, *, table: Sequence[int], value_mx: int):
        self._parent.phase_correct_for_del_lookup(q_address=self, table=table, value_mx=value_mx)

    def phase_gradient(self, multiplier: float | int) -> None:
        v = self.UNPHYSICAL_branch_vals
        for k in range(len(v)):
            p = v[k] * multiplier << (64 - len(self))
            assert p == int(p)
            p = np.uint64(p & ~(-1 << 64))
            old = np.seterr(over="ignore")
            self._parent.branch_phases_u64[k] += p
            np.seterr(**old)

    def __irshift__(self, shift: int) -> quint:
        assert shift >= 0
        if shift > 0:
            assert self[:shift] == 0
            self.UNPHYSICAL_write_branch_vals(v >> shift for v in self.UNPHYSICAL_branch_vals)
        return self

    def __ilshift__(self, shift: int) -> quint:
        assert shift >= 0
        if shift > 0:
            assert self[-shift:] == 0
            self.UNPHYSICAL_write_branch_vals(v << shift for v in self.UNPHYSICAL_branch_vals)
        return self

    def UNPHYSICAL_write_branch_vals(self, new_vals: Iterable[int]) -> None:
        """Unphysically overwrites the value of the register in each branch."""
        m = ~(-1 << self._length)
        m <<= self._offset
        k = 0
        for v_raw in new_vals:
            v = int(v_raw)
            v <<= self._offset
            self._buffer[k] &= ~m
            self._buffer[k] |= v & m
            k += 1
        assert k == len(self._buffer)

    def UNPHYSICAL_force_del(self, dealloc: bool) -> None:
        """Unphysically zeroes the contents of a register.

        This method is useful when initially writing a method, when
        you haven't yet decided on how to do the uncomputation but
        want to verify that what you've written so far is correct.

        This method is useful when writing unit tests that use
        `qpu.verify_clean_finish()`, for focusing the test on
        the uncomputation of one specific value instead of all
        values.

        This method is useful when emulating certain complex
        effects. For example, the internal implementation of a
        `del_measure_x` method could call this method to clear the
        register after applying appropriate phase kickbacks.

        This method shouldn't be directly used by code that is
        attempting to behave like a real quantum computer (except
        in the service of emulating real effects like the behavior
        of del_measure_x).
        """
        m = ~(-1 << self._length)
        m <<= self._offset
        for k in range(len(self._buffer)):
            self._buffer[k] &= ~m
        if dealloc:
            self._parent._dealloc_register(self)

    def del_by_equal_to(self, v: int | Lookup):
        """Uncomputes the register to zero using the given value as a reference.

        If the value is a classical integer, it's magically verified the register
        is actually equal to the value (to catch bugs) and then the register is
        zero'd.

        If the value is a Lookup, then an X-basis measurement based uncomputation
        of the register is performed and the phase correction data is dumped into the
        lookup's vent. Verification isn't performed immediately in this case; it's
        expected that needing to correctly zero the phases will verify the deletion.
        """
        if isinstance(v, int):
            if not all(e == v for e in self.UNPHYSICAL_branch_vals):
                raise ValueError(f"Not equal to the specified constant: {v} != {self}")
            self.UNPHYSICAL_force_del(dealloc=True)
            return

        from scatter_script._lookup import Lookup

        if isinstance(v, Lookup):
            mx = self.del_measure_x()
            v.perform_venting(phase_correction_table_for_qrom_uncompute(table=v.table, mx=mx))
        self._parent._dealloc_register(self)

    def swap_with(self, other: quint) -> None:
        """Swaps the contents of this register with the given register."""
        assert len(self) == len(other)
        self._parent.cost_counters[CostKey("swap")] += len(self)
        a = self.UNPHYSICAL_branch_vals
        b = other.UNPHYSICAL_branch_vals
        self.UNPHYSICAL_write_branch_vals(b)
        other.UNPHYSICAL_write_branch_vals(a)

    def mx_rz(self) -> int:
        """Performs X basis measurements of the register's qubits, clearing them to 0.

        Each qubit causes phase kickback equal to a Z gate if and only if its old value
        was 1 and its measurement result was 1. Correcting this kickback requires that
        the register's value was a function of other registers' values.
        """
        result = random.randrange(1 << len(self))
        vals = self.UNPHYSICAL_branch_vals
        for k in range(len(vals)):
            if (result & vals[k]).bit_count() & 1:
                self._parent.branch_phases_u64[k] ^= 1 << 63
        self.UNPHYSICAL_force_del(dealloc=False)
        return result

    def del_measure_x(self) -> int:
        """Deallocates the register by measuring its value in the X basis.

        Each qubit causes phase kickback equal to a Z gate if and only if its old value
        was 1 and its measurement result was 1. Correcting this kickback requires that
        the register's value was a function of other registers' values.
        """
        result = self.mx_rz()
        self._parent._dealloc_register(self)
        return result

    def del_measure_qft(self) -> int:
        """Performs a QFT basis measurement of the register, clearing it to 0.

        Causes phase kickback proportional to the register's old value times the
        measurement result, times 2**-len(register) of a turn. Correcting this
        kickback requires that the register's value was a function of other registers'
        values.
        """
        if len(self) > 64:
            raise NotImplementedError(f"del_measure_qft on register longer than 64 qubits")
        result = random.randrange(1 << len(self))
        tick_scale = np.uint64(result << (64 - len(self)))
        vals = np.array(self.UNPHYSICAL_branch_vals, dtype=np.uint64)
        old = np.seterr(over="ignore")
        self._parent.branch_phases_u64 += vals * tick_scale
        np.seterr(**old)
        self.UNPHYSICAL_force_del(dealloc=True)
        return result

    def init_from_lookup(
        self,
        *,
        q_address: quint | rvalue_multi_int | tuple[quint, ...],
        table: Sequence[int] | np.ndarray,
    ):
        assert self == 0
        if isinstance(q_address, tuple):
            total = rvalue_multi_int.from_stacking(q_address)
        else:
            total = q_address
        self._parent.cost_counters[CostKey("init_lookup", {"N": len(table), "w": len(self)})] += 1
        self.UNPHYSICAL_write_branch_vals(table[q] for q in total.UNPHYSICAL_branch_vals)

    def del_from_lookup(
        self,
        *,
        Q_address: quint | rvalue_multi_int | tuple[quint, ...],
        table: Sequence[int] | np.ndarray,
    ):
        if isinstance(Q_address, tuple):
            total = rvalue_multi_int.from_stacking(Q_address)
        else:
            total = Q_address

        self._parent.cost_counters[CostKey("del_lookup", {"N": len(table), "w": len(self)})] += 1
        mx = self.del_measure_x()
        self._parent.phase_correct_for_del_lookup(q_address=total, table=table, value_mx=mx)

    def __int__(self) -> int:
        return self.UNPHYSICAL_copy().__int__()

    def __bool__(self) -> bool:
        return self.UNPHYSICAL_copy().__bool__()

    def enumerate_windows(self, window_size: int) -> Iterator[tuple[int, quint]]:
        k = 0
        for offset in range(0, len(self), window_size):
            yield k, self[offset : offset + window_size]
            k += 1

    def __iter__(self) -> Iterator[quint]:
        for k in range(len(self)):
            yield self[k]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item: int | slice) -> quint:
        if isinstance(item, int):
            idx = range(self._length)[item]
            return quint(
                buffer=self._buffer,
                offset=self._offset + idx,
                length=1,
                parent=self._parent,
                alloc_id=self._alloc_id,
            )
        if isinstance(item, slice):
            rng = range(self._length)[item]
            if rng.step != 1:
                raise NotImplementedError("step != 1")
            return quint(
                buffer=self._buffer,
                offset=self._offset + rng.start,
                length=len(rng),
                parent=self._parent,
                alloc_id=self._alloc_id,
            )
        return NotImplemented

    def _wrap(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return rvalue_multi_int.from_value(other, expected_count=len(self._buffer))

    def __setitem__(self, item: int | slice, new_value: int | quint | rvalue_multi_int) -> quint:
        self[item].UNPHYSICAL_write_branch_vals(self._wrap(new_value).UNPHYSICAL_branch_vals)
        return self[item]

    def __ixor__(self, offset: int | quint | rvalue_multi_int) -> quint:
        from scatter_script._lookup import Lookup

        if isinstance(offset, Lookup):
            phases = self._parent.lookup_xor__defered_phasing(
                Q_target=self, Q_address=offset.index, table=offset.table
            )
            offset.perform_venting(phases)
            return self

        if isinstance(offset, int):
            self._parent.cost_counters[CostKey("X")] += len(self)
        else:
            self._parent.cost_counters[CostKey("CX")] += len(self)
        offset = self._wrap(offset)
        self.UNPHYSICAL_write_branch_vals(
            a ^ b
            for a, b in zip(self.UNPHYSICAL_branch_vals, offset.UNPHYSICAL_branch_vals, strict=True)
        )
        return self

    def ghz_lookup(self, value: int) -> Lookup:
        """Special case for a lookup with a single-qubit address.

        It makes sense to treat this case differently because the costs of this case
        are disproportionately lower than a general lookup.
        """
        from scatter_script._lookup import Lookup

        assert len(self) == 1
        return Lookup([0, value], index=self, vent=self)

    def __iadd__(self, offset: int | quint | rvalue_multi_int | Lookup) -> quint:
        from scatter_script._lookup import Lookup

        if isinstance(offset, Lookup):
            phases = self._parent.lookup_add__defered_phasing(
                Q_target=self, Q_address=offset.index, table=offset.table
            )
            offset.perform_venting(phases)
            return self

        self._parent.cost_counters[CostKey("__iadd__", {"n": len(self)})] += 1
        # Assumes using the adder from https://arxiv.org/abs/1709.06648
        # Pad the offset register up to the same size as the target, and use ancilla qubits to reduce Toffolis.
        qubits_in_offset = len(offset) if isinstance(offset, quint) else 0
        self._parent._note_implicitly_used_temporary_qubits(
            max(0, len(self) - qubits_in_offset) + max(0, len(self) - 1)
        )

        offset = self._wrap(offset)
        self.UNPHYSICAL_write_branch_vals(
            a + b
            for a, b in zip(self.UNPHYSICAL_branch_vals, offset.UNPHYSICAL_branch_vals, strict=True)
        )
        return self

    def __isub__(self, offset: int | quint | rvalue_multi_int) -> quint:
        from scatter_script._lookup import Lookup

        if isinstance(offset, Lookup):
            phases = self._parent.lookup_sub__defered_phasing(
                Q_target=self, Q_address=offset.index, table=offset.table
            )
            offset.perform_venting(phases)
            return self

        self._parent.cost_counters[CostKey("__iadd__", {"n": len(self)})] += 1
        # Assumes using the adder from https://arxiv.org/abs/1709.06648
        # Pad the offset register up to the same size as the target, and use ancilla qubits to reduce Toffolis.
        qubits_in_offset = len(offset) if isinstance(offset, quint) else 0
        self._parent._note_implicitly_used_temporary_qubits(
            max(0, len(self) - qubits_in_offset) + max(0, len(self) - 1)
        )

        offset = self._wrap(offset)
        self.UNPHYSICAL_write_branch_vals(
            a - b
            for a, b in zip(self.UNPHYSICAL_branch_vals, offset.UNPHYSICAL_branch_vals, strict=True)
        )
        return self

    def __imul__(self, other: int | quint | rvalue_multi_int) -> quint:
        if ((self._wrap(other) & 1) == 0).UNPHYSICAL_is_nonzero_in_any_branch():
            raise ValueError(f"Irreversible multiplication {self=} {other=}")

        self._parent.cost_counters[CostKey("__imul__", {"n": len(self)})] += 1
        offset = self._wrap(other)
        self.UNPHYSICAL_write_branch_vals(
            a * b
            for a, b in zip(self.UNPHYSICAL_branch_vals, offset.UNPHYSICAL_branch_vals, strict=True)
        )
        return self

    def iadd_mod(
        self,
        offset: int | quint | rvalue_multi_int | np.ndarray,
        *,
        modulus: int | quint | rvalue_multi_int,
    ) -> quint:
        if (self >= modulus).UNPHYSICAL_is_nonzero_in_any_branch():
            raise ValueError(
                f"Applied iadd_mod to a register with values not below the modulus {self=} {modulus=}"
            )

        self._parent.cost_counters[CostKey("iadd_mod", {"n": len(self)})] += 1
        offset = self._wrap(offset)
        modulus = self._wrap(modulus)
        self.UNPHYSICAL_write_branch_vals(
            (a + b) % n
            for a, b, n in zip(
                self.UNPHYSICAL_branch_vals,
                offset.UNPHYSICAL_branch_vals,
                modulus.UNPHYSICAL_branch_vals,
                strict=True,
            )
        )
        return self

    def isub_mod(
        self, offset: int | quint | rvalue_multi_int, *, modulus: int | quint | rvalue_multi_int
    ) -> quint:
        return self.iadd_mod(-offset, modulus=modulus)

    def imul_mod(
        self,
        factor: int | quint | rvalue_multi_int | np.ndarray,
        *,
        modulus: int | quint | rvalue_multi_int | np.ndarray,
    ) -> quint:
        if (self >= modulus).UNPHYSICAL_is_nonzero_in_any_branch():
            raise ValueError(
                f"Applied imul_mod to a register with values not below the modulus {self=} {modulus=}"
            )

        self._parent.cost_counters[CostKey("imul_mod", {"n": len(self)})] += 1
        factor = self._wrap(factor)
        modulus = self._wrap(modulus)
        self.UNPHYSICAL_write_branch_vals(
            (a * b) % n
            for a, b, n in zip(
                self.UNPHYSICAL_branch_vals,
                factor.UNPHYSICAL_branch_vals,
                modulus.UNPHYSICAL_branch_vals,
                strict=True,
            )
        )
        return self

    def imul_inv_mod(
        self,
        factor: int | quint | rvalue_multi_int,
        *,
        modulus: int | quint | rvalue_multi_int | np.ndarray,
    ) -> quint:
        return self.imul_mod(pow(int(factor), -1, int(modulus)), modulus=modulus)

    def __pow__(self, power: int | quint | rvalue_multi_int, modulo=None) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy().__pow__(power, modulo)

    def __rpow__(self, other: int | quint | rvalue_multi_int, modulo=None) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy().__rpow__(other, modulo)

    def __mod__(self, other: int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() % other

    def __rmod__(self, other: int) -> rvalue_multi_int:
        return other % self.UNPHYSICAL_copy()

    def __add__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() + other

    def __sub__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() - other

    def __rsub__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return other - self.UNPHYSICAL_copy()

    def __neg__(self) -> rvalue_multi_int:
        return -self.UNPHYSICAL_copy()

    def __invert__(self) -> rvalue_multi_int:
        return ~self.UNPHYSICAL_copy()

    def __lshift__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy().__lshift__(other)

    def __rshift__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy().__rshift__(other)

    def __gt__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int | LookupCmp:
        from scatter_script._lookup import Lookup, LookupCmp

        if isinstance(other, Lookup):
            return LookupCmp(self, other, ">")
        return self.UNPHYSICAL_copy() > other

    def __lt__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int | LookupCmp:
        from scatter_script._lookup import Lookup, LookupCmp

        if isinstance(other, Lookup):
            return LookupCmp(other, self, ">")
        return self.UNPHYSICAL_copy() < other

    def __ge__(
        self, other: int | quint | rvalue_multi_int | Lookup
    ) -> rvalue_multi_int | LookupCmp:
        from scatter_script._lookup import Lookup, LookupCmp

        if isinstance(other, Lookup):
            return LookupCmp(self, other, ">=")
        return self.UNPHYSICAL_copy() >= other

    def __le__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int | LookupCmp:
        from scatter_script._lookup import Lookup, LookupCmp

        if isinstance(other, Lookup):
            return LookupCmp(other, self, ">=")
        return self.UNPHYSICAL_copy() <= other

    def __eq__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() == other

    def __ne__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() != other

    def __and__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() & other

    __rand__ = __and__

    def __or__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() | other

    __ror__ = __or__

    def __xor__(self, other: int | quint | rvalue_multi_int) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() ^ other

    __rxor__ = __xor__

    def __mul__(self, other: int | quint) -> rvalue_multi_int:
        return self.UNPHYSICAL_copy() * other

    __rmul__ = __mul__
    __radd__ = __mul__

    def __repr__(self) -> str:
        vs = self.UNPHYSICAL_branch_vals
        b = ",".join(bin(v)[2:][::-1].ljust(len(self), "0") for v in vs)
        d = ",".join(str(v) for v in vs)
        return f"<quint len={len(self)} val={d} bits={b}>"
