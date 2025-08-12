from __future__ import annotations

import collections
import random
from typing import Sequence, Any

import numpy as np

from scatter_script._cost_key import CostKey
from scatter_script._quint import quint, rvalue_multi_int
from scatter_script._lookup import Lookup, LookupCmp
from scatter_script._util import phase_correction_table_for_qrom_uncompute


class QPU:
    """A quantum computer simulator specialized in fuzzing of reversible arithmetic.

    The simulator works by tracking N classical registers paired with N phase values.
    The values of the classical registers are changed by applying operations like
    inplace addition etc. The phase values can be changed by applying things like Z
    gates to qubits.

    Computations are checked by verifying the classical registers end up in the correct
    final states given their input states, that intermediate values have been zero'd out,
    and that the phase values have also been correctly zero'd (using the method
    `qpu.verify_clean_finish()`).

    Supports classical arithmetic on registers, but doesn't strictly enforce that
    irreversible operations cannot be performed. The user is responsible for ensuring
    the operations they do are actually reversible. Nothing will stop you from
    writing `a -= a`, and nothing will notice this implies catastrophic phase error on
    the value that was stored in `a` due to the irreversibility.

    Supports measurement based uncomputation, with verification of phase corrections
    done by fuzzing. A register can be measured in the X basis, or QFT basis, clearing
    the register and causing a phase kickback into the system's phase determined by the
    register's old value and the returned measurement result. It is the caller's
    responsibility to then correct the phase kickback (using the fact that the register's
    old value was a function of other registers still present). At the end of a
    computation, the method `qpu.verify_clean_finish()` can be called to check that all
    phase kickbacks were cancelled. Due to the randomness in the kickback this doesn't
    verify the corrections are correct for all trajectories, but by repeatedly running
    the method confidence is gained by fuzzing.

    If the user accidentally performs an X basis measurement of a register that isn't a
    function of remaining registers, the simulators state becomes incorrect. This case
    requires accounting for interference effects between all branches of the computation,
    which this simulator completely ignores.

    Has basic support for superposition masking, but isn't capable of verifying that
    the mask actually prevents the masked value from being learned. A register can be
    initialized "superposed", which causes the simulator to set its classical values
    randomly.
    """

    def __init__(self, *, num_branches: int):
        """Creates a quantum simulator.

        Args:
            num_branches: When a 'superposed' value is created, this is how
                many random samples are taken from its range and then tracked
                forwards. The idea is to show that each branch individually
                behaves correctly (keeping in mind that the simulator cannot
                do interference between branches).
        """
        self.num_branches: int = num_branches
        self.cost_counters: collections.Counter[CostKey] = collections.Counter()
        self.branch_phases_u64: np.ndarray = np.zeros(num_branches, dtype=np.uint64)
        self.allocated: list[quint] = []
        self._alloc_count = 0
        self.uncompute_info: dict[Any, list[Any]] = {}

    def push_uncompute_info(self, obj: Any, *, key: Any = None) -> None:
        """Push arbitrary information onto a named stack, for use later when uncomputing.

        For example, this could be phase corrections from the deletion of a table lookup
        when computing a function. If the table lookup also happens when uncomputing the
        function, the phase corrections from the computation can be merged into the phase
        corrections for the deletion of the table during the uncomputation. This method is
        a way to communicate those corrections from the computation to the uncomputation.
        """
        if key not in self.uncompute_info:
            self.uncompute_info[key] = []
        self.uncompute_info[key].append(obj)

    def pop_uncompute_info(self, *, key: Any = None) -> Any:
        stack = self.uncompute_info[key]
        result = stack.pop()
        if not stack:
            del self.uncompute_info[key]
        return result

    def alloc_toffoli_state(self) -> quint:
        result = self.alloc_quint(length=3, scatter=True, scatter_range=4)
        result[2] ^= result[1] & result[0]
        return result

    def verify_clean_finish(self):
        """Checks that everything is uncomputed, then discards allocated registers.

        Verifies all allocated registers are in the zero state.
        Verifies all phases are zero.
        Verifies all uncomputation stacks are empty.

        Keeps cost counters.
        """
        problems = []
        for q in self.allocated:
            if q.UNPHYSICAL_copy().UNPHYSICAL_is_nonzero_in_any_branch():
                problems.append(q)
        if problems:
            lines = ["Some registers weren't uncomputed or del_measured:"]
            for q in problems:
                lines.append(f"    alloc_id={q._alloc_id} {q}")
            raise ValueError("\n".join(lines))
        if self.uncompute_info:
            raise ValueError(f"There was uncompute information left over: {self.uncompute_info}")
        if np.any(self.branch_phases_u64):
            denom_power = 64
            p = [int(e) for e in self.branch_phases_u64]
            while all(e & 1 == 0 for e in p):
                denom_power -= 1
                p = [e >> 1 for e in p]
            lines = ["Some phases weren't exactly uncomputed. Final phases:"]
            for e in p:
                if not e:
                    lines.append("    0")
                else:
                    lines.append(f"    {e} / {1 << denom_power} * 2π")
            raise ValueError("\n".join(lines))
        if self._alloc_count > 0:
            raise ValueError(
                f"Some registers weren't deallocated (allocated qubit count is {self._alloc_count})."
            )
        if self._alloc_count < 0:
            raise ValueError(
                f"More deallocations than allocations (allocated qubit count is {self._alloc_count})."
            )

        self.allocated.clear()
        self._alloc_count = 0
        self.branch_phases_u64[:] = 0

    def _note_implicitly_used_temporary_qubits(self, used_qubits: int):
        self.cost_counters[CostKey("qubits")] = max(
            self._alloc_count + used_qubits, self.cost_counters.get(CostKey("qubits"), 0)
        )

    def _dealloc_register(self, register: quint):
        if self._alloc_count < len(register):
            raise ValueError("Deallocated more than was allocated.")
        self._alloc_count -= len(register)

    def phase_correct_for_del_lookup(
        self, *, q_address: quint | rvalue_multi_int, table: Sequence[int], value_mx: int
    ) -> None:
        """Performs the phase correction part of a measurement-based uncomputation of a lookup.

        Equivalent to:

            qc.z(table[q_address] & value_mx)

        Args:
            q_address: The address register that picked a row from the table.
            table: The table of values to choose from.
            value_mx: The measurement result that was returned from measuring the
                qubits of the output register in the X basis.
        """
        address_length = len(table).bit_length()
        if isinstance(q_address, quint) and len(q_address) > address_length:
            q_address = q_address[:address_length]
        phase_correction_table = phase_correction_table_for_qrom_uncompute(table=table, mx=value_mx)
        self.phase_flip_by_lookup(Q_address=q_address, table=phase_correction_table)

    def phase_flip_by_lookup(
        self, *, Q_address: quint | rvalue_multi_int, table: Sequence[bool]
    ) -> None:
        self.cost_counters[CostKey("phaseflip_by_lookup", {"N": len(table)})] += 1
        addresses = Q_address.UNPHYSICAL_branch_vals
        for k in range(len(addresses)):
            if table[addresses[k]]:
                self.branch_phases_u64[k] ^= 1 << 63

    def alloc_wandering_and(
        self,
        control1: quint | rvalue_multi_int | int | bool,
        control2: quint | rvalue_multi_int | int | bool,
        *,
        out: quint | None = None,
        length: int | None = None,
        disable_wandering: bool = False,
    ) -> tuple[int, int, quint]:
        """Performs an AND gate as if by gate teleportation without corrections.

        The teleportation results in the controls potentially being inverted during
        the computation of the AND gate. The method returns whether or not this
        occurred, instead of attempting to correct it. Effectively this means the
        method returns whether the gate that was actually performed was AND, NAND,
        ANDNOT, or the other ANDNOT.

        Args:
            control1: The first control register. Allowed to be a multi-qubit
                register, in which case uncorrected AND gates are broadcast
                over the qubits.
            control2: The second control register. Must be the same length as
                as `control1`.
            out: Defaults to None (allocate new output register). The output
                register to write the result into. Must be zero'd.
            length: Defaults to None (infer). Explicit length for output register.
            disable_wandering: Defaults to False. When set to True, the random
                comparisons will be forced to 0 so that the qubit result is exactly
                the AND of the two inputs.

        Returns:
            A tuple (cmp1, cmp2, result) where result == (control1 ^ cmp1) & (control2 ^ cmp2).

            cmp1: Whether or not the teleportation was inverted relative to the first control.
            cmp2: Whether or not the teleportation was inverted relative to the second control.
            result: The result of the AND gate.

        Example:
            >>> from scatter_script import QPU, quint
            >>> qpu = QPU(num_branches=16)
            >>> a = qpu.alloc_quint(length=1, scatter=True)
            >>> b = qpu.alloc_quint(length=1, scatter=True)
            >>> cmp1, cmp2, c = qpu.alloc_wandering_and(a, b)
            >>> assert c == (a ^ cmp1) & (b ^ cmp2)

            >>> # manually correct the result
            >>> if cmp1:
            ...     c ^= b
            >>> if cmp2:
            ...     c ^= a
            >>> if cmp1 and cmp2:
            ...     c ^= 1
            >>> assert c == a & b
        """
        if length is not None:
            inferred_length = length
        elif out is not None and isinstance(out, quint):
            inferred_length = len(out)
        elif isinstance(control1, quint):
            inferred_length = len(control1)
        elif isinstance(control2, quint):
            inferred_length = len(control2)
        else:
            raise ValueError(
                f"Can't infer length from {control1=}, {control2=}, {out=}, {length=}.\nSpecify explicit length="
            )

        if out is not None and not isinstance(out, quint):
            raise ValueError(f"{out=} is not None and not isinstance(out, quint)")
        if not isinstance(control1, (quint, rvalue_multi_int, int, bool)):
            raise ValueError(f"not isinstance({control1=}, (quint, rvalue_multi_int, int, bool))")
        if not isinstance(control2, (quint, rvalue_multi_int, int, bool)):
            raise ValueError(f"not isinstance({control2=}, (quint, rvalue_multi_int, int, bool))")
        if isinstance(control1, quint) and len(control1) != inferred_length:
            raise ValueError(f"{len(control1)=} != {inferred_length=}")
        if isinstance(control2, quint) and len(control2) != inferred_length:
            raise ValueError(f"{len(control2)=} != {inferred_length=}")
        if isinstance(out, quint) and len(out) != inferred_length:
            raise ValueError(f"{len(out)=} != {inferred_length=}")

        if out is None:
            out = self.alloc_quint(length=inferred_length)
        elif out.UNPHYSICAL_is_nonzero_in_any_branch():
            raise ValueError(f"Output register must be zero'd, but saw {out=} != 0")

        self.cost_counters[CostKey("uncorrected_and")] += inferred_length

        if disable_wandering:
            cmp1 = 0
            cmp2 = 0
        else:
            cmp1 = random.randrange(1 << inferred_length)
            cmp2 = random.randrange(1 << inferred_length)
        out.UNPHYSICAL_write_branch_vals(
            rvalue_multi_int.from_value(
                (control1 ^ cmp1) & (control2 ^ cmp2), expected_count=self.num_branches
            ).UNPHYSICAL_branch_vals
        )

        return cmp1, cmp2, out

    def alloc_and(
        self,
        control1: quint | rvalue_multi_int | int | bool,
        control2: quint | rvalue_multi_int | int | bool,
        *,
        out: quint | None = None,
        length: int | None = None,
    ) -> quint:
        """Computes the result of an AND gate, storing it in an output qubit.

        Args:
            control1: The first control register. Allowed to be a multi-qubit
                register, in which case AND gates are broadcast over the qubits.
            control2: The second control register. Must be the same length as
                as `control1`.
            out: Defaults to None (allocate new output register). The output
                register to write the result into. Must be zero'd.
            length: Defaults to None (infer). Explicit length for output register.

        Returns:
            A quint result where result == control1 & control2.

        Example:
            >>> from scatter_script import QPU, quint
            >>> qpu = QPU(num_branches=16)
            >>> a = qpu.alloc_quint(length=1, scatter=True)
            >>> b = qpu.alloc_quint(length=1, scatter=True)
            >>> c = qpu.alloc_and(a, b)
            >>> assert c == a & b
        """
        return self.alloc_wandering_and(
            control1=control1, control2=control2, out=out, length=length, disable_wandering=True
        )[2]

    def alloc_phase_gradient(self, *, length: int) -> quint:
        """Returns a result representing the state QFT |-1>.

        This state has the property that adding a register X into it will
        phase X by an amount proportional its value.

        The phase gradient state must be deleted using `del_phase_gradient_state`
        to finalize the phasing effect in the simulator.
        """
        if length > 64:
            raise NotImplementedError(f"{length=} > 64")
        self.cost_counters[CostKey("alloc_phase_gradient_state", {"n": length})] += 1
        Q_phase_gradient = self.alloc_quint(length=length, scatter=True)
        v = Q_phase_gradient.UNPHYSICAL_branch_vals
        old = np.seterr(over="ignore")
        for k in range(len(v)):
            p = v[k] << (64 - length)
            p = np.uint64(int(p) & ~(-1 << 64))
            self.branch_phases_u64[k] += p
        np.seterr(**old)
        return Q_phase_gradient

    def del_phase_gradient(self, Q_phase_gradient: quint) -> None:
        length = len(Q_phase_gradient)
        v = Q_phase_gradient.UNPHYSICAL_branch_vals
        old = np.seterr(over="ignore")
        for k in range(len(v)):
            p = v[k] << (64 - length)
            p = np.uint64(int(p) & ~(-1 << 64))
            self.branch_phases_u64[k] -= p
        np.seterr(**old)
        Q_phase_gradient.UNPHYSICAL_force_del(dealloc=True)

    def alloc_quint(
        self,
        *,
        length: int,
        val: int | None = None,
        scatter: bool = False,
        scatter_range: range | int | None = None,
    ) -> quint:
        """Allocates a new simulated register.

        Args:
            length: The number of qubits in the register. The register is typically interpreted as
                a 2s complement unsigned integer, meaning it can represent integers in
                [0, 2**length).
            val: Defaults to None (unused). A deterministic initial value of the register, that will
                be used across all branches.
            scatter: Defaults to False (unused). When set to True, the register will be assigned a
                random value within each branch.
            scatter_range: Defaults to None (meaning `range(1 << length)`). Requires `scatter=True`
                to be relevant. Controls the range of possible random values when scattering.

        Returns:
            The allocated register.
        """
        if val and scatter:
            raise ValueError(f"Can't specify both {val=} and {scatter=}.")
        if scatter_range is not None and not scatter:
            raise ValueError(f"Must specify scatter=True to specify {scatter_range=}.")
        if scatter:
            if scatter_range is None:
                scatter_range = range(1 << length)
            elif isinstance(scatter_range, int):
                scatter_range = range(scatter_range)
            elif isinstance(scatter_range, range):
                pass
            else:
                raise NotImplementedError(f"{scatter_range=}")
            m = [
                random.randrange(scatter_range.start, scatter_range.stop, scatter_range.step)
                for _ in range(self.num_branches)
            ]
        elif val is not None:
            m = [val] * self.num_branches
        else:
            m = [0] * self.num_branches
        result = quint(buffer=m, offset=0, length=length, parent=self, alloc_id=len(self.allocated))
        self.allocated.append(result)
        self._alloc_count += len(result)
        self.cost_counters[CostKey("qubits")] = max(
            self._alloc_count, self.cost_counters.get(CostKey("qubits"), 0)
        )
        return result

    def alloc_lookup(
        self,
        *,
        Q_address: quint | rvalue_multi_int | tuple[quint, ...],
        table: Sequence[int] | np.ndarray,
        output_length: int,
    ) -> quint:
        q_out = self.alloc_quint(length=output_length)
        q_out.init_from_lookup(table=table, q_address=Q_address)
        return q_out

    def phase(self, a: int | quint | rvalue_multi_int, turns: float) -> None:
        turns %= 1
        if turns == 0.5:
            op_name = "z"
        elif turns == 0.25 or turns == 0.75:
            op_name = "s"
        else:
            op_name = "phase"
        if isinstance(a, int):
            self.cost_counters[CostKey("global_phase")] += 1
            if a < 0:
                raise NotImplementedError(f"phase of negative integer {a=}")
            p = turns * a.bit_count()
            p *= 2**64
            assert p == int(p)
            p = np.uint64(int(p) & ~(-1 << 64))
            old = np.seterr(over="ignore")
            self.branch_phases_u64[:] += p
            np.seterr(**old)
            return

        if isinstance(a, quint):
            self.cost_counters[CostKey(op_name)] += len(a)
        else:
            self.cost_counters[CostKey(op_name, {"unclassified_rval": True})] += 1
        v1 = a.UNPHYSICAL_branch_vals
        for k in range(len(v1)):
            p = turns * v1[k].bit_count()
            p *= 2**64
            assert p == int(p)
            p = np.uint64(int(p) & ~(-1 << 64))
            old = np.seterr(over="ignore")
            self.branch_phases_u64[k] += p
            np.seterr(**old)

    def global_phase_flip(self) -> None:
        self.branch_phases_u64[:] ^= 1 << 63

    def phase_flip_if_ge(self, a: quint, b: Lookup):
        assert isinstance(a, quint)
        assert isinstance(b, Lookup)
        phases = self.phase_flip_if_ge_lookup__defered_phasing(
            Q_target=a, Q_address=b.index, table=b.table
        )
        b.perform_venting(phases)

    def phase_flip_if_lt(self, a: quint, b: Lookup):
        assert isinstance(a, quint)
        assert isinstance(b, Lookup)
        phases = self.phase_flip_if_lt_lookup__defered_phasing(
            Q_target=a, Q_address=b.index, table=b.table
        )
        b.perform_venting(phases)

    def z(self, a: int | quint | rvalue_multi_int) -> None:
        if isinstance(a, Lookup):
            self.phase_flip_by_lookup(Q_address=a.index, table=a.table)
            return
        if isinstance(a, LookupCmp):
            if a.cmp == ">=":
                self.phase_flip_if_ge(a.a, a.b)
            elif a.cmp == ">":
                self.phase_flip_if_lt(a.b, a.a)
            else:
                raise NotImplementedError(f"{a=}")
            return
        if isinstance(a, int):
            if a.bit_count() & 1:
                self.global_phase_flip()
            return

        v = a.UNPHYSICAL_branch_vals
        for k in range(len(v)):
            if v[k].bit_count() & 1:
                self.branch_phases_u64[k] ^= 1 << 63

    def s(self, a: int | quint | rvalue_multi_int) -> None:
        self.phase(a, 0.25)

    def s_dag(self, a: int | quint | rvalue_multi_int) -> None:
        self.phase(a, -0.25)

    def ccz(
        self, a: quint | rvalue_multi_int, b: quint | rvalue_multi_int, c: quint | rvalue_multi_int
    ) -> None:
        assert isinstance(a, quint) and isinstance(b, quint) and isinstance(c, quint)
        self.cost_counters[CostKey("ccz")] += min(len(a), len(b), len(c))
        v1 = a.UNPHYSICAL_branch_vals
        v2 = b.UNPHYSICAL_branch_vals
        v3 = c.UNPHYSICAL_branch_vals
        for k in range(len(v1)):
            if (v1[k] & v2[k] & v3[k]).bit_count() & 1:
                self.branch_phases_u64[k] ^= 1 << 63

    def cz(self, a: quint | rvalue_multi_int, b: quint | rvalue_multi_int) -> None:
        if isinstance(a, quint) and isinstance(b, quint):
            self.cost_counters[CostKey("CZ")] += min(len(a), len(b))
        elif isinstance(a, quint):
            self.cost_counters[CostKey("CZ")] += len(a)
        elif isinstance(b, quint):
            self.cost_counters[CostKey("CZ")] += len(b)
        else:
            self.cost_counters[CostKey("CZ", {"unclassified_rval": True})] += 1
        v1 = rvalue_multi_int.from_value(a, expected_count=self.num_branches).UNPHYSICAL_branch_vals
        v2 = rvalue_multi_int.from_value(b, expected_count=self.num_branches).UNPHYSICAL_branch_vals
        for k in range(len(v1)):
            if (v1[k] & v2[k]).bit_count() & 1:
                self.branch_phases_u64[k] ^= 1 << 63

    def cz_multi_target(
        self,
        control: quint | rvalue_multi_int | int,
        potential_targets: Sequence[quint | rvalue_multi_int | int],
        mask: Sequence[bool],
    ) -> None:
        if isinstance(control, int):
            assert 0 <= control <= 1
            if control == 1:
                for t, m in zip(potential_targets, mask):
                    if m:
                        self.z(t)
            return

        if isinstance(control, quint):
            assert len(control) == 1
        quantum_targets = 0
        for t in potential_targets:
            if isinstance(t, quint):
                assert len(t) == 1
                quantum_targets += 1
            elif isinstance(t, rvalue_multi_int):
                quantum_targets += 1
        if quantum_targets > 0:
            self.cost_counters[CostKey("CZ_multi_target", {"n": quantum_targets})] += 1
        c = control.UNPHYSICAL_branch_vals
        for target, m in zip(potential_targets, mask):
            if m:
                t = rvalue_multi_int.from_value(
                    target, expected_count=self.num_branches
                ).UNPHYSICAL_branch_vals
                for k in range(len(c)):
                    if (c[k] & t[k]) & 1:
                        self.branch_phases_u64[k] ^= 1 << 63

    def lookup1_add(
        self,
        *,
        Q_target: quint,
        Q_control: quint | rvalue_multi_int,
        offset_if_false: int,
        offset_if_true: int,
    ) -> None:
        """Special case of lookup+subtract where the address is a single qubit.

        This case is substantially cheaper, so has its own method if wanted.
        """
        Q_tmp = self.alloc_lookup(
            Q_address=Q_control,
            table=[offset_if_false, offset_if_true],
            output_length=len(Q_target),
        )
        Q_target += Q_tmp
        mx = Q_tmp.del_measure_x(e)
        v0 = (mx & offset_if_false).bit_count() & 1
        v1 = (mx & offset_if_true).bit_count() & 1
        if v0 ^ v1:
            self.z(Q_control[0])
        if v0:
            self.global_phase_flip()

    def lookup1_sub(
        self,
        *,
        Q_target: quint,
        Q_control: quint | rvalue_multi_int,
        offset_if_false: int,
        offset_if_true: int,
    ) -> None:
        """Special case of lookup+subtract where the address is a single qubit.

        This case is substantially cheaper, so has its own method if wanted.
        """
        Q_tmp = self.alloc_lookup(
            Q_address=Q_control,
            table=[offset_if_false, offset_if_true],
            output_length=len(Q_target),
        )
        Q_target -= Q_tmp
        mx = Q_tmp.del_measure_x()
        v0 = (mx & offset_if_false).bit_count() & 1
        v1 = (mx & offset_if_true).bit_count() & 1
        if v0 ^ v1:
            self.z(Q_control[0])
        if v0:
            self.global_phase_flip()

    def lookup_sub__defered_phasing(
        self, *, Q_target: quint, Q_address: quint | rvalue_multi_int, table: Sequence[int]
    ) -> np.ndarray:
        Q_tmp = self.alloc_lookup(Q_address=Q_address, table=table, output_length=len(Q_target))
        Q_target -= Q_tmp
        return phase_correction_table_for_qrom_uncompute(table=table, mx=Q_tmp.del_measure_x())

    def lookup_add__defered_phasing(
        self, *, Q_target: quint, Q_address: quint | rvalue_multi_int, table: Sequence[int]
    ) -> np.ndarray:
        Q_tmp = self.alloc_lookup(Q_address=Q_address, table=table, output_length=len(Q_target))
        Q_target += Q_tmp
        return phase_correction_table_for_qrom_uncompute(table=table, mx=Q_tmp.del_measure_x())

    def lookup_xor__defered_phasing(
        self, *, Q_target: quint, Q_address: quint | rvalue_multi_int, table: Sequence[int]
    ) -> np.ndarray:
        Q_tmp = self.alloc_lookup(Q_address=Q_address, table=table, output_length=len(Q_target))
        Q_target ^= Q_tmp
        return phase_correction_table_for_qrom_uncompute(table=table, mx=Q_tmp.del_measure_x())

    def phase_flip_if_lt_lookup__defered_phasing(
        self, *, Q_target: quint, Q_address: quint | rvalue_multi_int, table: Sequence[int]
    ) -> np.ndarray:
        Q_tmp = self.alloc_lookup(Q_address=Q_address, table=table, output_length=len(Q_target))
        Q_target.phase_flip_if_lt(Q_tmp)
        return phase_correction_table_for_qrom_uncompute(table=table, mx=Q_tmp.del_measure_x())

    def phase_flip_if_ge_lookup__defered_phasing(
        self, *, Q_target: quint, Q_address: quint | rvalue_multi_int, table: Sequence[int]
    ) -> np.ndarray:
        Q_tmp = self.alloc_lookup(Q_address=Q_address, table=table, output_length=len(Q_target))
        Q_target.phase_flip_if_ge(Q_tmp)
        return phase_correction_table_for_qrom_uncompute(table=table, mx=Q_tmp.del_measure_x())
