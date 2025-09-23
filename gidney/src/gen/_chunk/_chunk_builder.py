from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast, Literal, TYPE_CHECKING

import stim

from gen._core._complex_util import sorted_complex, xor_sorted
from gen._core._pauli_map import PauliMap
from gen._core._tile import Tile
from gen._core._flow import Flow
from gen._core._flow_semi_auto import FlowSemiAuto

_SWAP_CONJUGATED_MAP = {"XCZ": "CX", "YCZ": "CY", "YCX": "XCY", "SWAPCX": "CXSWAP"}

if TYPE_CHECKING:
    from gen._chunk import Chunk, ChunkSemiAuto


class ChunkBuilder:
    """Helper class for building stim circuits.

    Handles qubit indexing (complex -> int).
    Handles measurement tracking (naming results and referring to them by name).
    """

    def __init__(
        self,
        allowed_qubits: Iterable[complex] | None = None,
        *,
        unknown_qubit_append_mode: Literal["auto", "error", "skip", "include"] = "auto",
    ):
        """Creates a Builder for creating a circuit over the given qubits.

        Args:
            allowed_qubits: Defaults to None (everything allowed). Specifies the qubit positions
                that the circuit is permitted to contain.
            unknown_qubit_append_mode: Defaults to 'auto'. The available options are:
                - 'error': When a qubit position outside `allowed_qubits` is encountered,
                    raise an exception.
                - 'include': When a qubit position outside `allowed_qubits` is encountered,
                    automatically add it into `builder.q2i` and `builder.allowed_qubits`.
                - 'skip': When a qubit position outside `allowed_qubits` is encountered,
                    ignore it. Note that, for two-qubit and multi-qubit operations, this
                    will ignore the pair or group of targets containing the skipped position.
                - 'auto':  Replace by 'include' if allowed_qubits is None else 'error'.
        """
        if unknown_qubit_append_mode == "auto":
            unknown_qubit_append_mode = "include" if allowed_qubits is None else "error"
        self._num_measurements = 0
        self._recorded_measurements: dict[Any, list[int]] = {}
        self.circuit = stim.Circuit()
        self.unknown_qubit_append_mode: Literal["error", "skip", "include"] = cast(
            Any, unknown_qubit_append_mode
        )
        self.q2i: dict[complex, int] = {}
        self.flows: list[Flow | FlowSemiAuto] = []
        self.discarded_output_flows: list[PauliMap] = []
        self.discarded_input_flows: list[PauliMap] = []
        if allowed_qubits is not None:
            for i, q in enumerate(sorted_complex(allowed_qubits)):
                self.q2i[q] = i

        for q, i in self.q2i.items():
            self.circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])

    def _ensure_indices(
        self, qs: Iterable[complex], *, context_gate: Any, context_targets: Any, context_arg: Any
    ) -> bool:
        missing = [q for q in qs if q not in self.q2i]
        if not missing:
            return True

        bad_types = [q for q in missing if not isinstance(q, (int, float, complex))]
        if bad_types:
            raise ValueError(
                f"Expected qubit positions (an int, float, or complex), "
                f"but got {bad_types[0]!r}.\n"
                f"    gate={context_gate!r}\n"
                f"    targets={context_targets!r}\n"
                f"    arg={context_arg!r}"
            )

        if self.unknown_qubit_append_mode == "error":
            raise KeyError(
                f"The qubit positions {missing!r} aren't in `builder.allowed_qubits`, but "
                f"`builder.unknown_qubit_append_mode` is {self.unknown_qubit_append_mode!r}"
            )
        elif self.unknown_qubit_append_mode == "include":
            for q in missing:
                i = len(self.q2i)
                self.q2i[q] = i
                self.circuit.insert(
                    0, stim.CircuitInstruction("QUBIT_COORDS", [i], [q.real, q.imag])
                )
        elif self.unknown_qubit_append_mode == "skip":
            pass
        else:
            raise NotImplementedError(f"{self.unknown_qubit_append_mode=}")
        return False

    def _rec(self, key: Any, value: list[int]) -> None:
        if key in self._recorded_measurements:
            raise ValueError(
                f"Attempted to record a measurement for {key=}, but the key is already used."
            )
        self._recorded_measurements[key] = value

    def record_measurement_group(self, sub_keys: Iterable[Any], *, key: Any) -> None:
        """Combines multiple measurement keys into one key.

        Args:
            sub_keys: The measurement keys to combine.
            key: Where to store the combined result.
        """
        self._rec(key, self.lookup_mids(sub_keys))

    def lookup_mids(self, keys: Iterable[Any], *, ignore_unmatched: bool = False) -> list[int]:
        """Looks up measurement indices by key.

        Measurement keys are created automatically when appending measurement operations into the
        circuit via the builder's append method. They are also created manually by methods like
        `builder.record_measurement_group`.

        Args:
            keys: The measurement keys to lookup.
            ignore_unmatched: Defaults to False. If set to True, keys that don't correspond
                to measurements are ignored instead of raising an error.

        Returns:
            A list of offsets indicating when the measurements occurred.
        """
        result: list[int] = []
        missing: list[Any] = []
        for key in keys:
            recs = self._recorded_measurements.get(key)
            if recs is None:
                missing.append(key)
            else:
                result.extend(recs)
        if missing and not ignore_unmatched:
            raise ValueError(
                "Some of the given measurement record keys don't exist.\n"
                f"Unmatched keys: {missing!r}\n"
                f"Given keys: {list(keys)!r}"
            )
        return xor_sorted(result)

    def add_discarded_flow_input(self, flow: PauliMap | Tile) -> None:
        if isinstance(flow, Tile):
            flow = flow.to_pauli_map()
        self.discarded_input_flows.append(flow)

    def add_discarded_flow_output(self, flow: PauliMap | Tile) -> None:
        if isinstance(flow, Tile):
            flow = flow.to_pauli_map()
        self.discarded_output_flows.append(flow)

    def add_flow(
        self,
        *,
        start: PauliMap | Tile | Literal["auto"] | None = None,
        end: PauliMap | Tile | Literal["auto"] | None = None,
        ms: Iterable[Any] | Literal["auto"] = (),
        ignore_unmatched_ms: bool = False,
        obs_key: Any = None,
        center: complex | None = None,
        flags: Iterable[str] = frozenset(),
        sign: bool | None = None,
    ) -> None:
        self.flows.append(
            FlowSemiAuto(
                start=start,
                end=end,
                mids=(
                    "auto"
                    if ms == "auto"
                    else self.lookup_mids(ms, ignore_unmatched=ignore_unmatched_ms)
                ),
                obs_key=obs_key,
                center=center,
                flags=flags,
                sign=sign,
            )
        )

    def finish_chunk_keep_semi_auto(
        self,
        *,
        discarded_inputs: Iterable[PauliMap] = (),
        discarded_outputs: Iterable[PauliMap] = (),
        wants_to_merge_with_prev: bool = False,
        wants_to_merge_with_next: bool = False,
    ) -> ChunkSemiAuto:
        from gen._chunk import ChunkSemiAuto

        return ChunkSemiAuto(
            circuit=self.circuit,
            q2i=self.q2i,
            flows=self.flows,
            discarded_inputs=discarded_inputs,
            discarded_outputs=discarded_outputs,
            wants_to_merge_with_next=wants_to_merge_with_next,
            wants_to_merge_with_prev=wants_to_merge_with_prev,
        )

    def finish_chunk(
        self,
        *,
        wants_to_merge_with_prev: bool = False,
        wants_to_merge_with_next: bool = False,
        failure_mode: Literal["error", "ignore", "print"] = "error",
    ) -> Chunk:
        return self.finish_chunk_keep_semi_auto(
            discarded_inputs=self.discarded_input_flows,
            discarded_outputs=self.discarded_output_flows,
            wants_to_merge_with_next=wants_to_merge_with_next,
            wants_to_merge_with_prev=wants_to_merge_with_prev,
        ).solve(failure_mode=failure_mode)

    def append(
        self,
        gate: str,
        targets: Iterable[complex | Sequence[complex] | PauliMap | Tile | Any] = (),
        *,
        arg: float | Iterable[float] | None = None,
        measure_key_func: (
            Callable[[complex], Any]
            | Callable[[tuple[complex, complex]], Any]
            | Callable[[PauliMap | Tile], Any]
            | None
        ) = lambda e: e,
        tag: str = "",
    ) -> None:
        """Appends an instruction to the builder's circuit.

        This method differs from `stim.Circuit.append` in the following ways:

        1) It targets qubits by position instead of by index. Also, it takes two
        qubit targets as pairs instead of interleaved. For example, instead of
        saying

            a = builder.q2i[5 + 1j]
            b = builder.q2i[5]
            c = builder.q2i[0]
            d = builder.q2i[1j]
            builder.circuit.append('CZ', [a, b, c, d])

        you would say

            builder.append('CZ', [(5+1j, 5), (0, 1j)])

        2) It canonicalizes. In particular, it will:
            - Sort targets. For example:
                `H 3 1 2` -> `H 1 2 3`
                `CX 2 3 1 0` -> `CX 1 0 2 3`
                `CZ 2 3 6 0` -> `CZ 0 6 2 3`
            - Replace rare gates with common gates. For example:
                `XCZ 1 2` -> `CX 2 1`
            - Not append target-less gates at all. For example:
                `CX      ` -> ``

            Canonicalization makes the form of the final circuit stable,
            despite things like python's `set` data structure having
            inconsistent iteration orders. This makes the output easier
            to unit test, and more viable to store under source control.

        3) It tracks measurements. When appending a measurement, its index is
        stored in the measurement tracker keyed by the position of the qubit
        being measured (or by a custom key, if `measure_key_func` is specified).
        The indices of the measurements can be looked up later via
        `builder.lookup_recs([key1, key2, ...])`.

        Args:
            gate: The name of the gate to append, such as "H" or "M" or "CX".
            targets: The qubit positions that the gate operates on. For single
                qubit gates like H or M this should be an iterable of complex
                numbers. For two qubit gates like CX or MXX it should be an
                iterable of pairs of complex numbers. For MPP it should be an
                iterable of gen.PauliMap instances.
            arg: Optional. The parens argument or arguments used for the gate
                instruction. For example, for a measurement gate, this is the
                probability of the incorrect result being reported.
            measure_key_func: Customizes the keys used to track the indices of
                measurement results. By default, measurements are keyed by
                position, but thus won't work if a circuit measures the same
                qubit multiple times. This function can transform that position
                into a different value (for example, you might set
                `measure_key_func=lambda pos: (pos, 'first_cycle')` for
                measurements during the first cycle of the circuit.
            tag: Defaults to "" (no tag). A custom tag to attach to the
                instruction(s) appended into the stim circuit.
        """
        __tracebackhide__ = True
        data = stim.gate_data(gate)

        if data.name == "TICK":
            if arg is not None:
                raise ValueError(f"TICK takes no arguments but got {arg=}.")
            if targets:
                raise ValueError(f"TICK takes no targets but got {targets=}.")
            self.circuit.append("TICK", tag=tag)

        elif data.name == "SHIFT_COORDS":
            if arg is None:
                raise ValueError(f"SHIFT_COORDS expects {arg=} to not be None.")
            if targets:
                raise ValueError(f"SHIFT_COORDS takes no targets but got {targets=}.")
            self.circuit.append("SHIFT_COORDS", [], arg, tag=tag)

        elif data.name == "DETECTOR" or data.name == "OBSERVABLE_INCLUDE":
            t0 = self._num_measurements
            times = self.lookup_mids(targets)
            rec_targets = [stim.target_rec(t - t0) for t in sorted(times)]
            self.circuit.append(data.name, rec_targets, arg, tag=tag)

        elif data.name == "MPP":
            self._append_mpp(
                gate=gate,
                targets=cast(Any, targets),
                arg=arg,
                measure_key_func=cast(Any, measure_key_func),
                tag=tag,
            )

        elif data.name == "E":
            targets = list(targets)
            if len(targets) != 1 or not isinstance(targets[0], PauliMap):
                raise NotImplementedError(
                    "gate='CORRELATED_ERROR' "
                    "and len(targets) != 1 "
                    "and not isinstance(targets[0], gen.PauliMap)"
                )
            if arg:
                qs = sorted_complex(targets[0].keys())
                if self._ensure_indices(
                    qs, context_gate=gate, context_targets=targets, context_arg=arg
                ):
                    stim_targets = []
                    for q in qs:
                        i = self.q2i[q]
                        stim_targets.append(stim.target_pauli(i, targets[0][q]))
                    self.circuit.append("CORRELATED_ERROR", stim_targets, arg, tag=tag)

        elif data.is_two_qubit_gate:
            self._append_2q(
                gate=gate,
                data=data,
                targets=cast(Any, targets),
                arg=arg,
                measure_key_func=cast(Any, measure_key_func),
                tag=tag,
            )

        elif data.is_single_qubit_gate:
            self._append_1q(
                gate=gate,
                data=data,
                targets=cast(Any, targets),
                arg=arg,
                measure_key_func=cast(Any, measure_key_func),
                tag=tag,
            )

        else:
            raise NotImplementedError(f"{gate=}")

    def _append_mpp(
        self,
        *,
        gate: str,
        targets: PauliMap | Tile | Iterable[PauliMap | Tile],
        arg: float | Iterable[float] | None = None,
        measure_key_func: Callable[[PauliMap | Tile], Any] | None,
        tag: str,
    ) -> None:
        if not targets:
            return
        if arg == 0:
            arg = None
        if isinstance(targets, (PauliMap, Tile)):
            raise ValueError(
                f"{gate=} but {targets=} is a single gen.PauliMap instead of a list of "
                f"gen.PauliMap."
            )
        for target in targets:
            if not isinstance(target, (PauliMap, Tile)):
                raise ValueError(f"{gate=} but {target=} isn't a " f"gen.PauliMap, or gen.Tile.")

        # Canonicalize qubit ordering of the pauli strings.
        stim_targets = []
        for target in targets:
            pauli_map: PauliMap = PauliMap(target)
            if not pauli_map:
                raise NotImplementedError(f"Attempted to measure empty pauli string {pauli_map=}.")
            qs = sorted_complex(pauli_map)
            if self._ensure_indices(
                qs, context_gate=gate, context_targets=targets, context_arg=arg
            ):
                for q in qs:
                    i = self.q2i[q]
                    stim_targets.append(stim.target_pauli(i, pauli_map[q]))
                    stim_targets.append(stim.target_combiner())
                stim_targets.pop()

        self.circuit.append(gate, stim_targets, arg, tag=tag)

        for target in targets:
            if measure_key_func is not None:
                self._rec(measure_key_func(cast(Any, target)), [self._num_measurements])
            self._num_measurements += 1

    def _append_1q(
        self,
        *,
        gate: str,
        data: stim.GateData,
        targets: Iterable[complex],
        arg: float | Iterable[float] | None,
        measure_key_func: Callable[[complex], Any] | None,
        tag: str,
    ) -> None:
        __tracebackhide__ = True
        targets = tuple(targets)
        self._ensure_indices(targets, context_gate=gate, context_targets=targets, context_arg=arg)
        indices: list[tuple[int, int]] = []
        for k in range(len(targets)):
            i = self.q2i.get(targets[k])
            if i is not None:
                indices.append((i, k))
        indices = sorted(indices)
        if not indices:
            return

        self.circuit.append(gate, [e[0] for e in indices], arg, tag=tag)
        if data.produces_measurements:
            for _, k in indices:
                t = targets[k]
                if measure_key_func is not None:
                    self._rec(measure_key_func(t), [self._num_measurements])
                self._num_measurements += 1

    def _append_2q(
        self,
        *,
        gate: str,
        data: stim.GateData,
        targets: Iterable[Sequence[complex]],
        arg: float | Iterable[float] | None,
        measure_key_func: Callable[[tuple[complex, complex]], Any] | None,
        tag: str,
    ) -> None:
        __tracebackhide__ = True

        for target in targets:
            if not hasattr(target, "__len__") or len(target) != 2:
                raise ValueError(
                    f"{gate=} is a two-qubit gate, "
                    f"but {target=} isn't a pair of complex numbers."
                )
            a, b = cast(Any, target)
            self._ensure_indices(
                (a, b), context_gate=gate, context_targets=targets, context_arg=arg
            )

        # Canonicalize gate and target pairs.
        targets = [tuple(cast(Any, pair)) for pair in targets]
        index_pairs: list[tuple[int, int, int]] = []
        index_swapped = data.name in _SWAP_CONJUGATED_MAP
        index_sorted = data.is_symmetric_gate
        for k in range(len(targets)):
            a, b = targets[k]
            ai = self.q2i.get(a)
            bi = self.q2i.get(b)
            if ai is not None and bi is not None:
                if index_swapped or (index_sorted and ai > bi):
                    ai, bi = bi, ai
                index_pairs.append((ai, bi, k))
        index_pairs = sorted(index_pairs)
        if not index_pairs:
            return

        if index_swapped:
            gate = _SWAP_CONJUGATED_MAP[data.name]

        self.circuit.append(gate, [i for pair in index_pairs for i in pair[:2]], arg, tag=tag)

        # Record both qubit orderings.
        if data.produces_measurements:
            for _, _, k in index_pairs:
                a, b = targets[k]
                if measure_key_func is not None:
                    k1 = measure_key_func((a, b))
                    k2 = measure_key_func((b, a))
                    self._rec(k1, [self._num_measurements])
                    if k1 != k2:
                        self._rec(k2, [self._num_measurements])
                self._num_measurements += 1

    def demolition_measure_with_feedback_passthrough(
        self,
        xs: Iterable[complex] = (),
        ys: Iterable[complex] = (),
        zs: Iterable[complex] = (),
        *,
        measure_key_func: Callable[[complex], Any] = lambda e: e,
    ) -> None:
        """Performs demolition measurements that look like measurements w.r.t. detectors.

        This is done by adding feedback operations that flip the demolished qubits depending
        on the measurement result. This feedback can then later be removed using
        stim.Circuit.with_inlined_feedback. The benefit is that it can be easier to
        programmatically create the detectors using the passthrough measurements, and
        then they can be automatically converted.
        """
        self.append("MX", xs, measure_key_func=measure_key_func)
        self.append("MY", ys, measure_key_func=measure_key_func)
        self.append("MZ", zs, measure_key_func=measure_key_func)
        self.append("TICK")
        self.append("RX", xs)
        self.append("RY", ys)
        self.append("RZ", zs)
        for qs, b in [(xs, "Z"), (ys, "X"), (zs, "X")]:
            for q in qs:
                self.classical_paulis(control_keys=[measure_key_func(q)], targets=[q], basis=b)

    def classical_paulis(
        self, *, control_keys: Iterable[Any], targets: Iterable[complex], basis: str
    ) -> None:
        """Appends the tensor product of the given controls and targets into the circuit."""
        gate = f"C{basis}"
        targets = tuple(targets)
        self._ensure_indices(
            targets, context_targets=targets, context_gate=f"classical C{basis}", context_arg=None
        )
        indices: list[int] = []
        for t in targets:
            i = self.q2i.get(t)
            if i is not None:
                indices.append(i)
        indices = sorted(indices)
        t0 = self._num_measurements
        times = self.lookup_mids(control_keys)
        rec_targets = [stim.target_rec(t - t0) for t in sorted(times)]
        for rec in rec_targets:
            for i in indices:
                self.circuit.append(gate, [rec, i])
