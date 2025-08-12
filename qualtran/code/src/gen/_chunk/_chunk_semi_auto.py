from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TYPE_CHECKING, Any

import stim

from gen._chunk._chunk import Chunk
from gen._chunk._flow_util import solve_semi_auto_flows
from gen._core import PauliMap, Tile, Flow, FlowSemiAuto

if TYPE_CHECKING:
    pass


class ChunkSemiAuto:
    """A variant of `gen.Chunk` that supports partially specified flows.

    Use `gen.ChunkSemiAuto.solve()` to solve the partially specified flows and
    return a solved `gen.Chunk`.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        flows: Iterable[FlowSemiAuto | Flow],
        discarded_inputs: Iterable[PauliMap | Tile] = (),
        discarded_outputs: Iterable[PauliMap | Tile] = (),
        wants_to_merge_with_next: bool = False,
        wants_to_merge_with_prev: bool = False,
        q2i: dict[complex, int] | None = None,
        o2i: dict[Any, int] | None = None,
    ):
        """

        Args:
            circuit: The circuit implementing the chunk's functionality.
            flows: A series of stabilizer flows that the circuit implements.
            discarded_inputs: Explicitly rejected in flows. For example, a data
                measurement chunk might reject flows for stabilizers from the
                anticommuting basis. If they are not rejected, then compilation
                will fail when attempting to combine this chunk with a preceding
                chunk that has those stabilizers from the anticommuting basis
                flowing out.
            discarded_outputs: Explicitly rejected out flows. For example, an
                initialization chunk might reject flows for stabilizers from the
                anticommuting basis. If they are not rejected, then compilation
                will fail when attempting to combine this chunk with a following
                chunk that has those stabilizers from the anticommuting basis
                flowing in.
            wants_to_merge_with_next: Defaults to False. When set to True,
                the chunk compiler won't insert a TICK between this chunk
                and the next chunk.
            wants_to_merge_with_prev: Defaults to False. When set to True,
                the chunk compiler won't insert a TICK between this chunk
                and the previous chunk.
            q2i: Defaults to None (infer from QUBIT_COORDS instructions in circuit else
                raise an exception). The gen-qubit-coordinate-to-stim-qubit-index mapping
                used to translate between gen's qubit keys and stim's qubit keys.
            o2i: Defaults to None (raise an exception if observables present in circuit).
                The gen-observable-key-to-stim-observable-index mapping used to translate
                between gen's observable keys and stim's observable keys.
        """

        if q2i is None:
            q2i = {x + 1j * y: i for i, (x, y) in circuit.get_final_qubit_coordinates().items()}
            if len(q2i) != circuit.num_qubits:
                raise ValueError(
                    "The given circuit doesn't have enough `QUBIT_COORDS` instructions to "
                    "determine the gen-coordinate-to-stim-qubit-index mapping. You must manually "
                    "specify it by passing a `q2i={...}` argument, or add the missing "
                    "`QUBIT_COORDS`."
                )
        if o2i is None:
            if circuit.num_observables:
                raise ValueError(
                    "The given circuit has `OBSERVABLE_INCLUDE` instructions. You must specify "
                    "the gen-observable-key-to-stim-observable-index mapping by passing an"
                    "`o2k={...}` argument."
                )
            o2i = {}
            for flow in flows:
                if flow.obs_key is not None and flow.obs_key not in o2i:
                    o2i[flow.obs_key] = len(o2i)

        self.q2i: dict[complex, int] = q2i
        self.o2i: dict[Any, int] = o2i
        self.circuit: stim.Circuit = circuit
        self.flows: tuple[Flow | FlowSemiAuto, ...] = tuple(flows)
        self.discarded_inputs: tuple[PauliMap, ...] = tuple(
            e.to_pauli_map() if isinstance(e, Tile) else e for e in discarded_inputs
        )
        self.discarded_outputs: tuple[PauliMap, ...] = tuple(
            e.to_pauli_map() if isinstance(e, Tile) else e for e in discarded_outputs
        )
        self.wants_to_merge_with_next = wants_to_merge_with_next
        self.wants_to_merge_with_prev = wants_to_merge_with_prev
        assert all(isinstance(e, PauliMap) for e in self.discarded_inputs)
        assert all(isinstance(e, PauliMap) for e in self.discarded_outputs)

    def solve(self, *, failure_mode: Literal["error", "ignore", "print"] = "error") -> Chunk:
        """Solves any partially specified flows, and returns a `gen.Chunk` with the solution."""
        failure_out = []
        solved_flows = solve_semi_auto_flows(
            flows=self.flows,
            circuit=self.circuit,
            q2i=self.q2i,
            o2i=self.o2i,
            failure_mode=failure_mode,
            failure_out=failure_out,
        )
        if failure_out and failure_mode == "print":
            circuit = self.circuit.copy()
            circuit.insert(0, stim.CircuitInstruction("TICK"))
            circuit.append(stim.CircuitInstruction("TICK"))
            for flow in failure_out:
                if flow.start != "auto" and flow.start:
                    circuit.insert(
                        0,
                        stim.CircuitInstruction(
                            "CORRELATED_ERROR",
                            [stim.target_pauli(self.q2i[q], p) for q, p in flow.start.items()],
                            [0],
                            tag="BAD-FLOW",
                        ),
                    )
                if flow.end != "auto" and flow.end:
                    circuit.append(
                        stim.CircuitInstruction(
                            "CORRELATED_ERROR",
                            [stim.target_pauli(self.q2i[q], p) for q, p in flow.end.items()],
                            [0],
                            tag="BAD-FLOW",
                        )
                    )
        else:
            circuit = self.circuit
        return Chunk(
            circuit=circuit,
            q2i=self.q2i,
            flows=solved_flows,
            discarded_inputs=self.discarded_inputs,
            discarded_outputs=self.discarded_outputs,
            wants_to_merge_with_next=self.wants_to_merge_with_next,
            wants_to_merge_with_prev=self.wants_to_merge_with_prev,
        )
