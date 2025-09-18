from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import stim

from gen._core._pauli_map import PauliMap
from gen._core._tile import Tile
from gen._core._complex_util import xor_sorted


class Flow:
    """A rule for how a stabilizer travels into, through, and/or out of a circuit."""

    def __init__(
        self,
        *,
        start: PauliMap | Tile | None = None,
        end: PauliMap | Tile | None = None,
        mids: Iterable[int] = (),
        obs_key: Any = None,
        center: complex | None = None,
        flags: Iterable[str] = frozenset(),
        sign: bool | None = None,
    ):
        """Initializes a Flow.

        Args:
            start: Defaults to None (empty). The Pauli product operator at the beginning of the
                circuit (before *all* operations, including resets).
            end: Defaults to None (empty). The Pauli product operator at the end of the
                circuit (after *all* operations, including measurements).
            mids: Defaults to empty. Indices of measurements that mediate the flow (that multiply
                into it as it traverses the circuit).
            center: Defaults to None (unspecified). Specifies a 2d coordinate to use in metadata
                when the flow is completed into a detector. Incompatible with obs_key.
            obs_key: Defaults to None (detector flow). Identifies that this is an observable flow
                (instead of a detector flow) and gives a name that be used when linking chunks.
            flags: Defaults to empty. Custom information about the flow, that can be used by code
                operating on chunks for a variety of purposes. For example, this could identify the
                "color" of the flow in a color code.
            sign: Defaults to None (unsigned). The expected sign of the flow.
        """
        if start == "auto":
            raise ValueError(f"gen.Flow no longer supports {start=}. Use gen.FlowSemiAuto instead.")
        if end == "auto":
            raise ValueError(f"gen.Flow no longer supports {end=}. Use gen.FlowSemiAuto instead.")
        if mids == "auto":
            raise ValueError(f"gen.Flow no longer supports {mids=}. Use gen.FlowSemiAuto instead.")
        if obs_key is None and center is None:
            if isinstance(start, Tile) and start.measure_qubit is not None:
                center = start.measure_qubit
            if isinstance(end, Tile) and end.measure_qubit is not None:
                center = end.measure_qubit
        if obs_key is None and center is None:
            qubits: list[complex] = []
            if isinstance(start, PauliMap):
                qubits.extend(start.keys())
            if isinstance(end, PauliMap):
                qubits.extend(end.keys())
            if isinstance(start, Tile):
                qubits.extend(start.data_set)
            if isinstance(end, Tile):
                qubits.extend(end.data_set)
            center = sum(qubits) / (len(qubits) or 1)
        if isinstance(flags, str):
            raise TypeError(f"{flags=} is a str instead of a set")
        if obs_key is None and isinstance(start, PauliMap) and start.name is not None:
            obs_key = start.name
        if obs_key is None and isinstance(end, PauliMap) and end.name is not None:
            obs_key = end.name
        if isinstance(start, PauliMap) and start.name is not None:
            assert obs_key == start.name
            start = start.with_name(None)
        if isinstance(end, PauliMap) and end.name is not None:
            assert obs_key == end.name
            end = end.with_name(None)

        if start is not None and not isinstance(start, (PauliMap, Tile)):
            raise ValueError(
                f"{start=} is not None and not isinstance(start, (gen.PauliMap, gen.Tile))"
            )
        if end is not None and not isinstance(end, (PauliMap, Tile)):
            raise ValueError(
                f"{end=} is not None and not isinstance(end, (gen.PauliMap, gen.Tile))"
            )
        if isinstance(start, Tile):
            start = start.to_pauli_map()
        elif start is None:
            start = PauliMap()
        if isinstance(end, Tile):
            end = end.to_pauli_map()
        elif end is None:
            end = PauliMap()
        self.start: PauliMap = start
        self.end: PauliMap = end
        self.measurement_indices: tuple[int, ...] = tuple(xor_sorted(mids))
        self.flags: frozenset[str] = frozenset(flags)
        self.obs_key: Any = obs_key
        self.center: complex | None = center
        self.sign: bool | None = sign

    def to_stim_flow(
        self, *, q2i: dict[complex, int], o2i: dict[Any, int] | None = None
    ) -> stim.Flow:
        out = self.end.to_stim_pauli_string(q2i)
        if self.sign:
            out.sign = -1
        return stim.Flow(
            input=self.start.to_stim_pauli_string(q2i),
            output=out,
            measurements=self.measurement_indices,
            included_observables=None if self.obs_key is None else [o2i[self.obs_key]],
        )

    @property
    def key_start(self) -> PauliMap:
        return self.start.with_name(self.obs_key)

    @property
    def key_end(self) -> PauliMap:
        return self.end.with_name(self.obs_key)

    def with_edits(
        self,
        *,
        start: PauliMap | None = None,
        end: PauliMap | None = None,
        measurement_indices: Iterable[int] | None = None,
        obs_key: Any = "__not_specified!!",
        center: complex | None = None,
        flags: Iterable[str] | None = None,
        sign: Any = "__not_specified!!",
    ) -> Flow:
        return Flow(
            start=self.start if start is None else start,
            end=self.end if end is None else end,
            mids=(
                self.measurement_indices
                if measurement_indices is None
                else cast(Any, measurement_indices)
            ),
            obs_key=self.obs_key if obs_key == "__not_specified!!" else obs_key,
            center=self.center if center is None else center,
            flags=self.flags if flags is None else flags,
            sign=self.sign if sign == "__not_specified!!" else sign,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Flow):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.measurement_indices == other.measurement_indices
            and self.obs_key == other.obs_key
            and self.flags == other.flags
            and self.center == other.center
            and self.sign == other.sign
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.start,
                self.end,
                self.measurement_indices,
                self.obs_key,
                self.flags,
                self.center,
                self.sign,
            )
        )

    def __str__(self) -> str:
        q: Any

        start_terms = []
        for q, p in self.start.items():
            q = complex(q)
            if q.real == 0:
                q = "0+" + str(q)
            q = str(q).replace("(", "").replace(")", "")
            start_terms.append(f"{p}[{q}]")

        end_terms = []
        for q, p in self.end.items():
            q = complex(q)
            if q.real == 0:
                q = "0+" + str(q)
            q = str(q).replace("(", "").replace(")", "")
            end_terms.append(f"{p}[{q}]")

        for m in self.measurement_indices:
            end_terms.append(f"rec[{m}]")

        if not start_terms:
            start_terms.append("1")
        if not end_terms:
            end_terms.append("1")

        key = "" if self.obs_key is None else f" (obs={self.obs_key})"
        result = f'{"*".join(start_terms)} -> {"*".join(end_terms)}{key}'
        if self.sign is None:
            pass
        elif self.sign:
            result = "-" + result
        else:
            result = "+" + result
        if self.flags:
            result += f" (flags={sorted(self.flags)})"
        return result

    def __repr__(self):
        return (
            f"gen.Flow(start={self.start!r}, "
            f"end={self.end!r}, "
            f"measurement_indices={self.measurement_indices!r}, "
            f"flags={sorted(self.flags)}, "
            f"obs_key={self.obs_key!r}, "
            f"center={self.center!r}, "
            f"sign={self.sign!r}"
        )

    def with_xz_flipped(self) -> Flow:
        return self.with_edits(start=self.start.with_xz_flipped(), end=self.end.with_xz_flipped())

    def with_transformed_coords(self, transform: Callable[[complex], complex]) -> Flow:
        return self.with_edits(
            start=self.start.with_transformed_coords(transform),
            end=self.end.with_transformed_coords(transform),
            center=None if self.center is None else transform(self.center),
        )

    def fuse_with_next_flow(self, next_flow: Flow, *, next_flow_measure_offset: int) -> Flow:
        if next_flow.start != self.end:
            raise ValueError("other.start != self.end")
        if next_flow.obs_key != self.obs_key:
            raise ValueError("other.obs_key != self.obs_key")
        if self.center is None:
            new_center = next_flow.center
        elif next_flow.center is None:
            new_center = self.center
        else:
            new_center = (self.center + next_flow.center) / 2
        assert isinstance(self.measurement_indices, tuple)
        assert isinstance(next_flow.measurement_indices, tuple)
        return Flow(
            start=self.start,
            end=next_flow.end,
            center=new_center,
            mids=(
                *(m + next_flow_measure_offset * (m < 0) for m in self.measurement_indices),
                *(m + next_flow_measure_offset for m in next_flow.measurement_indices),
            ),
            obs_key=self.obs_key,
            flags=self.flags | next_flow.flags,
            sign=(
                None if self.sign is None or next_flow.sign is None else self.sign ^ next_flow.sign
            ),
        )

    def __mul__(self, other: Flow) -> Flow:
        """Merges two flows together field-by-field."""
        if other.obs_key != self.obs_key:
            raise ValueError("other.obs_key != self.obs_key")

        new_start: PauliMap = self.start * other.start
        new_end: PauliMap = self.end * other.end
        new_center: complex | None
        if self.center is not None and other.center is not None:
            new_center = (self.center + other.center) / 2
        elif self.center is not None:
            new_center = self.center
        elif other.center is not None:
            new_center = other.center
        else:
            new_center = None

        return Flow(
            start=new_start,
            end=new_end,
            mids=xor_sorted(self.measurement_indices + other.measurement_indices),
            obs_key=self.obs_key,
            flags=self.flags | other.flags,
            center=new_center,
            sign=(None if self.sign is None or other.sign is None else self.sign ^ other.sign),
        )
