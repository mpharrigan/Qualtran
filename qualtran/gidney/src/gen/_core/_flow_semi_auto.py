from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast, Literal

import stim

from gen._core._flow import Flow
from gen._core._pauli_map import PauliMap
from gen._core._tile import Tile
from gen._core._complex_util import xor_sorted


class FlowSemiAuto:
    """A rule for how a stabilizer travels into, through, and/or out of a circuit."""

    def __init__(
        self,
        *,
        start: PauliMap | Tile | Literal["auto"] | None = None,
        end: PauliMap | Tile | Literal["auto"] | None = None,
        mids: Iterable[int] | Literal["auto"] = (),
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
            center: Defaults to None (auto). Specifies a 2d coordinate to use in metadata when the
                flow is completed into a detector. Incompatible with obs_key. Derived automatically
                when not specified.
            obs_key: Defaults to None (detector flow). Identifies that this is an observable flow
                (instead of a detector flow) and gives a name that be used when linking chunks.
            flags: Defaults to empty. Custom information about the flow, that can be used by code
                operating on chunks for a variety of purposes. For example, this could identify the
                "color" of the flow in a color code.
            sign: Defaults to None (unsigned). The expected sign of the flow.
        """
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

        if start is not None and start != "auto" and not isinstance(start, (PauliMap, Tile)):
            raise ValueError(
                f"{start=} is not None and start != 'auto' and "
                f"not isinstance(start, (gen.PauliMap, gen.Tile))"
            )
        if end is not None and end != "auto" and not isinstance(end, (PauliMap, Tile)):
            raise ValueError(
                f"{end=} is not None and end != 'auto' and "
                f"not isinstance(end, (gen.PauliMap, gen.Tile))"
            )
        if isinstance(start, Tile):
            start = start.to_pauli_map()
        elif start is None:
            start = PauliMap()
        if isinstance(end, Tile):
            end = end.to_pauli_map()
        elif end is None:
            end = PauliMap()
        self.start: PauliMap | Literal["auto"] = start
        self.end: PauliMap | Literal["auto"] = end
        self.measurement_indices: tuple[int, ...] | Literal["auto"] = cast(
            Any, (mids if mids == "auto" else tuple(xor_sorted(cast(Any, mids))))
        )
        self.flags: frozenset[str] = frozenset(flags)
        self.obs_key: Any = obs_key
        self.center: complex | None = center
        self.sign: bool | None = sign
        if mids == "auto" and not start and not end:
            raise ValueError("measurement_indices == 'auto' and not start and not end")

    def to_stim_flow(
        self, *, q2i: dict[complex, int], o2i: dict[Any, int] | None = None
    ) -> stim.Flow:
        inp = None if self.start == "auto" else self.start.to_stim_pauli_string(q2i)
        out = None if self.end == "auto" else self.end.to_stim_pauli_string(q2i)
        if self.sign:
            if out is not None:
                out.sign = -1
            elif inp is not None:
                inp.sign = -1
        return stim.Flow(
            input=self.start.to_stim_pauli_string(q2i),
            output=out,
            measurements=(
                None if self.measurement_indices == "auto" else cast(Any, self.measurement_indices)
            ),
            included_observables=None if self.obs_key is None else [o2i[self.obs_key]],
        )

    def with_edits(
        self,
        *,
        start: PauliMap | Literal["auto"] | None = None,
        end: PauliMap | Literal["auto"] | None = None,
        measurement_indices: Iterable[int] | Literal["auto"] | None = None,
        obs_key: Any = "__not_specified!!",
        center: complex | None = None,
        flags: Iterable[str] | None = None,
        sign: Any = "__not_specified!!",
    ) -> FlowSemiAuto:
        return FlowSemiAuto(
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

    def to_flow(self) -> Flow:
        """Converts the solved FlowSemiAuto to a Flow.

        If there are still 'auto' fields present, the conversion fails.
        """
        if self.start == "auto":
            raise ValueError(f"Can't convert to a non-semi-auto flow because {self.start=}.")
        if self.end == "auto":
            raise ValueError(f"Can't convert to a non-semi-auto flow because {self.end=}.")
        if self.measurement_indices == "auto":
            raise ValueError(
                f"Can't convert to a non-semi-auto flow because {self.measurement_indices=}."
            )
        return Flow(
            start=cast(Any, self.start),
            end=cast(Any, self.end),
            mids=cast(Any, self.measurement_indices),
            obs_key=self.obs_key,
            center=self.center,
            flags=self.flags,
            sign=self.sign,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FlowSemiAuto):
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

    def __str__(self) -> str:
        q: Any

        start_terms = []
        if self.start == "auto":
            start_terms.append("auto")
        else:
            for q, p in cast(PauliMap, self.start).items():
                start_terms.append(f"{p}[{q}]")

        end_terms = []
        if self.end == "auto":
            end_terms.append("auto")
        else:
            for q, p in cast(PauliMap, self.end).items():
                q = complex(q)
                if q.real == 0:
                    q = "0+" + str(q)
                q = str(q).replace("(", "").replace(")", "")
                end_terms.append(f"{p}[{q}]")

        if self.measurement_indices == "auto":
            end_terms.append("rec[auto]")
        else:
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
            f"gen.FlowSemiAuto(start={self.start!r}, "
            f"end={self.end!r}, "
            f"measurement_indices={self.measurement_indices!r}, "
            f"flags={sorted(self.flags)}, "
            f"obs_key={self.obs_key!r}, "
            f"center={self.center!r}, "
            f"sign={self.sign!r}"
        )
