from __future__ import annotations

import collections
import dataclasses
import pathlib
import random
import sys
from typing import Iterable, Literal, Callable, Any

from facto.operations.vec_sim import VecSim


@dataclasses.dataclass(frozen=True)
class CircuitProgram:
    instructions: tuple[CircuitInstruction, ...]

    def to_quirk_link(self, latex_escape: bool = False) -> str:
        def add_col():
            cols.append([1] * 16)

        def ensure_unmodified_col():
            if any(e in ["•", "zpar", "xpar", "ypar"] for e in cols[-1]):
                add_col()

        def ensure_empty_col():
            if any(e != 1 for e in cols[-1]):
                add_col()

        def ensure_space(inst: CircuitInstruction):
            for q, c in inst.qubit_terms:
                for k in range(len(c)):
                    if cols[-1][q + k] != 1:
                        add_col()
                        return

        cols: list[list[Any]] = []
        init: list[Any] = [0] * 16
        labels = []
        add_col()
        d_par_control = {"@": "zpar", "X": "xpar", "Y": "ypar", "Z": "zpar"}
        d_target = {"X": "X", "Y": "Y", "Z": "Z", "?": "Z", "H": "H"}
        for instruction in self.instructions:
            if instruction.name == "input":
                ((q, c),) = instruction.qubit_terms
                if c == "|CCZ⟩":
                    ensure_empty_col()
                    init[q + 0] = "+"
                    init[q + 1] = "+"
                    init[q + 2] = "+"
                    cols[-1][q + 0] = "•"
                    cols[-1][q + 1] = "•"
                    cols[-1][q + 2] = "Z"
                    ensure_empty_col()
                elif c == "|i⟩":
                    init[q] = "i"
        for instruction in self.instructions:
            if instruction.name == "input":
                ((q, c),) = instruction.qubit_terms
                if c != "|CCZ⟩" and c != "|i⟩":
                    ensure_space(instruction)
                    if c not in labels:
                        labels.append(c)
                    cols[-1][q] = f"~lbl{labels.index(c)}"
            elif instruction.name == "output":
                ((q, c),) = instruction.qubit_terms
                ensure_space(instruction)
                if c not in labels:
                    labels.append(c)
                cols[-1][q] = f"~lbl{labels.index(c)}"
            elif instruction.name == "tick":
                pass
            elif instruction.name == "halftick":
                pass
            elif instruction.name == "drop":
                pass
            elif instruction.name == "annotate":
                pass
            elif instruction.name == "highlight_box_start":
                pass
            elif instruction.name == "highlight_box_end":
                pass
            elif instruction.name == "M":
                ensure_space(instruction)
                ensure_unmodified_col()
                for q, c in instruction.qubit_terms:
                    if c != "Z":
                        add_col()
                        break
                for q, c in instruction.qubit_terms:
                    cols[-1][q] = "Measure"
                    if c == "Z":
                        pass
                    elif c == "X":
                        cols[-2][q] = "H"
                    elif c == "Y":
                        cols[-2][q] = "X^½"
                    else:
                        raise NotImplementedError(f"{instruction=}")
            elif instruction.name == "C" and len(instruction.qubit_terms) == 1:
                ensure_space(instruction)
                ensure_unmodified_col()
                ((q, c),) = instruction.qubit_terms
                for k in range(len(c)):
                    cols[-1][k + q] = d_target[c[k]]
            elif instruction.name == "C" and len(instruction.qubit_terms) > 1:
                ensure_empty_col()

                def priority(term: tuple[int, str]) -> int:
                    count = sum(e != "_" for e in term[1])
                    if "H" in term[1]:
                        return -100
                    if "@" in term[1]:
                        return +100
                    if "X" in term[1]:
                        return -10
                    if "?" in term[1]:
                        return -1
                    return count

                terms = sorted(instruction.qubit_terms, key=priority)
                q1, c1 = terms[0]
                q2, c2 = terms[1]
                for k in range(len(c1)):
                    if c1[k] != "_":
                        cols[-1][k + q1] = d_target[c1[k]]
                count2 = sum(e != "_" for e in c2)
                for k in range(len(c2)):
                    if c2[k] != "_":
                        if count2 == 1 and c2[k] in "@CZ" and len(terms) > 2:
                            cols[-1][k + q2] = "•"
                        else:
                            cols[-1][k + q2] = d_par_control[c2[k]]
                for q, c in terms[2:]:
                    v = [e for e in c if e != "_"]
                    assert v == ["@"] or v == ["Z"] or v == ["C"]
                    for k in range(len(c)):
                        if c[k] != "_":
                            cols[-1][k + q] = "•"

            else:
                raise NotImplementedError(f"{instruction=}")
        parts = []
        parts.append("""{"cols":[""")
        first = True
        for col in cols:
            while col and col[-1] == 1:
                col.pop()
            if first:
                first = False
            else:
                parts.append(",")
            parts.append("[")
            parts.append(",".join("1" if e == 1 else f'"{e}"' for e in col))
            parts.append("]")
        parts.append("]")
        while init and init[-1] == 0:
            init.pop()
        if init:
            parts.append(f',"init":[')
            for k, e in enumerate(init):
                if k > 0:
                    parts.append(",")
                parts.append(str(e) if e in [0, 1] else f'"{e}"')
            parts.append(f"]")
        if labels:
            parts.append(f',"gates":[')
            for k, label in enumerate(labels):
                if k > 0:
                    parts.append(",")
                parts.append(
                    f"""{{"id":"~lbl{k}","name":"{label}","matrix":"{{{{1,0}},{{0,1}}}}"}}"""
                )
            parts.append(f"]")
        parts.append("}")
        unescaped_result = "".join(parts)
        escaped = []
        for c in unescaped_result:
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,":
                escaped.append(c)
            else:
                for b in c.encode("utf8"):
                    escaped.append("%")
                    escaped.append(hex(b)[2:].rjust(2, "0"))
        result = "https://algassert.com/quirk#circuit=" + "".join(escaped)
        if latex_escape:
            result = result.replace("%", "\\%")
            result = result.replace("#", "\\#")
        return result

    @staticmethod
    def from_program_text(program_text: str) -> CircuitProgram:
        vals = []
        for line in program_text.strip().splitlines():
            inst = CircuitInstruction.from_line(line)
            if inst is not None:
                vals.append(inst)
        return CircuitProgram(tuple(vals))

    def verify(
        self,
        *,
        shots: int,
        in2q: dict[str, int],
        out2q: dict[str, int],
        in_func: Callable[[dict[str, bool]], dict[str, bool]],
        out_func: Callable[[dict[str, bool]], dict[str, bool]],
    ):

        for step in range(shots):
            sim = VecSim()
            src1 = {k: bool(random.randrange(2)) for k in in2q.keys()}
            src2 = {k: bool(random.randrange(2)) for k in in2q.keys()}
            in2a = in_func(src1)
            in2b = in_func(src2)
            assert in2a.keys() == in2b.keys()
            out2a = out_func(src1)
            out2b = out_func(src2)
            assert out2a.keys() == out2b.keys()

            for k, q in in2q.items():
                sim.do_qalloc_z(q)
            differences = [k for k in in2a.keys() if in2a[k] != in2b[k]]
            if differences:
                sim.do_h(in2q[differences[0]])
                for q in differences[1:]:
                    sim.do_cx(in2q[differences[0]], in2q[q])
            for k, q in in2q.items():
                if in2a[k]:
                    sim.do_x(q)

            for inst in self.instructions:
                inst.run_in_simulator(sim)
            assert sim.q2i.keys() == set(out2q.values()), sim.q2i.keys() ^ set(out2q.values())

            for k, q in out2q.items():
                if out2a[k]:
                    sim.do_x(q)
            differences = [k for k in out2a.keys() if out2a[k] != out2b[k]]
            if differences:
                for q in differences[1:]:
                    sim.do_cx(out2q[differences[0]], out2q[q])
                sim.do_h(out2q[differences[0]])

            for k in out2a.keys():
                uncomputed = sim.do_mz_discard(out2q[k])
                assert uncomputed == 0, (in2a, in2b, out2a, out2b)


@dataclasses.dataclass(frozen=True)
class CircuitInstruction:
    name: Literal[
        "tick",
        "halftick",
        "highlight_box_start",
        "annotate",
        "highlight_box_end",
        "output",
        "input",
        "bit",
        "drop",
        "C",
        "S",
        "Sd",
        "M",
        "M_debug_force_0",
    ]
    qubit_terms: tuple[tuple[int, str], ...] = ()

    @staticmethod
    def from_line(line: str) -> CircuitInstruction | None:
        line = line.strip()
        if "#" in line:
            line = line[: line.index("#")].strip()
        if not line.strip():
            return None
        if line == "tick":
            return CircuitInstruction("tick")
        if line == "halftick":
            return CircuitInstruction("halftick")
        if line == "highlight_box_start":
            return CircuitInstruction("highlight_box_start")
        if line.startswith("annotate "):
            data = _parse_terms(line[len("annotate ") :])
            return CircuitInstruction("annotate", data)
        if line.startswith("highlight_box_end "):
            q0, label = line[len("highlight_box_end ") :].split(":")
            q0 = int(q0)
            return CircuitInstruction("highlight_box_end", ((q0, label),))
        if line.startswith("output "):
            data = _parse_terms(line[len("output ") :])
            return CircuitInstruction("output", data)
        if line.startswith("drop "):
            data = [int(e) for e in line[len("drop ") :].split()]
            return CircuitInstruction("drop", tuple((e, "") for e in data))
        if line.startswith("input "):
            data = _parse_terms(line[len("input ") :])
            return CircuitInstruction("input", data)
        if line.startswith("bit "):
            data = _parse_terms(line[len("bit ") :])
            return CircuitInstruction("bit", data)
        if line.startswith("C "):
            data = _parse_terms(line[len("C ") :])
            return CircuitInstruction("C", data)
        if line.startswith("S "):
            data = _parse_terms(line[len("S ") :])
            return CircuitInstruction("S", data)
        if line.startswith("Sd "):
            data = _parse_terms(line[len("Sd ") :])
            return CircuitInstruction("Sd", data)
        if line.startswith("M "):
            data = _parse_terms(line[len("M ") :])
            return CircuitInstruction("M", data)
        if line.startswith("M_debug_force_0 "):
            data = _parse_terms(line[len("M_debug_force_0 ") :])
            return CircuitInstruction("M_debug_force_0", data)
        raise NotImplementedError(f"{line=}")

    def run_in_simulator(self, sim: VecSim):
        if self.name == "tick":
            return
        if self.name == "halftick":
            return
        if self.name == "highlight_box_start":
            return
        if self.name == "annotate":
            return
        if self.name == "highlight_box_end":
            return
        if self.name == "output":
            return
        if self.name == "drop":
            for q, v in self.qubit_terms:
                sim.do_mz_discard(q)
            return
        if self.name == "input":
            for q, v in self.qubit_terms:
                if v == "|i⟩":
                    sim.do_qalloc_y(q)
                elif v == "|CCZ⟩":
                    sim.do_qalloc_x(q)
                    sim.do_qalloc_x(q + 1)
                    sim.do_qalloc_x(q + 2)
                    sim.do_ccz(q, q + 1, q + 2)
                elif v == "|0⟩":
                    sim.do_qalloc_z(q)
                elif v == "|+⟩":
                    sim.do_qalloc_x(q)
                elif v.startswith("|") and v.endswith("⟩"):
                    raise NotImplementedError(f"{self=}")
                elif q not in sim.q2i:
                    raise ValueError(f"Missing expected input index={q} label={v}")
            return
        if self.name == "C" or self.name == "S" or self.name == "Sd":

            def do_cc():
                for k, (q0, gates) in enumerate(self.qubit_terms):
                    ctrl = f"__c_tmp{k}"
                    sim.do_h(ctrl)
                    for dq, gate in enumerate(gates):
                        q = q0 + dq
                        if gate == "_" or gate == " ":
                            pass
                        elif gate == "X":
                            sim.do_cx(ctrl, q)
                        elif gate == "Y":
                            sim.do_cy(ctrl, q)
                        elif gate == "Z" or gate == "C" or gate == "@":
                            sim.do_cz(ctrl, q)
                        elif gate == "H":
                            sim.do_ch(ctrl, q)
                        else:
                            raise NotImplementedError(f"{self=}")
                    sim.do_h(ctrl)

            for k in range(len(self.qubit_terms)):
                sim.do_qalloc_z(f"__c_tmp{k}")
            do_cc()
            if len(self.qubit_terms) == 1 and self.name == "C":
                sim.do_z("__c_tmp0")
            elif len(self.qubit_terms) == 1 and self.name == "S":
                sim.do_s("__c_tmp0")
            elif len(self.qubit_terms) == 1 and self.name == "Sd":
                sim.do_s_dag("__c_tmp0")
            elif len(self.qubit_terms) == 2 and self.name == "S":
                sim.do_cs("__c_tmp0", "__c_tmp1")
            elif len(self.qubit_terms) == 2 and self.name == "Sd":
                sim.do_cs_dag("__c_tmp0", "__c_tmp1")
            elif len(self.qubit_terms) == 2 and self.name == "C":
                sim.do_cz("__c_tmp0", "__c_tmp1")
            elif len(self.qubit_terms) == 3 and self.name == "C":
                sim.do_ccz("__c_tmp0", "__c_tmp1", "__c_tmp2")
            else:
                raise NotImplementedError(f"{self=}")
            do_cc()
            for k in range(len(self.qubit_terms)):
                uncomputed = sim.do_mz_discard(f"__c_tmp{k}")
                assert uncomputed == 0
            return
        if self.name == "M":
            for q, v in self.qubit_terms:
                if v == "X":
                    sim.do_h(q)
                elif v == "Y":
                    sim.do_h_yz(q)
                elif v == "Z":
                    pass
                else:
                    raise NotImplementedError(f"{self=}")
                sim.do_mz(q)
            return
        if self.name == "M_debug_force_0":
            for q, v in self.qubit_terms:
                if v == "X":
                    sim.do_h(q)
                elif v == "Y":
                    sim.do_h_yz(q)
                elif v == "Z":
                    pass
                else:
                    raise NotImplementedError(f"{self=}")
                m = sim.do_mz(q, prefer_result=False)
                assert not m, "failed to force 0"
            return
        raise NotImplementedError(f"{self=}")


class CircuitDrawer:
    def __init__(self):
        self.lines = []
        self.gate_diam = 15
        self.gate_pitch = 20
        self.wire_pitch = 20
        self.q_offset = 0
        self.t_offset = 0
        self.measure_times: dict[int, float] = {}
        self.intro_times: dict[int, float] = {}
        self.end_times: dict[int, float] = {}
        self.t = 0
        self.box_stack = []
        self._tick_used = set()

    def draw_program(self, program: CircuitProgram):
        for instruction in program.instructions:
            self.run_instruction(instruction)

    def run_instruction(self, instruction: CircuitInstruction):
        if instruction.name in ["C", "M", "S", "Sd"]:
            min_q = min(q for q, _ in instruction.qubit_terms)
            max_q = max(q + len(v) for q, v in instruction.qubit_terms)
            for q in range(min_q, max_q):
                if q in self._tick_used:
                    self.t += 1
                    self._tick_used.clear()
                    break
            for q in range(min_q, max_q):
                self._tick_used.add(q)
            for s, v in instruction.qubit_terms:
                for k in range(len(v)):
                    q = s + k
                    if q not in self.intro_times and v[k] != "_" and v[k] != " ":
                        self.intro_times[q] = self.t

        if instruction.name == "tick":
            self._tick_used.clear()
            self.t += 1
        elif instruction.name == "halftick":
            self._tick_used.clear()
            self.t += 0.5
        elif instruction.name == "highlight_box_start":
            self.box_stack.append(self.t)
        elif instruction.name == "annotate":
            for q, v in instruction.qubit_terms:
                self.draw_mid_state(q=q, s=v, t=self.t)
        elif instruction.name == "highlight_box_end":
            if self._tick_used:
                self.t += 1
            self._tick_used.clear()
            t0 = self.box_stack.pop()
            for q0, label in instruction.qubit_terms:
                self.draw_box(
                    t0=t0 - 0.66,
                    t1=self.t - 0.33,
                    q0=q0,
                    q1=max(self.intro_times.keys()) + 1,
                    label=label.replace("\\n", "\n"),
                )
        elif instruction.name == "output":
            for q, v in instruction.qubit_terms:
                self.draw_end_state(q=q, s=v)
        elif instruction.name == "drop":
            for q, v in instruction.qubit_terms:
                self.end_times[q] = self.t
        elif instruction.name == "bit":
            for q, v in instruction.qubit_terms:
                self.intro_times[q] = self.t + 0.5
                self.measure_times[q] = self.t + 0.5
                self.draw_state(q=q, s=v, t=self.t)
        elif instruction.name == "input":
            for q, v in instruction.qubit_terms:
                if v == "|CCZ⟩":
                    self.intro_times[q] = self.t + 0.5
                    self.intro_times[q + 1] = self.t + 0.5
                    self.intro_times[q + 2] = self.t + 0.5
                    self.draw_state(q=q + 1, s=v, multi_qubit=True, t=self.t)
                else:
                    self.draw_state(q=q, s=v, t=self.t)
        elif instruction.name == "C":
            self.draw_cphase(t=self.t, data=instruction.qubit_terms)
        elif instruction.name == "S":
            self.draw_cphase(t=self.t, data=instruction.qubit_terms, phase=1j)
        elif instruction.name == "Sd":
            self.draw_cphase(t=self.t, data=instruction.qubit_terms, phase=-1j)
        elif instruction.name == "M":
            for q, v in instruction.qubit_terms:
                if len(v) != 1:
                    raise NotImplementedError(f"{instruction=}")
                self.draw_measure_box(t=self.t, q=q, basis=v)
        elif instruction.name == "M_debug_force_0":
            for q, v in instruction.qubit_terms:
                if len(v) != 1:
                    raise NotImplementedError(f"{instruction=}")
                self.draw_measure_box(t=self.t, q=q, basis=v, highlight=True)
            self.t += 1
        else:
            raise NotImplementedError(f"{instruction=}")

    def finish(self) -> str:
        early_lines = []
        xy = self.proj(self.t + 1, max(self.intro_times, default=1) + 1)
        early_lines.append(
            f"""<svg viewBox="-100 -25 {int(xy.real) + 200} {int(xy.imag) + 50}" font-family="Times New Roman" xmlns="http://www.w3.org/2000/svg">"""
        )
        for q in self.intro_times.keys() | self.measure_times.keys() | self.end_times.keys():
            t = self.intro_times.get(q, 0)
            t2 = self.end_times.get(q, self.t)
            mt = self.measure_times.get(q)
            xy0 = self.proj(t, q)
            xy2 = self.proj(t2, q)
            x0 = xy0.real
            x2 = xy2.real
            y = xy0.imag
            if t == mt:
                xy1 = self.proj(mt, q)
                x1 = xy1.real
                y1 = y - 1
                y2 = y + 1
                early_lines.append(
                    f"""<path d="M{x1-3},{y1} L{x2},{y1} M{x1-3},{y2} L{x2},{y2}" stroke="black" fill="none"/>"""
                )
            elif mt is not None:
                xy1 = self.proj(mt, q)
                x1 = xy1.real
                y1 = y - 1
                y2 = y + 1
                early_lines.append(
                    f"""<path d="M{x0 - 3},{y} L{x1},{y} M{x1},{y1} L{x2},{y1} M{x1},{y2} L{x2},{y2}" stroke="black" fill="none"/>"""
                )
            else:
                early_lines.append(
                    f"""<path d="M{x0 - 3},{y} L{x2},{y}" stroke="black" fill="none"/>"""
                )

        result = early_lines + list(self.lines)
        result.append("""</svg>""")
        return "\n".join(result)

    def write_to(self, path: str | pathlib.Path):
        with open(path, "w") as f:
            print(self.finish(), file=f)
        print(f"wrote file://{pathlib.Path(path).absolute()}", file=sys.stderr)

    def proj(self, t: float, q: float) -> complex:
        q += self.q_offset
        t += self.t_offset
        return (
            (q + 1) * self.wire_pitch * 1j
            + t * self.gate_pitch
            + self.gate_diam / 2
            + self.wire_pitch * 1j / 2
        )

    def draw_measure_box(self, t: int, q: int, basis: str, *, highlight: bool = False):
        xy = self.proj(t, q)
        x = xy.real
        y = xy.imag
        self.measure_times[q] = t
        fill = "black"
        if highlight:
            fill = "red"
        self.lines.append(
            f"""<rect x="{x - 0.5 * self.gate_diam}" y="{y - 0.5 * self.gate_diam}" width="{self.gate_diam}" height="{self.gate_diam}" stroke="black" fill="{fill}" />"""
        )
        self.lines.append(
            f"""<text x="{x}" y="{y+1}" font-size="9" text-anchor="middle" dominant-baseline="middle" fill="white">M{basis}</text>"""
        )

    def draw_string_box(self, t: int, q: int, s: str, dash: bool):
        invert = not s.replace("@", "").replace("C", "").replace(" ", "")
        s = s.replace("@", "C")
        xy = self.proj(t, q)
        x = xy.real
        y = xy.imag
        w = self.gate_diam
        h = self.gate_pitch * (len(s) - 1) + self.gate_diam
        stroke = "black"
        fill = "white"
        if invert:
            fill = "black"
            stroke = "none"
        extra = ""
        if dash:
            extra = ''' stroke-dasharray="3 2"'''
        self.lines.append(
            f"""<rect x="{x - 0.5*self.gate_diam}" y="{y - 0.5*self.gate_diam}" width="{w}" height="{h}" stroke="{stroke}" fill="{fill}"{extra}/>"""
        )

        for k, c in enumerate(s):
            if c != " ":
                fill = "black"
                if invert:
                    fill = "white"
                self.lines.append(
                    f"""<text x="{x}" y="{y + k*20 + 1}" font-size="16" text-anchor="middle" dominant-baseline="middle" fill="{fill}">{c}</text>"""
                )

    def draw_cphase(
        self,
        data: dict[int, str] | Iterable[tuple[int, str]],
        *,
        t: int,
        phase: complex = -1,
        bend: bool = False,
    ):
        if isinstance(data, dict):
            data = data.items()
        new_data = []
        hits = collections.Counter()
        for q_start, gate_str in data:
            gate_str = gate_str.replace("_", " ").rstrip()
            while gate_str and gate_str[0] == " ":
                gate_str = gate_str[1:]
                q_start += 1
            new_data.append((q_start, gate_str))
            for k in range(len(gate_str)):
                hits[q_start + k] += 1

        max_hits = max(hits.values())
        actual_uses = collections.Counter()
        offsets = []
        for q_start, gate_str in new_data:
            max_uses = max(actual_uses[q_start + k] for k in range(len(gate_str)))
            for k in range(len(gate_str)):
                actual_uses[q_start + k] += 1
            if all(hits[k + q_start] == 1 for k in range(len(gate_str))):
                offsets.append(max_hits / 2 - 0.5)
            else:
                offsets.append(max_uses)

        xys = []
        for (q_start, gate_str), offset in zip(new_data, offsets):
            xys.append(self.proj(t + offset, q_start))

        min_y = min(e.imag for e in xys)
        max_y = max(e.imag for e in xys)
        if any(offsets):
            pts = []
            for (q_start, gate_str), offset in zip(new_data, offsets):
                pts.append(self.proj(t + offset, q_start + (len(gate_str) - 1) / 2))
            for k in range(len(pts)):
                v0 = pts[k]
                v1 = pts[k - 1]
                self.lines.append(
                    f"""<path d="M{v0.real},{v0.imag} L{v1.real},{v1.imag}" stroke="black" fill="none" stroke-width="3"/>"""
                )
        else:
            classical_control = sum("@" in e[1] or "C" in e[1] for e in data) >= len(data) - 1
            for x0 in [xys[0].real + 1, xys[0].real - 1] if classical_control else [xys[0].real]:
                if bend:
                    self.lines.append(
                        f"""<path d="M{x0},{min_y} C{x0-self.gate_diam} {min_y}, {x0-self.gate_pitch} {max_y}, {x0} {max_y}" stroke="black" fill="none"/>"""
                    )
                else:
                    self.lines.append(
                        f"""<path d="M{x0},{min_y} L{x0},{max_y}" stroke="black" fill="none"/>"""
                    )
        if phase == -1j:
            txt = "-i"
        elif phase == 1j:
            txt = "i"
        elif phase == -1:
            txt = ""
        else:
            raise NotImplementedError(f"{phase=}")
        if txt:
            self.lines.append(
                f"""<text x="{xys[0].real}" y="{min_y-self.gate_diam/2-1}" font-size="14" text-anchor="middle">{txt}</text>"""
            )

        has_clifford = phase != -1
        non_classical = 0
        for _, gate_str in new_data:
            if not (set(gate_str) <= set(" _XYZ@C")):
                has_clifford = True
            if not (set(gate_str) <= set(" _C@")):
                non_classical += 1
        can_be_pauli = non_classical <= 1 and not has_clifford

        for (q_start, gate_str), offset in zip(new_data, offsets):
            self.draw_string_box(t + offset, q_start, gate_str, dash=can_be_pauli)

    def draw_state(self, *, q: int, s: str, multi_qubit: int = False, t: int = 0):
        self.intro_times[q] = t + 0.5
        xy = self.proj(t + 0.5, q)
        if multi_qubit:
            rot = -90 if multi_qubit == -1 else 90
            baseline = "bottom" if multi_qubit == -1 else "hanging"
            self.lines.append(
                f"""<text x="0" y="0" font-size="18" text-anchor="middle" dominant-baseline="{baseline}" transform="translate({xy.real-5},{xy.imag}) rotate({rot})">{s}</text>"""
            )
        else:
            self.lines.append(
                f"""<text x="{xy.real-3}" y="{xy.imag}" font-size="18" text-anchor="end" dominant-baseline="middle">{s}</text>"""
            )

    def draw_end_state(self, *, q: int, s: str):
        xy = self.proj(self.t, q)
        self.lines.append(
            f"""<text x="{xy.real+3}" y="{xy.imag}" font-size="18" text-anchor="start" dominant-baseline="middle">{s}</text>"""
        )

    def draw_mid_state(self, *, q: int, s: str, t: int):
        xy = self.proj(t, q)
        self.lines.append(
            f"""<text x="{xy.real}" y="{xy.imag-4}" font-size="12" text-anchor="middle" dominant-baseline="bottom">{s}</text>"""
        )

    def draw_box(self, *, t0: float, t1: float, q0: float, q1: float, label: str):
        xy1 = self.proj(t0, q0) - self.wire_pitch / 2 * 1j
        xy2 = self.proj(t1, q1) - self.wire_pitch / 2 * 1j
        xy3 = (xy1.real + xy2.real) / 2 + xy2.imag * 1j
        self.lines.append(
            f"""<rect x="{xy1.real}" y="{xy1.imag}" width="{xy2.real - xy1.real}" height="{xy2.imag - xy1.imag}" stroke="red" fill="none" stroke-dasharray="4 1" stroke-width="2" />"""
        )
        for label in label.splitlines():
            self.lines.append(
                f"""<text x="{xy3.real}" y="{xy3.imag + 2}" font-size="12" text-anchor="middle" dominant-baseline="text-before-edge" fill="red">{label}</text>"""
            )
            xy3 += 12j


def _parse_terms(line) -> tuple[tuple[int, str], ...]:
    terms = line.split()
    data: list[tuple[int, str]] = []
    for term in terms:
        if ":" in term:
            offset, ps = term.split(":")
            offset = int(offset)
        else:
            offset = 0
            ps = term
        data.append((offset, ps))

    return tuple(data)
