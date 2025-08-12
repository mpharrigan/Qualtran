import functools
import math
from typing import Iterable, Sequence

import numpy as np
import stim


@functools.lru_cache()
def make_tableaus() -> dict[str, stim.Tableau]:
    gates: dict[str, stim.Tableau] = {}
    data: stim.GateData
    for name, data in stim.gate_data().items():
        if data.is_unitary and data.is_single_qubit_gate:
            t = data.tableau
            for alias in data.aliases:
                gates[alias] = t
    gates[" "] = gates["I"]
    return gates


@functools.lru_cache()
def make_unitaries() -> dict[str, np.ndarray]:
    gates: dict[str, np.ndarray] = {}
    data: stim.GateData
    s = 0.5**0.5
    known_vals = [
        0,
        1,
        1j,
        s,
        -s,
        0.5 - 0.5j,
        -0.5 - 0.5j,
        0.5 + 0.5j,
        -0.5 + 0.5j,
        s - s * 1j,
        s + s * 1j,
        -s * 1j,
        s * 1j,
        -1j,
        -1,
    ]
    for name, data in stim.gate_data().items():
        if data.is_unitary and data.is_single_qubit_gate:
            u = data.unitary_matrix
            u = np.array(u, dtype=np.complex128)
            for i in range(2):
                for j in range(2):
                    v = u[i, j]
                    for v2 in known_vals:
                        if abs(v - v2) < 1e-5:
                            u[i, j] = v2
                            break
                    else:
                        raise NotImplementedError(f"{v=}")
            for alias in data.aliases:
                gates[alias] = u
    gates["T"] = np.array([[1, 0], [0, 1j**0.5]], dtype=np.complex128)
    gates["T_DAG"] = np.array([[1, 0], [0, 1j**-0.5]], dtype=np.complex128)
    gates["T_Z"] = gates["T"]
    gates["T_Z_DAG"] = gates["T_DAG"]
    gates["T_X"] = gates["H"] @ gates["T"] @ gates["H"]
    gates["T_X_DAG"] = gates["H"] @ gates["T_DAG"] @ gates["H"]
    gates["T_Y"] = gates["H_YZ"] @ gates["T"] @ gates["H_YZ"]
    gates["T_Y_DAG"] = gates["H_YZ"] @ gates["T_DAG"] @ gates["H_YZ"]
    gates[" "] = gates["I"]
    return gates


def gate_seq_to_matrix(gates: Iterable[str]) -> np.ndarray:
    unitaries = make_unitaries()
    result = np.eye(2, dtype=np.complex128)
    sign_xz = [[unitaries["T_X"], unitaries["T_X_DAG"]], [unitaries["T_Z"], unitaries["T_Z_DAG"]]]
    sign_step = 0
    for gate in gates:
        if gate == "+":
            u = sign_xz[sign_step & 1][0]
            sign_step += 1
        elif gate == "-":
            u = sign_xz[sign_step & 1][1]
            sign_step += 1
        else:
            u = unitaries[gate]
        result = u @ result
    return result


def u2_to_zero_state(mat: np.ndarray) -> tuple[float, float, float]:
    u = mat[0][0]
    _ = mat[0][1]
    v = mat[1][0]
    _ = mat[1][1]

    a = u * np.conj(u)
    b = u * np.conj(v)
    c = v * np.conj(u)
    d = v * np.conj(v)
    x = b.real + c.real
    y = c.imag - b.imag
    z = a.real - d.real
    return x, y, z


def u2_to_quaternion(mat: np.ndarray) -> tuple[float, float, float, float]:
    a = mat[0][0]
    b = mat[0][1]
    c = mat[1][0]
    d = mat[1][1]
    i = 1j
    x = b + c
    y = b * i + c * -i
    z = a - d
    s = a + d
    s *= -i
    p = max([s, x, y, z], key=abs)
    p /= math.sqrt(p.imag * p.imag + p.real * p.real)
    p *= 2
    x /= p
    y /= p
    z /= p
    s /= p
    x = x.real
    y = y.real
    z = z.real
    s = s.real
    if s < 0:
        s *= -1
        x *= -1
        y *= -1
        z *= -1
    return s, x, y, z


def simplify_hst_sequence(gates: Iterable[str]) -> list[str]:
    tableaus = make_tableaus()
    out = ["I"]
    frame = stim.Tableau(1)
    for c in gates:
        if c in tableaus:
            frame = tableaus[c] * frame
        elif c == "T" or c == "T_DAG":
            if c == "T_DAG":
                frame = tableaus["S_DAG"] * frame
            t = stim.PauliString("Z").before(frame, targets=[0])
            p = "_XYZ"[t[0]]
            if p == "Z":
                if t.sign == -1:
                    if out[-1] == "I":
                        out[-1] = "H"
                        out.append("T_X_DAG")
                        out.append("H")
                    elif out[-1] == "T_Z":
                        out[-1] = "I"
                    elif out[-1] == "T_Z_DAG":
                        out[-1] = "SQRT_Z_DAG"
                    else:
                        out.append("T_Z_DAG")
                else:
                    if out[-1] == "I":
                        out[-1] = "H"
                        out.append("T_X")
                        out.append("H")
                    elif out[-1] == "T_Z_DAG":
                        out[-1] = "I"
                    elif out[-1] == "T_Z":
                        out[-1] = "SQRT_Z"
                    else:
                        out.append("T_Z")
            elif p == "X":
                if t.sign == -1:
                    if out[-1] == "T_X":
                        out[-1] = "I"
                    elif out[-1] == "T_X_DAG":
                        out[-1] = "SQRT_X_DAG"
                    else:
                        out.append("T_X_DAG")
                else:
                    if out[-1] == "T_X_DAG":
                        out[-1] = "I"
                    elif out[-1] == "T_X":
                        out[-1] = "SQRT_X"
                    else:
                        out.append("T_X")
            elif p == "Y":
                if out[-1] == "I":
                    out[-1] = "S"
                    out.append("T_X_DAG")
                    out.append("S_DAG")
                elif out[-1] == "T_Z":
                    out[-1] = "T_Z_DAG"
                    out.append("T_X")
                    out.append("SQRT_Z")
                elif out[-1] == "T_Z_DAG":
                    out[-1] = "T_Z"
                    out.append("T_X_DAG")
                    out.append("SQRT_Z_DAG")
                elif out[-1] == "T_X":
                    out[-1] = "T_X_DAG"
                    out.append("T_Z_DAG")
                    out.append("SQRT_X")
                elif out[-1] == "T_X_DAG":
                    out[-1] = "T_X"
                    out.append("T_Z")
                    out.append("SQRT_X_DAG")
                else:
                    raise NotImplementedError(f"{out[-1]=}")
                if t.sign == -1:
                    out.append("SQRT_Y_DAG")
            else:
                raise NotImplementedError(f"{t=}")
        else:
            raise NotImplementedError(f"{c=}")
        while out and out[-1] in tableaus:
            frame = frame * tableaus[out.pop()]
        if not out:
            out.append("I")
    assert out[0] in ["I", "H", "S"]
    p2b = {"H": "Z", "I": "X", "S": "Y"}
    b2p = {b: p for p, b in p2b.items()}
    init_basis = p2b[out[0]]
    if len(out) == 1:
        p = stim.PauliString(1)
        p[0] = init_basis
        p = p.after(frame, targets=[0])
        out[0] = b2p["_XYZ"[p[0]]]
        if p.sign == -1:
            if out[0] == "Z":
                out.append("X")
            else:
                out.append("Z")
        return out

    options = []
    options.append([*out, *tableau_to_clifford_string(frame)])
    if out[-1] == "T_X":
        options.append(
            [*out[:-1], "T_X_DAG", *tableau_to_clifford_string(frame * tableaus["SQRT_X"])]
        )
    elif out[-1] == "T_X_DAG":
        options.append(
            [*out[:-1], "T_X", *tableau_to_clifford_string(frame * tableaus["SQRT_X_DAG"])]
        )
    elif out[-1] == "T_Z":
        options.append(
            [*out[:-1], "T_Z_DAG", *tableau_to_clifford_string(frame * tableaus["SQRT_Z"])]
        )
    elif out[-1] == "T_Z_DAG":
        options.append(
            [*out[:-1], "T_Z", *tableau_to_clifford_string(frame * tableaus["SQRT_Z_DAG"])]
        )
    if init_basis == "X":
        reps = {"T_Z": "T_Z_DAG", "T_Z_DAG": "T_Z"}
    elif init_basis == "Y":
        reps = {"T_Z": "T_Z_DAG", "T_Z_DAG": "T_Z", "T_X": "T_X_DAG", "T_X_DAG": "T_X"}
    elif init_basis == "Z":
        reps = {"T_X": "T_X_DAG", "T_X_DAG": "T_X"}
    else:
        raise NotImplementedError(f"{init_basis=}")
    options.append(
        [*[reps.get(e, e) for e in out], *tableau_to_clifford_string(frame * tableaus[init_basis])]
    )

    def preference_cost(sequence: list[str]) -> int:
        return len(sequence), sum(e == "-" for e in sequence), "Y" in sequence

    return min(options, key=preference_cost)


def tableau_to_clifford_string(tableau: stim.Tableau) -> str:
    decomp = "".join(inst.name for inst in tableau.to_circuit() for _ in inst.targets_copy())
    decomp = decomp.replace("SS", "Z")
    decomp = decomp.replace("ZS", "SZ")
    decomp = decomp.replace("HZH", "X")
    decomp = decomp.replace("XZ", "Y")
    decomp = decomp.replace("ZX", "Y")
    decomp = decomp.replace("HH", "")
    return decomp


def compress_tx_tz_sequence(seq: Sequence[str]) -> str:
    tableaus = make_tableaus()
    assert seq[0] in ["I", "H", "S"]
    if seq[0] == "I":
        prefix = ["I"]
    elif seq[0] == "H":
        prefix = ["H"]
    elif seq[0] == "S":
        prefix = ["S"]
    else:
        raise NotImplementedError(f"{seq[0]=}")
    ts = list(seq[1:])
    rev_end = []
    while ts and ts[-1] in tableaus:
        rev_end.append(ts.pop())
    out = []
    for k in range(len(ts)):
        if k % 2 == 1:
            if ts[k] == "T_Z":
                out.append("+")
            elif ts[k] == "T_Z_DAG":
                out.append("-")
            else:
                raise NotImplementedError(f"{k=} {ts[k]=}")
        else:
            if ts[k] == "T_X":
                out.append("+")
            elif ts[k] == "T_X_DAG":
                out.append("-")
            else:
                raise NotImplementedError(f"{k=} {ts[k]=}")
    rev_end = "".join(rev_end[::-1])
    return "".join([*prefix, *out, *rev_end])


def sign_rotation_string_to_sign_init_string(line: str) -> tuple[str, str, str]:
    saw_pm = False
    prefix_end = 0
    suffix_start = len(line)
    for k in range(len(line)):
        c = line[k]
        if c == " ":
            pass
        elif c in "+-":
            if not saw_pm:
                prefix_end = k
                saw_pm = True
        else:
            if saw_pm:
                suffix_start = k
                saw_pm = False
    if "+" not in line and "-" not in line:
        suffix_start = 0
    prefix = line[:prefix_end].strip()
    body = line[prefix_end:suffix_start].strip()
    assert set(body) <= set("+-")
    suffix = line[suffix_start:].strip()

    if prefix == "I" or prefix == "":
        init = "RX"
    elif prefix == "H":
        init = "RZ"
    elif prefix == "S":
        init = "RY"
    else:
        raise NotImplementedError(f"{prefix=}")

    if init == "RX" and len(body) > 0:
        body = body[1:]
        init = "RZ"
        suffix = "H" + suffix
    tableaus = make_tableaus()
    t = stim.Tableau(1)
    for gate in suffix:
        t = tableaus[gate] * t
    suffix = tableau_to_clifford_string(t)

    return init, body, suffix
