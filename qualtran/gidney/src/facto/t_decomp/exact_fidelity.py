from __future__ import annotations

from typing import Iterable
from mpmath import mp


class Val:
    SQRT_2: Val

    def __init__(self, obj: int | complex = 0, *, r: int = 0, i: int = 0, r2: int = 0, i2: int = 0):
        self.r = r
        self.i = i
        self.r2 = r2
        self.i2 = i2

        if isinstance(obj, int):
            self.r += obj
        elif (
            isinstance(obj, complex) and obj.real == round(obj.real) and obj.imag == round(obj.imag)
        ):
            self.r += round(obj.real)
            self.i += round(obj.imag)
        elif isinstance(obj, Val):
            self.r += obj.r
            self.i += obj.i
            self.r2 += obj.r2
            self.i2 += obj.i2
        else:
            raise NotImplementedError(f"{obj=}")

    def mp_approx(self):
        return self.r + self.r2 * mp.sqrt(2) + (self.i + self.i2 * mp.sqrt(2)) * 1j

    def sqrt2_divisibility(self) -> int:
        ri = self.r | self.i
        ri2 = self.r2 | self.i2
        count = 0
        while count < 64 and ri & 0xFF == 0 and ri2 & 0xFF == 0:
            ri >>= 8
            ri2 >>= 8
            count += 16
        while count < 64 and ri & 1 == 0 and ri2 & 1 == 0:
            ri >>= 1
            ri2 >>= 1
            count += 2
        if ri & 1 == 0:
            count += 1
        return count

    def __rshift__(self, other: int) -> Val:
        return Val(r=self.r >> other, i=self.i >> other, r2=self.r2 >> other, i2=self.i2 >> other)

    def __mul__(self, other: Val | int | complex) -> Val:
        if isinstance(other, (int, complex)):
            other = Val(other)
        if isinstance(other, Val):
            return Val(
                r=self.r * other.r
                - self.i * other.i
                + 2 * self.r2 * other.r2
                - 2 * self.i2 * other.i2,
                i=self.r * other.i
                + self.i * other.r
                + 2 * self.r2 * other.i2
                + 2 * self.i2 * other.r2,
                r2=self.r * other.r2 + self.r2 * other.r - self.i2 * other.i - self.i * other.i2,
                i2=self.r2 * other.i + self.i * other.r2 + self.i2 * other.r + self.r * other.i2,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other: Val | int | complex) -> Val:
        if isinstance(other, (int, complex)):
            other = Val(other)
        if isinstance(other, Val):
            return Val(
                r=self.r + other.r, i=self.i + other.i, r2=self.r2 + other.r2, i2=self.i2 + other.i2
            )
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> Val:
        return Val(r=-self.r, i=-self.i, r2=-self.r2, i2=-self.i2)

    def __sub__(self, other: Val | int | complex) -> Val:
        if isinstance(other, (int, complex)):
            other = Val(other)
        if isinstance(other, Val):
            return self + -other
        return NotImplemented

    def __rsub__(self, other: Val | int | complex) -> Val:
        if isinstance(other, (int, complex)):
            other = Val(other)
        if isinstance(other, Val):
            return other + -self
        return NotImplemented

    def __eq__(self, other) -> bool:
        if (
            isinstance(other, (int, complex))
            and round(other.real) == other.real
            and round(other.imag) == other.imag
        ):
            other = Val(other)
        if isinstance(other, Val):
            return (
                self.r == other.r
                and self.i == other.i
                and self.r2 == other.r2
                and self.i2 == other.i2
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.r, self.i, self.r2, self.i2))

    def __str__(self) -> str:
        val = self
        has_term = False
        out = ""
        if val.r:
            out += str(val.r)
            has_term = True
        if val.i:
            if val.i > 0 and has_term:
                out += "+"
            if val.i == -1:
                out += "-"
            elif val.i != 1:
                out += str(val.i)
            out += "i"
            has_term = True
        if val.r2:
            if val.r2 > 0 and has_term:
                out += "+"
            if val.r2 == -1:
                out += "-"
            elif val.r2 != 1:
                out += str(val.r2)
            out += "√2"
            has_term = True
        if val.i2:
            if val.i2 > 0 and has_term:
                out += "+"
            if val.i2 == -1:
                out += "-"
            elif val.i2 != 1:
                out += str(val.i2)
            out += "i√2"
            has_term = True
        if not has_term:
            out += "0"
        return out


Val.SQRT_2 = Val(r2=1)


class Ket:
    def __init__(self, amp0: Val | int | complex, amp1: Val | int, divisions_by_two: int = 0):
        amp0 = Val(amp0)
        amp1 = Val(amp1)
        d = min(divisions_by_two, amp0.sqrt2_divisibility() >> 1, amp1.sqrt2_divisibility() >> 1)
        self.amp0 = amp0 >> d
        self.amp1 = amp1 >> d
        self.divisions_by_two = divisions_by_two - d

    def mp_approx(self):
        a = self.amp0.mp_approx()
        b = self.amp1.mp_approx()
        for _ in range(self.divisions_by_two):
            a /= 2
            b /= 2
        return a, b

    def after_h_xy(self) -> Ket:
        return self.after_x().after_s()

    def after_h_yz(self) -> Ket:
        a = self.amp0 + self.amp1 * -1j
        b = self.amp0 * 1j - self.amp1
        a = a * Val.SQRT_2
        b = b * Val.SQRT_2
        return Ket(a, b, divisions_by_two=self.divisions_by_two + 1)

    def after_h_then_t(self) -> Ket:
        a = self.amp0 + self.amp1
        b = self.amp0 - self.amp1
        a = a * Val.SQRT_2
        b = b * (1 + 1j)
        return Ket(a, b, divisions_by_two=self.divisions_by_two + 1)

    def after_t(self) -> Ket:
        a = self.amp0 * 2
        b = self.amp1 * (1 + 1j) * Val.SQRT_2
        return Ket(a, b, divisions_by_two=self.divisions_by_two + 1)

    def after_t_dag(self) -> Ket:
        a = self.amp0 * 2
        b = self.amp1 * Val.SQRT_2 * (1 - 1j)
        return Ket(a, b, divisions_by_two=self.divisions_by_two + 1)

    def after_h(self) -> Ket:
        a = self.amp0 + self.amp1
        b = self.amp0 - self.amp1
        a = a * Val.SQRT_2
        b = b * Val.SQRT_2
        return Ket(a, b, divisions_by_two=self.divisions_by_two + 1)

    def after_s(self) -> Ket:
        a = self.amp0
        b = self.amp1
        b = b * 1j
        return Ket(a, b, divisions_by_two=self.divisions_by_two)

    def after_z(self) -> Ket:
        a = self.amp0
        b = self.amp1
        b = -b
        return Ket(a, b, divisions_by_two=self.divisions_by_two)

    def after_x(self) -> Ket:
        a = self.amp0
        b = self.amp1
        a, b = b, a
        return Ket(a, b, divisions_by_two=self.divisions_by_two)

    def after_gate(self, gate: str) -> Ket:
        if gate == "H_XY":
            return self.after_h_xy()
        elif gate == "H":
            return self.after_h()
        elif gate == "H_YZ":
            return self.after_h_yz()
        elif gate == "T":
            return self.after_t()
        elif gate == "S":
            return self.after_s()
        elif gate == "T_DAG":
            return self.after_t_dag()
        elif gate == "X":
            return self.after_x()
        elif gate == "Y":
            return self.after_x().after_z()
        elif gate == "Z":
            return self.after_z()
        else:
            raise NotImplementedError(f"{gate=}")

    @staticmethod
    def ket_0() -> Ket:
        return Ket(1, 0)

    @staticmethod
    def ket_1() -> Ket:
        return Ket(0, 1)

    @staticmethod
    def ket_p() -> Ket:
        return Ket(Val.SQRT_2, Val.SQRT_2, divisions_by_two=1)

    @staticmethod
    def ket_m() -> Ket:
        return Ket(Val.SQRT_2, -Val.SQRT_2, divisions_by_two=1)

    @staticmethod
    def ket_i() -> Ket:
        return Ket(Val.SQRT_2, 1j * Val.SQRT_2, divisions_by_two=1)

    @staticmethod
    def ket_j() -> Ket:
        return Ket(Val.SQRT_2, -1j * Val.SQRT_2, divisions_by_two=1)


def high_precision_fidelity_analysis_of_phase_gradient_gate_sequence(
    *, phase_gradient_qubit_index: int, gate_sequence: Iterable[str], digits_of_precision: int
) -> dict[str, float]:
    a = Ket.ket_0()
    h = False
    for gate in gate_sequence:
        if gate == "+" or gate == "-":
            h = not h
            if h:
                a = a.after_h()
            if gate == "+":
                a = a.after_t()
            else:
                a = a.after_t_dag()
            if h:
                a = a.after_h()
        else:
            a = a.after_gate(gate)

    old_precision = mp.dps
    try:
        mp.dps = digits_of_precision

        angle = mp.pi
        for k in range(phase_gradient_qubit_index):
            angle /= 2
        b0 = mp.sqrt(0.5)
        b1 = mp.exp(1j * angle) * mp.sqrt(0.5)
        a0, a1 = a.mp_approx()
        dot = a0 * mp.conj(b0) + a1 * mp.conj(b1)
        fidelity = (dot * mp.conj(dot)).real
        infidelity = 1 - fidelity
        trace_distance = mp.sqrt(infidelity)
        return {
            "mp_infidelity": infidelity,
            "float_infidelity": float(infidelity),
            "mp_trace_distance": trace_distance,
            "float_trace_distance": float(trace_distance),
        }
    finally:
        mp.dps = old_precision
