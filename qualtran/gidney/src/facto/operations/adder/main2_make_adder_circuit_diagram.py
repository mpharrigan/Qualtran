from __future__ import annotations

import argparse
import pathlib
import sys

from facto.operations._circuit_diagram_drawer import CircuitDrawer, CircuitProgram


def draw_basic_tof_teleport(d: CircuitDrawer):
    program = CircuitProgram.from_program_text(
        r"""
        # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22Measure%22%2C%22Measure%22%2C%22Measure%22%5D%2C%5B1%2C1%2C1%2C%22zpar%22%2C1%2C%22xpar%22%2C1%2C%22Z%5E-%C2%BD%22%5D%2C%5B1%2C1%2C1%2C1%2C%22zpar%22%2C%22xpar%22%2C%22Z%5E-%C2%BD%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22X%5E%C2%BD%22%2C%22zpar%22%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C1%2C1%2C1%2C%22zpar%22%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C1%2C%22zpar%22%2C1%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C%22zpar%22%2C%22zpar%22%2C1%2C1%2C1%2C%22Z%5E-%C2%BD%22%5D%2C%5B1%2C1%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%5D%7D
        input 0:a 1:b 2:t
        tick
        tick
        input 3:|CCZ⟩
        highlight_box_start
        tick
        highlight_box_end 3:Magic\nState
        tick
        highlight_box_start
        C 0:Z 3:X
        C 1:Z 4:X
        C 2:X 5:X
        M 3:Z 4:Z 5:Z
        highlight_box_end 0:Measure\nParities
        tick
        highlight_box_start
        Sd 0:Z_X 3:_@_
        Sd 0:_ZX 3:@__
        S 0:__X 3:@@_
        highlight_box_end -1:Clifford\nCorrections
        tick
        highlight_box_start
        S 0:_Z_ 3:@_@
        S 0:Z__ 3:_@@
        Sd 0:ZZ_ 3:__@
        highlight_box_end -1:Deferred Phase\nCorrections
        drop 3 4 5
        tick
        output 0:a 1:b 2:t⊕ab
    """
    )
    d.draw_program(program)

    in2q = {"a": 0, "b": 1, "t": 2}
    out2q = {"a": 0, "b": 1, "t": 2}
    program.verify(
        shots=4,
        in2q=in2q,
        out2q=out2q,
        in_func=lambda c: c,
        out_func=lambda c: {**c, "t": c["a"] & c["b"] ^ c["t"]},
    )


def draw_deferred_teleport(d: CircuitDrawer):
    program = CircuitProgram.from_program_text(
        r"""
        # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C%22%E2%80%A6%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C%22Z%5E%C2%BD%22%2C%22Z%5E%C2%BD%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C%22xpar%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C%22xpar%22%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22xpar%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C%22%E2%80%A6%22%5D%2C%5B1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22Measure%22%2C%22Measure%22%2C%22Measure%22%5D%2C%5B1%2C1%2C%22%E2%80%A6%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C%22zpar%22%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Measure%22%2C%22Measure%22%2C%22Measure%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22X%22%2C%22zpar%22%2C%22zpar%22%2C1%2C%22zpar%22%2C%22zpar%22%2C%22zpar%22%5D%2C%5B1%2C1%2C%22%E2%80%A6%22%5D%2C%5B1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C1%2C1%2C1%2C%22zpar%22%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C1%2C%22zpar%22%2C1%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C%22zpar%22%2C%22zpar%22%2C1%2C1%2C1%2C%22Z%5E-%C2%BD%22%5D%2C%5B1%2C1%2C1%2C1%2C%22Z%22%2C1%2C%22zpar%22%2C1%2C1%2C1%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C%22Z%22%2C1%2C1%2C1%2C%22zpar%22%2C1%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C%22Z%22%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C%22Z%22%2C%22Z%22%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C1%2C%22Z%22%2C1%2C%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%5D%2C%5B%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B%22X%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%5D%7D
        input 0:a
        input 1:b
        input 2:t
        tick
        tick
        highlight_box_start
        input 3:|CCZ⟩
        input 6:|i⟩
        input 7:|i⟩
        input 8:|i⟩
        tick
        C 3:Z__ 6:X__
        C 3:_Z_ 6:_X_
        C 3:__Z 6:XXX
        highlight_box_end 3:Semi-Auto\nMagic State
        tick
        highlight_box_start
        C 0:Z 3:X
        C 1:Z 4:X
        C 2:X 5:X
        M 3:Z 4:Z 5:Z
        highlight_box_end 0:Measure\nParities
        tick
        highlight_box_start
        C 3:_@_ 6:H__
        C 3:@__ 6:_H_
        C 3:@@_ 6:__H
        M 6:X 7:X 8:X
        highlight_box_end 3:Reaction-Limited\nBasis Choice
        tick
        highlight_box_start
        C 3:@@_@@@ 0:__X
        C 3:@___@_ 0:_Z_
        C 3:_@_@__ 0:Z__
        C 0:Z__ 4:@ 5:@
        C 0:ZZ_ 3:@ 4:@
        C 0:_Z_ 3:@ 5:@
        highlight_box_end 0:Backdated\nPauli Frame
        tick
        highlight_box_start
        S 3:@_@ 0:_Z_
        S 3:_@@ 0:Z__
        Sd 3:__@ 0:ZZ_
        highlight_box_end -1:Deferred Phase\nCorrections
        drop 3 4 5 6 7 8
        tick
        output 0:a
        output 1:b
        output 2:t⊕ab
    """
    )
    d.draw_program(program)

    in2q = {"a": 0, "b": 1, "t": 2}
    out2q = {"a": 0, "b": 1, "t": 2}
    program.verify(
        shots=4,
        in2q=in2q,
        out2q=out2q,
        in_func=lambda c: c,
        out_func=lambda c: {**c, "t": c["a"] & c["b"] ^ c["t"]},
    )


def make_adder_program() -> CircuitProgram:
    program = CircuitProgram.from_program_text(
        r"""
        # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22Counting4%22%5D%2C%5B%22X%5Et%22%2C%22X%5Et%22%2C%22Y%5Et%22%2C%22X%5Et%22%5D%2C%5B1%2C%22X%5Et%22%2C1%2C%22Z%5Et%22%5D%2C%5B%22~un0b%22%2C%22~sbi5%22%2C%22~pqmd%22%2C%22~ef5k%22%5D%2C%5B1%2C1%2C1%2C%22Swap%22%2C%22Swap%22%5D%2C%5B1%2C%22Swap%22%2C1%2C%22Swap%22%5D%2C%5B1%2C%22Swap%22%2C%22Swap%22%5D%2C%5B%22~un0b%22%2C%22~pqmd%22%2C1%2C%22~sbi5%22%2C%22~ef5k%22%5D%2C%5B%22zpar%22%2C%22zpar%22%2C%22X%22%5D%2C%5B%22~un0b%22%2C%22~pqmd%22%2C%22~1uop%22%2C%22~sbi5%22%2C%22~ef5k%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C%22Z%5E%C2%BD%22%2C%22Z%5E%C2%BD%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22Z%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C%22X%22%2C%22X%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22H%22%5D%2C%5B%22zpar%22%2C%22zpar%22%2C%22zpar%22%2C%22zpar%22%2C%22zpar%22%2C%22X%22%5D%2C%5B1%2C%22zpar%22%2C%22zpar%22%2C1%2C1%2C1%2C%22X%22%5D%2C%5B%22zpar%22%2C1%2C%22zpar%22%2C1%2C1%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22Measure%22%2C%22Measure%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C%22zpar%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Measure%22%2C%22Measure%22%2C%22Measure%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C%22X%22%2C%22zpar%22%2C%22zpar%22%2C%22zpar%22%2C%22zpar%22%2C%22zpar%22%5D%2C%5B%22~un0b%22%2C%22~pqmd%22%2C%22~1uop%22%2C%22~sbi5%22%2C%22~ef5k%22%2C%22~1lob%22%5D%2C%5B1%2C1%2C1%2C1%2C%22H%22%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C%22Measure%22%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Z%5E%C2%BD%22%2C%22Z%5E%C2%BD%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22zpar%22%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22X%22%2C%22X%22%5D%2C%5B1%2C%22zpar%22%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22X%22%2C1%2C%22X%22%5D%2C%5B1%2C1%2C%22zpar%22%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22X%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C%22zpar%22%2C1%2C1%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C%22zpar%22%2C1%2C1%2C1%2C1%2C1%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Measure%22%2C%22Measure%22%2C%22Measure%22%5D%2C%5B%22Z%22%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C%22zpar%22%2C1%2C%22zpar%22%2C%22zpar%22%5D%2C%5B1%2C%22Z%22%2C1%2C1%2C1%2C1%2C1%2C%22zpar%22%2C1%2C1%2C%22zpar%22%2C%22zpar%22%2C1%2C%22zpar%22%5D%2C%5B1%2C1%2C%22Z%22%2C1%2C%22zpar%22%2C1%2C%22zpar%22%2C%22zpar%22%2C1%2C%22zpar%22%2C%22zpar%22%2C1%2C%22zpar%22%2C%22zpar%22%5D%2C%5B1%2C1%2C1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C%22Z%22%5D%2C%5B%22Z%22%2C%22Z%22%2C1%2C1%2C1%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%22%2C1%2C%22Z%22%2C1%2C%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22Z%22%2C1%2C%22%E2%80%A2%22%2C1%2C1%2C%22%E2%80%A2%22%5D%2C%5B%22~un0b%22%2C%22~pqmd%22%2C%22~1uop%22%2C%22~sbi5%22%2C1%2C%22~1lob%22%5D%2C%5B%22zpar%22%2C%22X%22%2C%22zpar%22%5D%2C%5B1%2C%22Swap%22%2C1%2C%22Swap%22%5D%2C%5B1%2C1%2C1%2C%22Swap%22%2C1%2C%22Swap%22%5D%2C%5B%22~un0b%22%2C%22~sbi5%22%2C%22~1uop%22%2C%22~1lob%22%5D%2C%5B%22inputA2%22%2C1%2C%22-%3DA2%22%5D%2C%5B%22~un0b%22%2C%22~sbi5%22%2C%22~pqmd%22%2C%22~ef5k%22%5D%2C%5B1%2C%22X%5E-t%22%2C1%2C%22Z%5E-t%22%5D%2C%5B%22X%5E-t%22%2C%22X%5E-t%22%2C%22Y%5E-t%22%2C%22X%5E-t%22%5D%2C%5B%22Uncounting4%22%5D%5D%2C%22gates%22%3A%5B%7B%22id%22%3A%22~un0b%22%2C%22name%22%3A%22a_k%22%2C%22matrix%22%3A%22%7B%7B1%2C0%7D%2C%7B0%2C1%7D%7D%22%7D%2C%7B%22id%22%3A%22~pqmd%22%2C%22name%22%3A%22b_k%22%2C%22matrix%22%3A%22%7B%7B1%2C0%7D%2C%7B0%2C1%7D%7D%22%7D%2C%7B%22id%22%3A%22~sbi5%22%2C%22name%22%3A%22a_k%2B1%22%2C%22matrix%22%3A%22%7B%7B1%2C0%7D%2C%7B0%2C1%7D%7D%22%7D%2C%7B%22id%22%3A%22~ef5k%22%2C%22name%22%3A%22b_k%2B1%22%2C%22matrix%22%3A%22%7B%7B1%2C0%7D%2C%7B0%2C1%7D%7D%22%7D%2C%7B%22id%22%3A%22~1uop%22%2C%22name%22%3A%22s_k%22%2C%22matrix%22%3A%22%7B%7B1%2C0%7D%2C%7B0%2C1%7D%7D%22%7D%2C%7B%22id%22%3A%22~1lob%22%2C%22name%22%3A%22s_k%2B1%22%2C%22matrix%22%3A%22%7B%7B1%2C0%7D%2C%7B0%2C1%7D%7D%22%7D%5D%7D
        input 0:a[k-1]
        input 1:b[k-1]
        input 2:(a+b)[k-1]
        input 3:a[k]
        input 4:b[k]
        tick
        tick
        highlight_box_start
        input 5:|CCZ⟩
        input 8:|i⟩
        input 9:|i⟩
        input 10:|i⟩
        tick
        C 5:Z__ 8:X__
        C 5:Z_Z 8:_X_
        C 5:ZZ_ 8:__X
        C 5:H
        highlight_box_end 5:Semi-Auto\nMagic State
        tick
        highlight_box_start
        C 0:ZZZZZ 5:X
        C 0:_ZZ 5:_X_
        C 0:Z_Z 5:__X
        M 6:Z 7:Z
        highlight_box_end 0:Interact
        tick
        highlight_box_start
        C 5:_@@ 8:H__
        C 5:_@_ 8:_H_
        C 5:__@ 8:__H
        M 8:X 9:X 10:X
        highlight_box_end 6:Reaction-Limited\nBasis Choice
        tick
        halftick
        highlight_box_start
        C 5:_@@@@@ 3:__X
        highlight_box_end 5:Backdated\nX Frame
        tick
        halftick
        highlight_box_start
        tick
        halftick
        annotate 0:a[k-1]
        annotate 1:b[k-1]
        annotate 2:(a+b)[k-1]
        annotate 3:a[k]
        annotate 4:b[k]
        annotate 5:(a+b)[k]
        halftick
        tick
        tick
        highlight_box_end 0:Rest of Adder\n(recurse with k+=1)
        tick
        highlight_box_start
        input 11:|i⟩
        input 12:|i⟩
        input 13:|i⟩
        M 4:X
        C 0:ZZ_ 11:X__
        C 0:Z_Z 11:_X_
        C 0:_ZZ 11:__X
        highlight_box_end 0:Start Uncomputing\nInput b
        tick
        highlight_box_start
        C 4:@___ 11:H__
        C 4:@_@_ 11:_H_
        C 4:@__@ 11:__H
        M 11:X 12:X 13:X
        highlight_box_end 4:Reaction-Limited\nBasis Choice
        tick
        highlight_box_start
        C 0:Z 4:__@__@_@@
        C 1:Z 4:___@__@@_@
        C 2:Z 4:@_@@_@@_@@
        C 3:Z 4:@
        C 5:Z 4:@
        C 0:ZZ_ 6:@ 7:@
        C 0:Z_Z 4:@ 6:@
        C 0:_ZZ 4:@ 7:@
        highlight_box_end 0:Backdated\nZ Frame
        drop 4 6 7 8 9 10 11 12 13
        tick
        output 0:a[k-1]
        output 1:b[k-1]
        output 2:(a+b)[k-1]
        output 3:a[k]
        output 5:(a+b)[k]
    """
    )
    return program


def verify_adder_program(program: CircuitProgram) -> None:
    in2q = {"a0": 0, "b0": 1, "s0": 2, "a1": 3, "b1": 4}
    out2q = {"a0": 0, "b0": 1, "s0": 2, "a1": 3, "s1": 5}
    program.verify(
        shots=10,
        in2q=in2q,
        out2q=out2q,
        in_func=lambda c: {**c, "s0": c["a0"] ^ c["b0"]},
        out_func=lambda c: {
            **{k: v for k, v in c.items() if k != "b1"},
            "s0": c["a0"] ^ c["b0"],
            "s1": (((c["a0"] + c["b0"]) + 2 * (c["a1"] + c["b1"])) >> 1) & 1,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--show_url', action='store_true')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    d = CircuitDrawer()
    d.q_offset = 1
    p = make_adder_program()
    d.draw_program(p)
    d.write_to(out_dir / "adder-circuit-diagram.svg")
    if args.show_url:
        print("URL", p.to_quirk_link(), file=sys.stderr)
        print("Latex URL", file=sys.stderr)
        print("    ", p.to_quirk_link(latex_escape=True), file=sys.stderr)
        print(file=sys.stderr)
    verify_adder_program(p)

    # d = CircuitDrawer()
    # d.q_offset = 1
    # draw_basic_tof_teleport(d)
    # d.write_to(out_dir / "tof-teleport.svg")
    # print("URL", p.to_quirk_link(), file=sys.stderr)
    # print("Latex URL", file=sys.stderr)
    # print("    ", p.to_quirk_link(latex_escape=True), file=sys.stderr)
    # print(file=sys.stderr)
    #
    # d = CircuitDrawer()
    # d.q_offset = 1
    # draw_deferred_teleport(d)
    # d.write_to(out_dir / "tof-teleport-deferred.svg")
    # print("URL", p.to_quirk_link(), file=sys.stderr)
    # print("Latex URL", file=sys.stderr)
    # print("    ", p.to_quirk_link(latex_escape=True), file=sys.stderr)
    # print(file=sys.stderr)


if __name__ == "__main__":
    main()
