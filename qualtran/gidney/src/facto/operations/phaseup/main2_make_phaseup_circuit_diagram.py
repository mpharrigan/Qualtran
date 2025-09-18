from __future__ import annotations

import argparse
import pathlib
import sys

from facto.operations._circuit_diagram_drawer import CircuitDrawer, CircuitProgram


def make_phase_lookup_program_7() -> CircuitProgram:
    return CircuitProgram.from_program_text(
        r"""
        input 00:a₀
        input 01:a₁
        input 03:a₂
        input 07:a₃
        input 08:a₄
        input 10:a₅
        input 14:a₆

        tick
        tick
        highlight_box_start
        highlight_box_start

        C 0:Z 1:Z 2:X
        C 7:Z 8:Z 9:X
        tick
        tick

        C 0:Z 3:Z 4:X
        C 7:Z 10:Z 11:X
        C 1:Z 3:Z 5:X
        C 8:Z 10:Z 12:X
        C 2:Z 3:Z 6:X
        C 9:Z 10:Z 13:X
        tick
        tick

        C 07:Z 14:Z 15:X
        C 08:Z 14:Z 16:X
        C 09:Z 14:Z 17:X
        C 10:Z 14:Z 18:X
        C 11:Z 14:Z 19:X
        C 12:Z 14:Z 20:X
        C 13:Z 14:Z 21:X
        tick
        highlight_box_end 0:Produce all control products\nfor the low half and high half of the address
        highlight_box_end 7:
        tick

        annotate 0:a₀
        annotate 1:a₁
        annotate 2:a₀·a₁
        annotate 3:a₂
        annotate 4:a₀·a₂
        annotate 5:a₀·a₂
        annotate 6:a₀·a₁·a₂

        annotate 7:a₃
        annotate 8:a₄
        annotate 9:a₃·a₄
        annotate 10:a₅
        annotate 11:a₃·a₅
        annotate 12:a₄·a₅
        annotate 13:a₃·a₄·a₅
        annotate 14:a₆
        annotate 15:a₃·a₆
        annotate 16:a₄·a₆
        annotate 17:a₃·a₄·a₆
        annotate 18:a₅·a₆
        annotate 19:a₃·a₅·a₆
        annotate 20:a₄·a₅·a₆
        annotate 21:a₃·a₄·a₅·a₆
        tick
        tick

        highlight_box_start
        C 0:Z 7:???????????????
        C 1:Z 7:???????????????
        C 2:Z 7:???????????????
        C 3:Z 7:???????????????
        C 4:Z 7:???????????????
        C 5:Z 7:???????????????
        C 6:Z 7:???????????????
        tick
        C 0:?
        C 1:?
        C 2:?
        C 3:?
        C 4:?
        C 5:?
        C 6:?
        C 7:?
        C 8:?
        C 9:?
        C 10:?
        C 11:?
        C 12:?
        C 13:?
        C 14:?
        C 15:?
        C 16:?
        C 17:?
        C 18:?
        C 19:?
        C 20:?
        C 21:?
        tick
        highlight_box_end 0:Perform data-driven phasing\n(each ? is Z/I based on table data)
        tick
        
        highlight_box_start
        highlight_box_start
        M 15:X 16:X 17:X 18:X 19:X 20:X 21:X
        C 13:Z 14:Z 21:C
        drop 21
        C 12:Z 14:Z 20:C
        drop 20
        C 11:Z 14:Z 19:C
        drop 19
        C 10:Z 14:Z 18:C
        drop 18
        C 09:Z 14:Z 17:C
        drop 17
        C 08:Z 14:Z 16:C
        drop 16
        C 07:Z 14:Z 15:C
        drop 15
        tick
        tick

        M 4:X 5:X 6:X 11:X 12:X 13:X
        C 2:Z 3:Z 6:C
        drop 6
        C 9:Z 10:Z 13:C
        drop 13
        C 1:Z 3:Z 5:C
        drop 5
        C 8:Z 10:Z 12:C
        drop 12
        C 0:Z 3:Z 4:C
        drop 4
        C 7:Z 10:Z 11:C
        drop 11
        tick
        tick

        M 2:X 9:X
        C 0:Z 1:Z 2:C
        C 7:Z 8:Z 9:C
        drop 2 9
        tick
        highlight_box_end 0:Uncompute control products
        highlight_box_end 7:
        tick

        output 00:a₀
        output 01:a₁
        output 03:a₂
        output 07:a₃
        output 08:a₄
        output 10:a₅
        output 14:a₆
    """
    )


def make_phase_lookup_program_6() -> CircuitProgram:
    return CircuitProgram.from_program_text(
        r"""
        input 00:a₀
        input 01:a₁
        input 03:a₂
        input 07:a₃
        input 08:a₄
        input 10:a₅

        tick
        tick
        highlight_box_start
        highlight_box_start

        C 0:Z 1:Z 2:X
        C 7:Z 8:Z 9:X
        tick
        tick

        C 0:Z 3:Z 4:X
        C 7:Z 10:Z 11:X
        C 1:Z 3:Z 5:X
        C 8:Z 10:Z 12:X
        C 2:Z 3:Z 6:X
        C 9:Z 10:Z 13:X
        tick
        tick

        highlight_box_end 0:Produce all control products\nfor the low half and high half of the address
        highlight_box_end 7:
        tick

        annotate 0:a₀
        annotate 1:a₁
        annotate 2:a₀·a₁
        annotate 3:a₂
        annotate 4:a₀·a₂
        annotate 5:a₀·a₂
        annotate 6:a₀·a₁·a₂

        annotate 7:a₃
        annotate 8:a₄
        annotate 9:a₃·a₄
        annotate 10:a₅
        annotate 11:a₃·a₅
        annotate 12:a₄·a₅
        annotate 13:a₃·a₄·a₅
        tick
        tick

        highlight_box_start
        C 0:Z 7:???????
        C 1:Z 7:???????
        C 2:Z 7:???????
        C 3:Z 7:???????
        C 4:Z 7:???????
        C 5:Z 7:???????
        C 6:Z 7:???????
        tick
        C 0:?
        C 1:?
        C 2:?
        C 3:?
        C 4:?
        C 5:?
        C 6:?
        C 7:?
        C 8:?
        C 9:?
        C 10:?
        C 11:?
        C 12:?
        C 13:?
        tick
        highlight_box_end 0:Perform data-driven phasing\n(each ? is Z/I based on table data)
        tick

        highlight_box_start
        highlight_box_start

        M 4:X 5:X 6:X 11:X 12:X 13:X
        C 2:Z 3:Z 6:C
        drop 6
        C 9:Z 10:Z 13:C
        drop 13
        C 1:Z 3:Z 5:C
        drop 5
        C 8:Z 10:Z 12:C
        drop 12
        C 0:Z 3:Z 4:C
        drop 4
        C 7:Z 10:Z 11:C
        drop 11
        tick
        tick

        M 2:X 9:X
        C 0:Z 1:Z 2:C
        C 7:Z 8:Z 9:C
        drop 2 9
        tick
        highlight_box_end 0:Uncompute control products
        highlight_box_end 7:
        tick

        output 00:a₀
        output 01:a₁
        output 03:a₂
        output 07:a₃
        output 08:a₄
        output 10:a₅
    """
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
    p = make_phase_lookup_program_7()
    d.draw_program(p)
    d.write_to(out_dir / "phaseup-7q.svg")

    d = CircuitDrawer()
    d.q_offset = 1
    p = make_phase_lookup_program_6()
    d.draw_program(p)
    d.write_to(out_dir / "phaseup-6q.svg")
    if args.show_url:
        print("URL", p.to_quirk_link(), file=sys.stderr)
        print("Latex URL", file=sys.stderr)
        print("    ", p.to_quirk_link(latex_escape=True), file=sys.stderr)
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
