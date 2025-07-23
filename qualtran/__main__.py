import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from qualtran import Bloq, QInt, QUInt
from qualtran.l1._parse import BloqCode, parse_bloq_code


def dump_l1(bloq_code: BloqCode):
    bloq = bloq_code.load()
    print(f"Dumping {bloq}")
    print()

    from qualtran.l1 import bloqs_to_code

    bloqs_to_code(bloq)


def main(argv: Sequence[str] = sys.argv[1:]):
    """The main entrypoint for the qualtran command line interface."""
    parser = argparse.ArgumentParser(description="Qualtran command-line interface.")
    parser.add_argument("-c", "--compile", action="store_true", help="A boolean flag to compile.")
    parser.add_argument("-o", "--output", type=str, help="Output file name.")
    parser.add_argument("bloq", type=str, help="The bloq to process.")

    args = parser.parse_args(argv)
    bloq_code = parse_bloq_code(args.bloq)

    print(f"Compile flag: {args.compile}")
    print(f"Output file: {args.output}")
    print(f"Parsed Bloq: {bloq_code}")

    if args.compile:
        dump_l1(bloq_code)


if __name__ == "__main__":
    main()
