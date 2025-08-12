from __future__ import annotations

import multiprocessing
from typing import Any

from facto.algorithm.prep._problem_config import ProblemConfig


def _x(arg: dict[str, Any]):
    slice_start: int = int(arg["slice_start"])
    window1: int = int(arg["window1"])
    modulus: int = int(arg["modulus"])
    generator: int = int(arg["generator"])
    result = []
    for k in range(1 << window1):
        v = pow(generator, k << slice_start, modulus)
        result.append((slice_start, k, v))
    return result


def find_multipliers_for_conf(conf: ProblemConfig) -> dict[tuple[int, int], int]:
    pool = multiprocessing.Pool()
    results = pool.map(
        _x,
        [
            {
                "slice_start": slice_start,
                "modulus": conf.modulus,
                "window1": conf.window1,
                "generator": conf.generator,
            }
            for slice_start in range(0, conf.num_input_qubits, conf.window1)
        ],
    )
    pool.close()

    multipliers = {}
    for r in results:
        for slice_start, slice_value, output in r:
            multipliers[(slice_start, slice_value)] = output
    return multipliers
