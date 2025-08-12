from __future__ import annotations

import math
import multiprocessing
import sys
import time
from typing import Iterable, Sequence

import numpy as np

from facto.algorithm.prep._residue_util import bulk_discrete_log


def _solve_cases(arg: tuple[int, tuple[int, ...], int, np.dtype]) -> np.ndarray:
    base, inps, modulus, period_dtype = arg
    if any(v == 0 for v in inps):
        raise ValueError(f"dlog({base=}, 0, {modulus=}) doesn't exist")
    return np.array(bulk_discrete_log(base, inps, modulus), dtype=period_dtype)


def precompute_table1(
    *,
    periods: Sequence[int],
    generators: Sequence[int],
    values: Iterable[int],
    print_progress: bool = False,
    period_dtype: np.dtype,
) -> np.ndarray:
    assert len(periods) == len(generators)
    values: tuple[int, ...] = tuple(values)

    pool = multiprocessing.Pool()
    if print_progress:
        print("    solving dlogs for each period...", file=sys.stderr)
    inputs = [(g, values, p, period_dtype) for g, p in zip(generators, periods)]
    t0 = time.monotonic()
    outputs_out = np.zeros(shape=(len(periods) + 1, len(values)), dtype=period_dtype)
    for k, result in enumerate(pool.imap(_solve_cases, inputs)):
        if print_progress and min(k, len(inputs) - k).bit_count() == 1:
            print(
                f"    solved so far: {k}/{len(periods)} (elapsed={math.ceil(time.monotonic()-t0)}s)",
                file=sys.stderr,
            )
        outputs_out[k] = result
    pool.close()

    outputs_out[1:] -= outputs_out[:-1]
    return outputs_out
