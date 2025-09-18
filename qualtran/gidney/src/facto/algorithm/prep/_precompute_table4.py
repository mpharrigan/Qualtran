from __future__ import annotations

import math
import multiprocessing
import sys
import time
from typing import Sequence

import numpy as np

from facto.algorithm.prep._problem_config import ProblemConfig


def _solve_truncated_offsets(args: dict[str, int]) -> np.ndarray:
    window4 = args["window4"]
    total_period = args["total_period"]
    p = args["p"]
    modulus = args["modulus"]
    dropped_bits = args["dropped_bits"]
    u = total_period // p
    u *= pow(u, -1, p)
    rng1 = range(0, p.bit_length(), window4)
    rng2 = range(1 << window4)
    results = np.zeros(shape=(len(rng1), len(rng2)), dtype=np.uint64)
    for k1, v1 in enumerate(rng1):
        for k2, v2 in enumerate(rng2):
            w = ((u << v1) * v2 % total_period % modulus) >> dropped_bits
            results[k1, k2] = -w % (modulus >> dropped_bits)
    return results


def precompute_table4(
    *, conf: ProblemConfig, periods: Sequence[int], print_progress: bool = False
) -> np.ndarray:
    periods = tuple(periods)
    total_period = math.prod(periods)
    pool = multiprocessing.Pool()

    t0 = time.monotonic()
    iter_results = pool.imap(
        _solve_truncated_offsets,
        [
            {
                "window4": conf.window4,
                "total_period": total_period,
                "modulus": conf.modulus,
                "p": p,
                "dropped_bits": conf.dropped_bits,
            }
            for p in periods
        ],
    )
    table3 = np.zeros(shape=(len(periods), conf.num_windows4, 1 << conf.window4), dtype=np.uint64)
    for k, result in enumerate(iter_results):
        if print_progress and min(k, len(periods) - k).bit_count() == 1:
            print(
                f"    solved so far: {k}/{len(periods)} (elapsed={math.ceil(time.monotonic()-t0)}s)",
                file=sys.stderr,
            )
        table3[k] = result

    pool.close()
    return table3
