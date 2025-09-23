from __future__ import annotations

import math
import multiprocessing
import sys
import time
from typing import Sequence, Any

import numpy as np

from facto.algorithm.prep._problem_config import ProblemConfig


def _solve_cases(arg: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    period, generator, period_dtype, num_windows3a, num_windows3b, w3a, w3b = arg

    table3a = np.zeros(shape=(num_windows3a, num_windows3b, 1 << (w3a + w3b)), dtype=period_dtype)
    table3b = np.zeros(shape=(num_windows3a, num_windows3b, 1 << (w3a + w3b)), dtype=period_dtype)

    for ka in range(num_windows3a):
        for kb in range(num_windows3b):
            for k4a in range(1 << w3a):
                v_base = pow(generator, k4a << (ka * w3a), period)
                v2_base = -pow(v_base, -1, period)
                v_base <<= kb * w3b
                v2_base <<= kb * w3b
                v_base %= period
                v2_base %= period
                for k4b in range(1 << w3b):
                    addr = (k4a << w3b) | k4b
                    table3a[ka, kb, addr] = -v_base * k4b % period
                    table3b[ka, kb, addr] = -v2_base * k4b % period

    total = 1
    table3c = np.zeros(shape=1 << (2 * w3a), dtype=period_dtype)
    for k2 in range(1 << (w3a * 2)):
        table3c[k2] = total
        total *= generator
        total %= period

    return table3a, table3b, table3c


def precompute_table3(
    *,
    conf: ProblemConfig,
    periods: Sequence[int],
    generators: Sequence[int],
    period_dtype: np.dtype,
    print_progress: bool,
    pool: multiprocessing.Pool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if print_progress:
        print("    computing small windowed modexp offsets...", file=sys.stderr)
    t0 = time.monotonic()
    table3a = np.zeros(
        shape=(
            len(periods),
            conf.num_windows3a,
            conf.num_windows3b,
            1 << (conf.window3a + conf.window3b),
        ),
        dtype=period_dtype,
    )
    table3b = np.zeros(
        shape=(
            len(periods),
            conf.num_windows3a,
            conf.num_windows3b,
            1 << (conf.window3a + conf.window3b),
        ),
        dtype=period_dtype,
    )
    table3c = np.zeros(shape=(len(periods), 1 << (conf.window3a * 2)), dtype=period_dtype)
    inputs = [
        (p, g, period_dtype, conf.num_windows3a, conf.num_windows3b, conf.window3a, conf.window3b)
        for g, p in zip(generators, periods)
    ]
    own_pool = False
    if pool is None:
        pool = multiprocessing.Pool()
        own_pool = True
    for k, (result_a, result_b, result_c) in enumerate(pool.imap(_solve_cases, inputs)):
        if print_progress and min(k, len(inputs) - k).bit_count() == 1:
            print(
                f"    solved so far: {k}/{len(periods)} (elapsed={math.ceil(time.monotonic()-t0)}s)",
                file=sys.stderr,
            )
        table3a[k] = result_a
        table3b[k] = result_b
        table3c[k] = result_c
    if own_pool:
        pool.close()

    return table3a, table3b, table3c
