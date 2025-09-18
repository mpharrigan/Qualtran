from __future__ import annotations

import math
import multiprocessing
import sys
import time
from typing import Iterable

from facto.algorithm.prep._residue_util import find_multiplicative_generator_modulo_prime_number


def precompute_generators(*, periods: Iterable[int], print_progress: bool = False) -> list[int]:
    periods: tuple[int, ...] = tuple(periods)

    pool = multiprocessing.Pool()
    t0 = time.monotonic()
    generators = []
    for k, e in enumerate(pool.map(find_multiplicative_generator_modulo_prime_number, periods)):
        generators.append(e)
        if print_progress and min(k, len(periods) - k).bit_count() == 1:
            print(
                f"    picked so far: {k}/{len(periods)} (completion={math.floor(k/len(periods)*100)}%, elapsed={math.ceil(time.monotonic()-t0)}s)",
                file=sys.stderr,
            )

    return generators
