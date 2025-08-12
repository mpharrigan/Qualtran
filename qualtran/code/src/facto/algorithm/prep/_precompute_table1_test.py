from __future__ import annotations

import numpy as np

from facto.algorithm.prep._precompute_table1 import precompute_table1


def test_precompute_dlogs():
    periods = [19, 23, 101, 49331]
    values = [10001, 432886431]
    generators = [3, 5, 3, 6]
    result = precompute_table1(
        periods=periods, generators=generators, values=values, period_dtype=np.uint64
    )

    for k2 in range(len(values)):
        total = 0
        for k in range(len(periods)):
            total += int(result[k, k2])
            total %= 2**64
            assert pow(generators[k], total, periods[k]) == values[k2] % periods[k]
        assert result[len(periods), k2] == -total % 2**64
    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [6, 6],
                [(15 - 6) % 2**64, (4 - 6) % 2**64],
                [(29 - 15) % 2**64, (3 - 4) % 2**64],
                [(31743 - 29) % 2**64, (36353 - 3) % 2**64],
                [(-31743) % 2**64, (-36353) % 2**64],
            ],
            dtype=np.uint64,
        ),
    )
