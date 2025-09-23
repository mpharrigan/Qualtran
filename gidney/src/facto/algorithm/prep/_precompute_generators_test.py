from __future__ import annotations

from facto.algorithm.prep._precompute_generators import precompute_generators


def test_precompute_generators():
    periods = [5, 7, 11, 13, 17, 19, 23, 101, 49331]
    generators = precompute_generators(periods=periods)
    assert generators == [3, 3, 6, 6, 3, 3, 5, 3, 6]
