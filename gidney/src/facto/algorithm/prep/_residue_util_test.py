import random

import numpy as np
import pytest

from facto.algorithm.prep._residue_util import (
    iter_set_bits_of,
    ceil_lg2,
    iter_primes,
    find_multiplicative_generator_modulo_prime_number,
    sig_fig_str,
    choose,
    bulk_discrete_log,
    table_str,
)


def test_iter_set_bits_of():
    assert list(iter_set_bits_of(0)) == []
    assert list(iter_set_bits_of(1)) == [0]
    assert list(iter_set_bits_of(2)) == [1]
    assert list(iter_set_bits_of(3)) == [0, 1]
    assert list(iter_set_bits_of(4)) == [2]
    assert list(iter_set_bits_of(5)) == [0, 2]
    assert list(iter_set_bits_of(6)) == [1, 2]
    assert list(iter_set_bits_of(7)) == [0, 1, 2]
    assert list(iter_set_bits_of(8)) == [3]
    assert list(iter_set_bits_of(9)) == [0, 3]


def test_ceil_lg2():
    assert ceil_lg2(1) == 0
    assert ceil_lg2(2) == 1
    assert ceil_lg2(3) == 2
    assert ceil_lg2(4) == 2
    assert ceil_lg2(5) == 3
    assert ceil_lg2(6) == 3
    assert ceil_lg2(7) == 3
    assert ceil_lg2(8) == 3
    assert ceil_lg2(9) == 4
    assert ceil_lg2(10) == 4
    assert ceil_lg2(11) == 4
    assert ceil_lg2(12) == 4
    assert ceil_lg2(13) == 4


def test_iter_primes():
    v = iter_primes()
    assert next(v) == 2
    assert next(v) == 3
    assert next(v) == 5
    assert next(v) == 7
    assert next(v) == 11
    assert next(v) == 13
    assert next(v) == 17
    assert next(v) == 19
    assert next(v) == 23
    assert next(v) == 29


@pytest.mark.parametrize("p", [5, 7, 11, 13, 17, 19, 23, 101, 49331])
def test_find_multiplicative_generator_modulo_prime_number(p: int):
    v = find_multiplicative_generator_modulo_prime_number(p)
    seen = set()
    b = v
    while b not in seen:
        seen.add(b)
        b *= v
        b %= p
    assert len(seen) == p - 1


def test_sig_fig_str():
    assert sig_fig_str(0) == "   0"
    assert sig_fig_str(1) == "   1"
    assert sig_fig_str(2) == "   2"
    assert sig_fig_str(9) == "   9"
    assert sig_fig_str(10) == "  10"
    assert sig_fig_str(99) == "  99"
    assert sig_fig_str(100) == " 100"
    assert sig_fig_str(999) == " 999"
    assert sig_fig_str(1000) == "1.0K"
    assert sig_fig_str(9999) == "9.9K"
    assert sig_fig_str(10000) == " 10K"
    assert sig_fig_str(99999) == " 99K"
    assert sig_fig_str(100000) == "100K"
    assert sig_fig_str(999_999) == "999K"
    assert sig_fig_str(1_000_000) == "1.0M"
    assert sig_fig_str(9_999_999) == "9.9M"
    assert sig_fig_str(10_000_000) == " 10M"
    assert sig_fig_str(1_000_000_000) == "1.0B"
    assert sig_fig_str(9_999_999_999) == "9.9B"
    assert sig_fig_str(10_000_000_000) == " 10B"
    assert sig_fig_str(1_000_000_000_000_000) == "1.0E15"
    assert sig_fig_str(9_999_999_999_999_999) == "9.9E15"
    assert sig_fig_str(10_000_000_000_000_000) == "10E15"


def test_choose():
    assert choose(0, 0) == 1
    assert choose(1, 0) == 1
    assert choose(1, 1) == 1
    assert choose(2, 0) == 1
    assert choose(2, 1) == 2
    assert choose(2, 2) == 1

    assert choose(5, 0) == 1
    assert choose(5, 1) == 5
    assert choose(5, 2) == 10
    assert choose(5, 3) == 10
    assert choose(5, 4) == 5
    assert choose(5, 5) == 1

    assert choose(99, 21) == 1613054714739084379224


def test_bulk_discrete_log():
    vals = [
        3206280,
        3206821,
        3207012,
        3208663,
        3209654,
        3210488,
        3213054,
        3213076,
        3213155,
        3213188,
        3213342,
        3213405,
    ]
    g = 10
    p = 3903331
    outs = bulk_discrete_log(g, vals, p)
    for v, o in zip(vals, outs):
        assert pow(g, o, p) == v


def test_table_str():
    assert (
        table_str({0: "test", 1: 5002, 1j: "---"}).strip()
        == """
 test | 5_002
------|
     """.strip()
    )
