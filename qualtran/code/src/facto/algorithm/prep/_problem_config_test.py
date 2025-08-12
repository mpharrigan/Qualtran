from __future__ import annotations

import io
import math

import pytest

from facto.algorithm.prep import ExecutionConfig
from facto.algorithm.prep._problem_config import ProblemConfig
from facto.algorithm.prep._residue_util import prime_count_and_capacity_at_bit_length


def test_from_ini_path():
    s = io.StringIO()
    s.write(
        """
        modulus = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
        num_input_qubits = 11
        generator = 66
        
        ; comment
        window1 = 5
        window3a = 2
        window3b = 3
        window4 = 7
        min_wraparound_gap = 30
        len_accumulator = 40
        mask_bits = 10
        num_shots = 1
    """
    )
    s.seek(0)
    p = ProblemConfig.from_ini_content(s)
    assert (
        p.modulus
        == 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
    )
    assert p.window1 == 5
    assert p.window3a == 2
    assert p.window3b == 3
    assert p.window4 == 7
    assert p.min_wraparound_gap == 30
    assert p.generator == 66
    assert p.num_input_qubits == 11
    assert p.rns_primes_bit_length is None
    assert p.rns_primes_range_start is None
    assert p.rns_primes_range_stop is None
    assert p.rns_primes_skipped is None
    assert p.rns_primes_extra is None
    assert p.estimate_minimum_rns_period_bit_length() == 12


@pytest.mark.parametrize("n", [330, 512, 1024, 2048])
@pytest.mark.parametrize("m", [165, 1000, 1500, 2000])
@pytest.mark.parametrize("w1", [1, 2, 3, 4, 5, 6])
def test_estimate_minimum_rns_period_bit_length(n: int, m: int, w1: int):
    p = ExecutionConfig.vacuous_config(
        modulus_bitlength=n,
        num_input_qubits=m,
        num_periods=1,
        period_bitlength=1,
        w1=w1,
        w3a=1,
        w3b=1,
        len_acc=24,
    )
    ell = p.conf.estimate_minimum_rns_period_bit_length()
    required_cap = p.modulus.bit_length() * p.num_windows1
    actual_cap = prime_count_and_capacity_at_bit_length(ell)[1]
    ratio = actual_cap / required_cap
    assert 1 < ratio < 2.3
