import math

from facto.algorithm.prep._precompute_rns import find_rns_for_conf
from facto.algorithm.prep._problem_config import ProblemConfig


def test_find_rns_for_conf():
    conf = ProblemConfig.from_ini_content(
        """
        modulus = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
        generator = 3
        num_input_qubits = 165
        min_wraparound_gap = 20

        window1 = 5
        window3a = 2
        window3b = 3
        window4 = 1
        len_accumulator = 1
        mask_bits = 1
        num_shots = 1
    """
    )
    solution = find_rns_for_conf(conf)
    t = math.prod(solution.periods)
    assert t.bit_length() >= math.ceil(
        conf.modulus.bit_length() * conf.num_input_qubits / conf.window1
    )
    assert (t % conf.modulus).bit_length() + conf.min_wraparound_gap <= conf.modulus.bit_length()
    assert set(p.bit_length() for p in solution.periods) == {15}
