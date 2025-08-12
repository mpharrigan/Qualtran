from __future__ import annotations

import collections
import random

import numpy as np
import pytest

from facto.operations.phaseup._phaseup_imp import (
    do_sqrt_phaseup,
    estimate_cost_of_sqrt_phaseup,
    table2power_conversion_matrix,
    do_wandering_power_product,
    undo_wandering_power_product,
)
from scatter_script import CostKey, QPU, quint


def do_power_product(qpu: QPU, register: quint) -> list[int | quint]:
    result: list[int | quint] = [1]
    for q in register:
        for k in range(len(result)):
            if k == 0:
                result.append(q)
            else:
                result.append(qpu.alloc_and(result[k], q))
    return result


def undo_power_product(qpu: QPU, result: list[quint]) -> None:
    result = list(result)
    while result:
        h = len(result) >> 1
        czs = np.zeros(shape=h, dtype=np.uint8)
        for k in range(1, h)[::-1]:
            if result[h + k].del_measure_x():
                czs[k] ^= 1
        qpu.cz_multi_target(result[h], result[:h], czs)
        result = result[:h]


@pytest.mark.parametrize("n", range(10))
def test_do_phaseup(n: int):
    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=n, scatter=True)
        table = [random.random() < 0.5 for _ in range(1 << n)]
        addr.phaseflip_by_lookup(table)
        do_sqrt_phaseup(qpu, addr, table)
        addr.UNPHYSICAL_force_del(dealloc=True)
        qpu.verify_clean_finish()


@pytest.mark.parametrize("n", range(10))
def test_estimate_cost_of_sqrt_phaseup(n: int):
    qpu = QPU(num_branches=1)
    addr = qpu.alloc_quint(length=n, scatter=True)
    shots = 1
    for _ in range(shots):
        do_sqrt_phaseup(qpu, addr, [random.random() < 0.5 for _ in range(1 << n)])

    actual = collections.Counter()
    for k, v in qpu.cost_counters.items():
        if k.name == "CZ_multi_target":
            k = CostKey("CZ_multi_target")
        actual[k] += v / shots

    estimated = estimate_cost_of_sqrt_phaseup(n)

    actual.pop(CostKey("qubits"), None)
    assert actual.keys() <= estimated.keys()
    differences = []
    for k in estimated.keys():
        a = actual[k]
        e = estimated[k]
        if a != e:
            differences.append((k, a, e))
    assert not differences, differences


def test_exact2power_index_conversion_matrix():
    n = 4
    m = table2power_conversion_matrix(n)
    for _ in range(20):
        table = np.array([[random.random() < 0.5] for _ in range(1 << n)], dtype=np.uint8)
        table2 = (m @ table) & 1
        address = random.randrange(1 << n)
        expected = table[address]

        phase = 0
        for mask in range(1 << n):
            if table2[mask]:
                phase ^= address & mask == mask
        assert phase == expected

    np.set_printoptions(threshold=2000, linewidth=2000)
    text = str(m).replace("0", ".").replace(" ", "").replace("[", "").replace("]", "")
    assert (
        text
        == """
1...............
11..............
1.1.............
1111............
1...1...........
11..11..........
1.1.1.1.........
11111111........
1.......1.......
11......11......
1.1.....1.1.....
1111....1111....
1...1...1...1...
11..11..11..11..
1.1.1.1.1.1.1.1.
1111111111111111
""".strip()
    )


def test_undo_power_product():
    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=3, scatter=True)
        original_addr = addr.UNPHYSICAL_copy()
        vs = do_power_product(qpu, addr)
        assert len(vs) == 8
        undo_power_product(qpu, vs)
        assert addr == original_addr
        addr.UNPHYSICAL_force_del(dealloc=True)
        qpu.verify_clean_finish()


def test_do_power_product():
    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=0, scatter=True)
        vs = do_power_product(qpu, addr)
        assert len(vs) == 1
        assert vs[0] == 1

    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=1, scatter=True)
        vs = do_power_product(qpu, addr)
        assert len(vs) == 2
        assert vs[0] == 1
        assert vs[1] == addr[0]

    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=2, scatter=True)
        vs = do_power_product(qpu, addr)
        assert len(vs) == 4
        assert vs[0] == 1
        assert vs[1] == addr[0]
        assert vs[2] == addr[1]
        assert vs[3] == addr[0] & addr[1]

    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=3, scatter=True)
        vs = do_power_product(qpu, addr)
        assert len(vs) == 8
        assert vs[0] == 1
        assert vs[1] == addr[0]
        assert vs[2] == addr[1]
        assert vs[3] == addr[0] & addr[1]
        assert vs[4] == addr[2]
        assert vs[5] == addr[0] & addr[2]
        assert vs[6] == addr[1] & addr[2]
        assert vs[7] == addr[0] & addr[1] & addr[2]

    for _ in range(32):
        qpu = QPU(num_branches=20)
        addr = qpu.alloc_quint(length=4, scatter=True)
        vs = do_power_product(qpu, addr)
        assert len(vs) == 16
        assert vs[0] == 1
        assert vs[1] == addr[0]
        assert vs[2] == addr[1]
        assert vs[3] == addr[0] & addr[1]
        assert vs[4] == addr[2]
        assert vs[5] == addr[0] & addr[2]
        assert vs[6] == addr[1] & addr[2]
        assert vs[7] == addr[0] & addr[1] & addr[2]
        assert vs[8] == addr[3]
        assert vs[9] == addr[0] & addr[3]
        assert vs[10] == addr[1] & addr[3]
        assert vs[11] == addr[0] & addr[1] & addr[3]
        assert vs[12] == addr[2] & addr[3]
        assert vs[13] == addr[0] & addr[2] & addr[3]
        assert vs[14] == addr[1] & addr[2] & addr[3]
        assert vs[15] == addr[0] & addr[1] & addr[2] & addr[3]


def test_do_uncorrected_power_product():
    qpu = QPU(num_branches=25)
    addr = qpu.alloc_quint(length=4, scatter=True)
    actual = do_power_product(qpu, addr)

    cmp, mat, inv_mat, power_product = do_wandering_power_product(qpu, addr)

    for col in range(len(actual)):
        reconstruct = 0
        for row in range(len(actual)):
            if mat[row, col]:
                reconstruct = reconstruct ^ actual[row]
        assert reconstruct == power_product[col]

    for col in range(len(actual)):
        reconstruct = 0
        for row in range(len(actual)):
            if inv_mat[row, col]:
                reconstruct = reconstruct ^ power_product[row]
        assert reconstruct == actual[col]

    undo_wandering_power_product(qpu, power_product, cmp_data_from_computation=cmp)
    undo_power_product(qpu, actual)
    addr.UNPHYSICAL_force_del(dealloc=True)
    qpu.verify_clean_finish()
