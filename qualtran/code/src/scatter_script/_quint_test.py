from __future__ import annotations

import pytest

from scatter_script._quint import quint
from scatter_script._qpu import QPU


def test_quint_get_set():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    v = m[2:-3]
    assert int(v) == 0
    assert int(m) == 0
    assert v.UNPHYSICAL_branch_vals == (0, 0, 0)
    assert m.UNPHYSICAL_branch_vals == (0, 0, 0)

    v.UNPHYSICAL_write_branch_vals([5, 6, 7])
    with pytest.raises(NotImplementedError, match="unique"):
        _ = int(v)
    with pytest.raises(NotImplementedError, match="unique"):
        _ = int(m)
    assert v.UNPHYSICAL_branch_vals == (5, 6, 7)
    assert m.UNPHYSICAL_branch_vals == (5 * 4, 6 * 4, 7 * 4)

    v.UNPHYSICAL_write_branch_vals([-1, -1, -1])
    assert int(m) == 0b1111100
    assert v._offset == 2
    assert len(v) == 5


def test_add():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    v = m[2:-3]
    v += 2
    assert int(v) == 2
    v += 3
    assert int(v) == 5
    v += -1
    assert int(v) == 4


def test_sub():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    v = m[2:-3]
    v -= 2
    assert int(v) == 30
    v -= 3
    assert int(v) == 27
    v -= -1
    assert int(v) == 28


def test_iadd_mod():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    m.UNPHYSICAL_write_branch_vals((2, 3, 4))
    m.iadd_mod(2, modulus=5)
    assert m.UNPHYSICAL_branch_vals == (4, 0, 1)


def test_isub_mod():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    m.UNPHYSICAL_write_branch_vals((2, 3, 4))
    m.isub_mod(3, modulus=5)
    assert m.UNPHYSICAL_branch_vals == (4, 0, 1)


def test_imul_mod():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    m.UNPHYSICAL_write_branch_vals((2, 3, 4))
    m.imul_mod(2, modulus=5)
    assert m.UNPHYSICAL_branch_vals == (4, 1, 3)


def test_xor():
    qc = QPU(num_branches=3)
    m = qc.alloc_quint(length=10)
    v = m[2:-3]
    v ^= 2
    assert int(v) == 2
    v ^= 3
    assert int(v) == 1
    v ^= -1
    assert int(v) == 0b11110


def test_measurement_based_uncomputation_x():
    saw_phase = False
    for _ in range(32):
        qc = QPU(num_branches=32)
        a = qc.alloc_quint(length=1, scatter=True)
        b = qc.alloc_quint(length=1, scatter=True)
        c = qc.alloc_quint(length=1)
        c ^= a & b
        if c.del_measure_x():
            saw_phase |= any(qc.branch_phases_u64)
            a.cz(b)
        assert not any(qc.branch_phases_u64)
    assert saw_phase


def test_measurement_based_uncomputation_qft():
    saw_phase = False
    for _ in range(32):
        qc = QPU(num_branches=32)
        a = qc.alloc_quint(length=6, scatter=True)
        b = qc.alloc_quint(length=6, scatter=True)
        c = qc.alloc_quint(length=6)
        c += a
        c += b
        r = c.del_measure_qft()
        saw_phase |= any(qc.branch_phases_u64)
        a.phase_gradient(-r)
        b.phase_gradient(-r)
        assert not any(qc.branch_phases_u64)
    assert saw_phase
