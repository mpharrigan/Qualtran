from __future__ import annotations

import pytest

from scatter_script._qpu import QPU


def test_phase_uncompute():
    qpu = QPU(num_branches=32)
    for _ in range(4):
        r = qpu.alloc_quint(length=3, scatter=True)
        r.del_measure_qft()
    with pytest.raises(ValueError, match="phases weren't exactly uncomputed"):
        qpu.verify_clean_finish()


def test_alloc_uncorrected_and():
    qpu = QPU(num_branches=16)
    a = qpu.alloc_quint(length=1, scatter=True)
    b = qpu.alloc_quint(length=1, scatter=True)
    cmp1, cmp2, c = qpu.alloc_wandering_and(a, b)
    assert c == (a ^ cmp1) & (b ^ cmp2)

    if cmp1:
        c ^= b
    if cmp2:
        c ^= a
    if cmp1 and cmp2:
        c ^= 1
    assert c == a & b
    c.UNPHYSICAL_force_del(dealloc=False)
    cmp1, cmp2, _ = qpu.alloc_wandering_and(a, b, out=c)
    assert c == (a ^ cmp1) & (b ^ cmp2)

    a.UNPHYSICAL_force_del(dealloc=True)
    b.UNPHYSICAL_force_del(dealloc=True)
    c.UNPHYSICAL_force_del(dealloc=True)
    qpu.verify_clean_finish()


def test_alloc_and():
    qpu = QPU(num_branches=16)
    a = qpu.alloc_quint(length=3, scatter=True)
    b = qpu.alloc_quint(length=3, scatter=True)
    c = qpu.alloc_and(a, b)
    assert len(c) == 3
    assert c == a & b
    c.UNPHYSICAL_force_del(dealloc=False)
    _ = qpu.alloc_and(a, b, out=c)
    assert c == a & b

    a.UNPHYSICAL_force_del(dealloc=True)
    b.UNPHYSICAL_force_del(dealloc=True)
    c.UNPHYSICAL_force_del(dealloc=True)
    qpu.verify_clean_finish()
