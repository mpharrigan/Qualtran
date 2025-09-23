import pytest

import gen


def test_sorted_complex():
    assert gen.sorted_complex([1, 2j, 2, 1 + 2j]) == [2j, 1, 1 + 2j, 2]


def test_min_max_complex():
    with pytest.raises(ValueError):
        gen.min_max_complex([])
    assert gen.min_max_complex([], default=0) == (0, 0)
    assert gen.min_max_complex([], default=1 + 2j) == (1 + 2j, 1 + 2j)
    assert gen.min_max_complex([1j], default=0) == (1j, 1j)
    assert gen.min_max_complex([1j, 2]) == (0, 2 + 1j)
    assert gen.min_max_complex([1j + 1, 2]) == (1, 2 + 1j)


def test_xor_sorted():
    assert gen.xor_sorted([]) == []
    assert gen.xor_sorted([2]) == [2]
    assert gen.xor_sorted([2, 3]) == [2, 3]
    assert gen.xor_sorted([3, 2]) == [2, 3]
    assert gen.xor_sorted([2, 2]) == []
    assert gen.xor_sorted([2, 2, 2]) == [2]
    assert gen.xor_sorted([2, 2, 2, 2]) == []
    assert gen.xor_sorted([2, 2, 3]) == [3]
    assert gen.xor_sorted([3, 2, 2]) == [3]
    assert gen.xor_sorted([2, 3, 2]) == [3]
    assert gen.xor_sorted([2, 3, 3]) == [2]
    assert gen.xor_sorted([2, 3, 5, 7, 11, 13, 5]) == [2, 3, 7, 11, 13]
