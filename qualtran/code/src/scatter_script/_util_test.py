import random

import numpy as np

from scatter_script._util import slice_int, np_bit_count


def test_slice_int():
    class S:
        def __init__(self, v):
            self.v = v

        def __getitem__(self, item):
            return slice_int(self.v, item)

    assert S(0b111000111011)[0] == 1

    assert S(0b111000111010)[0] == 0
    assert S(0b111000111000)[1] == 0
    assert S(0b111000111010)[1] == 1
    assert S(0b111000111010)[2] == 0
    assert S(0b111000111010)[3] == 1
    assert S(0b111000111010)[4] == 1
    assert S(0b111000111010)[5] == 1
    assert S(0b111000111010)[6] == 0
    assert S(-1)[0] == 1
    assert S(-1)[99] == 1

    assert S(0b111000111010)[:4] == 0b1010
    assert S(0b111000111010)[1:5] == 0b1101
    assert S(0b111000111010)[4:] == 0b11100011
    assert S(-1)[43:77] == (1 << (77 - 43)) - 1


def test_np_bit_count():
    np.testing.assert_array_equal(
        np_bit_count(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint64)),
        np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.uint64),
    )

    inp = [random.randrange(1 << k) for k in range(64)]
    expected = [v.bit_count() for v in inp]
    np.testing.assert_array_equal(np_bit_count(np.array(inp)), np.array(expected))

    np.testing.assert_array_equal(np_bit_count(np.array([2**64 - 1])), np.array([64]))
    np.testing.assert_array_equal(np_bit_count(np.array([2**63 + 1])), np.array([2]))

    inp = [random.randrange(2**64) for _ in range(30)]
    expected = [v.bit_count() for v in inp]
    np.testing.assert_array_equal(
        np_bit_count(np.array(inp, dtype=np.uint64)), np.array(expected, dtype=np.uint64)
    )
