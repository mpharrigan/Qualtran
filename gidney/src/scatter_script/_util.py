from __future__ import annotations

from typing import Sequence

import numpy as np


def slice_int(x: int, s: int | slice) -> int:
    """Slices bits out of an integer and packs them into a result integer.

    Treats the integers as lists of bits, using 2s complement.
    """

    if isinstance(s, int):
        if s < 0:
            raise ValueError(
                f"Can't access negative index {s} because the intended length of an int is ambiguous."
            )
        return (x >> s) & 1

    if isinstance(s, slice):
        if (s.start is not None and s.start < 0) or (s.stop is not None and s.stop < 0):
            raise ValueError(
                f"Can't access negative slice {s} because the length of an int is ambiguous."
            )
        r = x
        if s.stop is not None:
            r &= ~(-1 << s.stop)
        if s.start is not None:
            r >>= s.start
        if s.step is not None and s.step != 1:
            raise NotImplementedError(f"Non-unit step in {s}")
        return r

    raise NotImplementedError(f"Don't know how to access {s!r} of {bin(x)}")


def np_bit_count(v: np.ndarray) -> np.ndarray:
    m1 = np.uint64(0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101)
    m2 = np.uint64(0b00110011_00110011_00110011_00110011_00110011_00110011_00110011_00110011)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    m8 = np.uint64(0x00FF00FF00FF00FF)
    m16 = np.uint64(0x0000FFFF0000FFFF)
    m32 = np.uint64(0x00000000FFFFFFFF)
    v = v.astype(np.uint64)
    v = ((v >> 1) & m1) + (v & m1)
    v = ((v >> 2) & m2) + (v & m2)
    v = ((v >> 4) & m4) + (v & m4)
    v = ((v >> 8) & m8) + (v & m8)
    v = ((v >> 16) & m16) + (v & m16)
    v = ((v >> 32) & m32) + (v & m32)
    return v


class ZeroArray:
    """Lightly pretends to be a numpy array full of zeroes."""

    def __init__(self, *, shape: tuple, dtype):
        self.shape = shape
        self.dtype = dtype
        self.val = np.zeros(shape=1, dtype=dtype)[0]

    def __len__(self) -> int:
        return self.shape[0]

    def __rsub__(self, other) -> ConstArray:
        if isinstance(other, int):
            return ConstArray(shape=self.shape, val=other, dtype=self.dtype)
        return NotImplemented

    def __getitem__(self, item):
        if isinstance(item, int):
            if len(self.shape) == 1:
                return self.val
            return ZeroArray(shape=self.shape[1:], dtype=self.dtype)
        raise NotImplementedError(f'{item=}')


class ConstArray:
    """Lightly pretends to be a numpy array full of zeroes."""

    def __init__(self, *, shape: tuple, val: int, dtype):
        self.shape = shape
        self.dtype = dtype
        self.val = val

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int):
            if len(self.shape) == 1:
                return self.val
            return ZeroArray(shape=self.shape[1:], dtype=self.dtype)
        raise NotImplementedError(f'{item=}')


def phase_correction_table_for_qrom_uncompute(*, table: Sequence[int], mx: int) -> np.ndarray:
    if isinstance(table, ZeroArray):
        return ZeroArray(shape=(len(table),), dtype=np.bool_)
    if isinstance(table, ConstArray):
        val = (table.val & mx).bit_count() & 1 == 1
        if val == 0:
            return ZeroArray(shape=(len(table),), dtype=np.bool_)
        return ConstArray(shape=(len(table),), val=val, dtype=np.bool_)
    return (np_bit_count(np.asarray(table) & mx) & 1).astype(np.bool_)
