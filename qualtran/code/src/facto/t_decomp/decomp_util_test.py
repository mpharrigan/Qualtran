import math

import numpy as np

from .decomp_util import (
    gate_seq_to_matrix,
    u2_to_quaternion,
    u2_to_zero_state,
)


def test_gate_seq_to_matrix():
    np.testing.assert_allclose(gate_seq_to_matrix("S"), gate_seq_to_matrix("TT"))
    np.testing.assert_allclose(gate_seq_to_matrix(["T_X_DAG"]), gate_seq_to_matrix("HSTSSH"))
    a = gate_seq_to_matrix(["H", "S"])
    b = gate_seq_to_matrix(["C_ZYX"])
    np.testing.assert_allclose(a / a[0, 0], b / b[0, 0])


def test_u2_to_zero_state():
    s = 0.5**0.5
    sj = s * 1j
    np.testing.assert_allclose(u2_to_zero_state(np.array([[s, s], [s, -s]])), [1, 0, 0])
    np.testing.assert_allclose(u2_to_zero_state(np.array([[s, s], [-s, s]])), [-1, 0, 0])
    np.testing.assert_allclose(u2_to_zero_state(np.array([[s, s], [sj, s]])), [0, 1, 0])
    np.testing.assert_allclose(u2_to_zero_state(np.array([[s, s], [-sj, s]])), [0, -1, 0])
    np.testing.assert_allclose(u2_to_zero_state(np.array([[1, 0], [0, 1]])), [0, 0, 1])
    np.testing.assert_allclose(u2_to_zero_state(np.array([[0, 1], [1, 0]])), [0, 0, -1])
    np.testing.assert_allclose(u2_to_zero_state(np.array([[0, 1], [1j, 0]])), [0, 0, -1])


def test_u2_to_quaternion():
    s = 0.5**0.5
    np.testing.assert_allclose(u2_to_quaternion(gate_seq_to_matrix("S")), [s, 0, 0, s])
    np.testing.assert_allclose(
        u2_to_quaternion(gate_seq_to_matrix("T")),
        [math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)],
    )
    np.testing.assert_allclose(
        u2_to_quaternion(gate_seq_to_matrix(["T_X"])),
        [math.cos(math.pi / 8), math.sin(math.pi / 8), 0, 0],
    )
    np.testing.assert_allclose(u2_to_quaternion(gate_seq_to_matrix(["SQRT_X_DAG"])), [s, -s, 0, 0])
    np.testing.assert_allclose(u2_to_quaternion(gate_seq_to_matrix(["SQRT_Y_DAG"])), [s, 0, -s, 0])
    np.testing.assert_allclose(
        u2_to_quaternion(gate_seq_to_matrix(["C_XYZ"])), [0.5, 0.5, 0.5, 0.5]
    )
    np.testing.assert_allclose(
        u2_to_quaternion(gate_seq_to_matrix(["C_ZYX"])), [0.5, -0.5, -0.5, -0.5]
    )
