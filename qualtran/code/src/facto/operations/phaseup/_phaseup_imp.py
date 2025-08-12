from __future__ import annotations

import collections
from typing import Sequence

import numpy as np

from scatter_script import CostKey, QPU, quint


def table2power_conversion_matrix(n: int) -> np.ndarray:
    """Returns a matrix for converting offset-indexed data into power-product-indexed data.

    Offset-indexed data is accumulated by comparing the address exactly to the iterator:

        output[address] == xorsum(
            input[M]
            for M in range(len(OFFSET_INDEXED_TABLE))
            if M & address == M
        )

    Power-product indexed data accumulates outputs by masking the address with the iterator:

        output[address] = xorsum(
            input[M]
            for M in range(len(POWER_PRODUCT_INDEXED_TABLE))
            if M & address == M
        )

    The benefit of the latter approach is that, when the address is quantum and the iterator is
    classical, it does O(N) superposed AND gates instead of O(N lg N) superposed AND gates to
    execute the xorsum.

    This method returns a matrix M such that

        POWER_PRODUCT_INDEXED_TABLE = M @ OFFSET_INDEXED_TABLE

    meaning it converts from normal data into the format required for power product indexing.

    Example:
        >>> m = table2power_conversion_matrix(5)
        >>> # surprise Sierpinsky triangle!
        >>> print(str(m).replace('0', '.').replace(' ', '').replace('[', '').replace(']', ''))
        1...............................
        11..............................
        1.1.............................
        1111............................
        1...1...........................
        11..11..........................
        1.1.1.1.........................
        11111111........................
        1.......1.......................
        11......11......................
        1.1.....1.1.....................
        1111....1111....................
        1...1...1...1...................
        11..11..11..11..................
        1.1.1.1.1.1.1.1.................
        1111111111111111................
        1...............1...............
        11..............11..............
        1.1.............1.1.............
        1111............1111............
        1...1...........1...1...........
        11..11..........11..11..........
        1.1.1.1.........1.1.1.1.........
        11111111........11111111........
        1.......1.......1.......1.......
        11......11......11......11......
        1.1.....1.1.....1.1.....1.1.....
        1111....1111....1111....1111....
        1...1...1...1...1...1...1...1...
        11..11..11..11..11..11..11..11..
        1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
        11111111111111111111111111111111
    """
    m = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    result = np.array([[1]], dtype=np.uint8)
    for k in range(n):
        result = np.kron(result, m)
    return result


def do_wandering_power_product(
    qpu: QPU, addr: quint, *, disable_wandering: bool = False
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, list[quint]]:
    """Produces 2**n cnot-independent values derived from a register.

    The non-wandered values would be all possible products of subsets of qubits in
    the register. Due to using wandering AND gates, the result will instead be a
    linear combination of those values (with the returned matrices indicating how
    to convert between the returned representation and the non-wandered
    representation).

    Args:
        qpu: The quantum simulator to allocate qubits from.
        addr: The address register to expand into a power product.
        disable_wandering: Defaults to False. When set to True, wandering is disabled.
            This results in the wander comparison results all being 0, the conversion
            matrices being the identity, and the returned qubits being exactly the
            power product. Used for debugging.

    Returns:
        A tuple (cmp, mat, inv_mat, values).
            cmp: list[tuple[int, int]]: The comparison results from the teleportations.
            mat: Conversion matrix to transform corrected results into these uncorrected results.
            inv_mat: Conversion matrix for CNOTs that would correct the results.
            values: The uncorrected power product qubits.

        where result[k] == prod(addr[j] for j in range(len(addr)) if k & (1 << j))
        where mat @ NORMAL_RESULT == values
        where inv_mat @ values == NORMAL_RESULT

        Note that result[0] is just the integer 1.
        Note that result[1 << k] is addr[k] for each k (not just equal; the literal same qubit).
    """
    correction_matrix = np.zeros(shape=(1 << len(addr), 1 << len(addr)), dtype=np.uint8)
    correction_matrix[0, 0] = 1
    result = [1]
    cmp = []
    for col in range(1, 1 << len(addr)):
        control_bit = col.bit_length() - 1
        prev = col & ~(1 << control_bit)
        if prev == 0:
            correction_matrix[col, col] = 1
            result.append(addr[control_bit])
        else:
            c1 = prev
            c2 = 1 << control_bit
            cmp1, cmp2, Q_and = qpu.alloc_wandering_and(
                result[c1], result[c2], disable_wandering=disable_wandering
            )
            cmp.append((cmp1, cmp2))
            result.append(Q_and)
            for r2 in np.flatnonzero(correction_matrix[:, c2]):
                for r1 in np.flatnonzero(correction_matrix[:, c1]):
                    correction_matrix[r1 | r2, col] ^= 1
            if cmp1:
                correction_matrix[:, col] ^= correction_matrix[:, c2]
            if cmp2:
                correction_matrix[:, col] ^= correction_matrix[:, c1]
            if cmp1 and cmp2:
                correction_matrix[0, col] ^= 1

    inverse_correction_matrix = matrix_inverse_mod2(correction_matrix)
    return cmp, correction_matrix, inverse_correction_matrix, result


def matrix_inverse_mod2(mat: np.ndarray) -> np.ndarray:
    """Computes the inverse of a matrix, working modulo 2."""
    mat = np.copy(mat).astype(np.uint8)
    mat &= 1
    cnots = []
    for col in range(mat.shape[1]):
        hits = np.flatnonzero(mat[:, col])
        for h in hits:
            if h >= col:
                pivot = h
                break
        else:
            raise ValueError("Not invertible.")
        for h in hits:
            if h != pivot:
                cnots.append((pivot, h))
                mat[h, :] ^= mat[pivot, :]
        if pivot != col:
            cnots.append((col, pivot))
            cnots.append((pivot, col))
            cnots.append((col, pivot))
            mat[pivot, :] ^= mat[col, :]
            mat[col, :] ^= mat[pivot, :]
            mat[pivot, :] ^= mat[col, :]

    inv = np.eye(mat.shape[0], dtype=np.uint8)
    for control, target in cnots:
        inv[target] ^= inv[control]
    return inv


def undo_wandering_power_product(
    qpu: QPU, result: list[quint], *, cmp_data_from_computation: list[tuple[int, int]]
) -> None:
    result = list(result)
    while result:
        h = len(result) >> 1
        zs = np.zeros(shape=h + 1, dtype=np.uint8)
        czs = np.zeros(shape=h, dtype=np.uint8)
        for k in range(1, h)[::-1]:
            p1, p2 = cmp_data_from_computation.pop()
            if result[h + k].del_measure_x():
                czs[k] ^= 1
                zs[k] ^= p2
                zs[h] ^= p1
                zs[0] ^= p1 & p2
        qpu.cz_multi_target(result[h], result[:h], czs)
        qpu.cz_multi_target(1, result[: h + 1], zs)
        result = result[:h]


def estimate_cost_of_sqrt_phaseup(addr_len: int) -> collections.Counter[CostKey]:
    n = addr_len
    h = n >> 1
    return collections.Counter(
        {
            CostKey("CZ_multi_target"): max(0, (1 << h) + n - 3),
            CostKey("uncorrected_and"): ((2 + n % 2) << h) - n - 2,
        }
    )


def do_sqrt_phaseup(qpu: QPU, addr: quint, table: Sequence[bool]):
    """Performs a phase lookup that costs at most 2*sqrt(len(table)) Toffoli gates.

    Let n = len(addr)
    Let N = len(table) = 1 << n
    let S = 2**(n//2)
    The AND gate cost of this method is S*(2 + n%2) - n - 2 <= 2*sqrt(N).
    The multi-target CZ count is S - 1 <= sqrt(N).

    Splits the address register into a low half and a high half.

    Performs control unfolding on each half, producing every possible
    combination of intersections within each half.

    Uses multi-target CZ gates between the two halves to resolve all
    phases that cross the boundary.

    Then resolves the within-half phases while uncomputing the unfolding.
    """
    assert len(table) == 1 << len(addr)
    n1 = len(addr) >> 1
    n2 = len(addr) - n1
    h1 = addr[:n1]
    h2 = addr[n1:]
    m1 = 1 << n1
    m2 = 1 << n2

    # CZ coupling matrix is driven by the data table.
    cz_matrix = np.array(table, dtype=np.uint8).reshape(m2, m1)

    # Convert from offset-indexed data to power-product-indexed data.
    conv1 = table2power_conversion_matrix(n1)
    conv2 = table2power_conversion_matrix(n2)
    cz_matrix = conv2 @ cz_matrix @ np.transpose(conv1)

    # Expand into power products, accounting for AND gate corrections via the CZ couplings.
    cmp1, _, correction_matrix1, Q_p1s = do_wandering_power_product(qpu, h1)
    cmp2, _, correction_matrix2, Q_p2s = do_wandering_power_product(qpu, h2)
    cz_matrix = correction_matrix2 @ cz_matrix @ np.transpose(correction_matrix1)

    # Apply Z and CZ gates encoding the table data.
    cz_matrix &= 1
    for k1 in range(len(Q_p1s)):
        qpu.cz_multi_target(Q_p1s[k1], Q_p2s, cz_matrix[:, k1])

    undo_wandering_power_product(qpu, Q_p2s, cmp_data_from_computation=cmp2)
    undo_wandering_power_product(qpu, Q_p1s, cmp_data_from_computation=cmp1)
