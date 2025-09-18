from __future__ import annotations

from facto.algorithm.prep import ExecutionConfig, ProblemConfig
from facto.algorithm._detailed_example_code import approx_modexp, loop4
from scatter_script import QPU, rvalue_multi_int, slice_int, CostKey


def test_approx_modexp():
    conf = ExecutionConfig.from_problem_config(
        ProblemConfig.from_ini_content(
            """
                modulus = 1146337137660851397915791125349
                num_input_qubits = 13
                num_shots = 1
                generator = 66
                window1 = 2
                window3a = 2
                window3b = 2
                window4 = 2
                min_wraparound_gap = 20
                mask_bits = 10
                len_accumulator = 20
            """
        )
    )
    qpu = QPU(num_branches=32)
    Q_exponent = qpu.alloc_quint(scatter=True, length=conf.num_input_qubits)
    g = rvalue_multi_int.from_value(conf.generator, expected_count=qpu.num_branches)
    initial_value = Q_exponent.UNPHYSICAL_copy()
    result = approx_modexp(qpu=qpu, conf=conf, Q_exponent=Q_exponent)
    assert qpu.cost_counters[CostKey("qubits")] == 104
    assert Q_exponent == initial_value

    exact = pow(g, Q_exponent, conf.modulus)
    approx = result << conf.dropped_bits
    error = exact - approx
    for err in error.UNPHYSICAL_branch_vals:
        if err * 2 > conf.modulus:
            err = conf.modulus - err
        deviation = err / conf.modulus
        assert deviation < 1e-3

    Q_exponent.UNPHYSICAL_force_del(dealloc=True)
    result.UNPHYSICAL_force_del(dealloc=True)
    qpu.verify_clean_finish()


def test_loop4_vs_simple_loop():
    conf = ExecutionConfig.from_problem_config(
        ProblemConfig.from_ini_content(
            """
        modulus = 1146337137660851397915791125349
        num_input_qubits = 11
        generator = 66
        window1 = 2
        window3a = 2
        window3b = 2
        window4 = 4
        min_wraparound_gap = 20
        len_accumulator = 20
        mask_bits = 4
        num_shots = 1
    """
        )
    )

    qpu = QPU(num_branches=1)
    for i in range(len(conf.periods)):
        Q_residue = qpu.alloc_quint(length=conf.rns_primes_bit_length, scatter=True)
        Q_result_accumulator = qpu.alloc_quint(
            length=conf.len_accumulator + 1, scatter=True, scatter_range=conf.truncated_modulus
        )

        # Compute expected value.
        (residue,) = [int(e) for e in Q_residue.UNPHYSICAL_branch_vals]
        w = conf.window4
        (expected,) = [int(e) for e in Q_result_accumulator.UNPHYSICAL_branch_vals]
        for j in range(conf.num_windows4):
            rw = slice_int(residue, slice(j * w, j * w + w))
            expected -= int(conf.table4[i, j, rw])
        expected %= conf.truncated_modulus

        # Actual compute.
        loop4(qpu=qpu, Q_residue=Q_residue, conf=conf, Q_acc=Q_result_accumulator, i=i)

        assert (expected,) == Q_result_accumulator.UNPHYSICAL_branch_vals

        Q_result_accumulator.UNPHYSICAL_force_del(dealloc=True)
        Q_residue.UNPHYSICAL_force_del(dealloc=True)
        qpu.verify_clean_finish()
