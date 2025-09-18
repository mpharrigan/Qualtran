from __future__ import annotations

import tempfile

from facto.algorithm.prep._execution_config import ExecutionConfig
from facto.algorithm.prep._problem_config import ProblemConfig


def test_compute_and_save_execution_config():
    input_content = """
        modulus = 1146337137660851397915791125349
        num_input_qubits = 11
        generator = 66
        window1 = 2
        window3a = 2
        window3b = 2
        window4 = 2
        min_wraparound_gap = 20
        len_accumulator = 40
        mask_bits = 10
        num_shots = 1
    """
    problem_config = ProblemConfig.from_ini_content(input_content)
    exec_config = ExecutionConfig.from_problem_config(problem_config)
    with tempfile.TemporaryDirectory() as d:
        exec_config.write_to_data_directory(d, input_conf=problem_config)


def test_vacuous_config():
    exec_config = ExecutionConfig.vacuous_config(
        num_shots=10,
        modulus_bitlength=8192,
        num_input_qubits=8192,
        num_periods=2048**2,
        period_bitlength=22,
        w1=6,
        w3a=3,
        w3b=3,
        w4=6,
        len_acc=32,
    )
    assert exec_config.estimated_logical_qubits == 8318
