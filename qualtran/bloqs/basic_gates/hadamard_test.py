#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import itertools
from typing import Dict

import cirq
import numpy as np
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QBit, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import Hadamard, IntEffect, IntState, OnEach, OneEffect, OneState
from qualtran.bloqs.basic_gates.hadamard import _hadamard, CHadamard
from qualtran.cirq_interop import cirq_gate_to_bloq
from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(OneState())
    q = bb.add(Hadamard(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───X───H───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_hadamard(bloq_autotester):
    bloq_autotester(_hadamard)


def test_unitary_vs_cirq():
    h = Hadamard()
    unitary = h.tensor_contract()
    cirq_unitary = cirq.unitary(cirq.H)
    np.testing.assert_allclose(unitary, cirq_unitary)


def test_not_classical():
    h = Hadamard()
    with pytest.raises(NotImplementedError, match=r'.*is not classically simulable\.'):
        h.call_classically(q=0)


def test_chadamard_vs_cirq():
    bloq = Hadamard().controlled()
    assert bloq == CHadamard()

    gate = cirq.H.controlled()
    np.testing.assert_allclose(cirq.unitary(gate), bloq.tensor_contract())


def test_cirq_interop():
    circuit = CHadamard().as_composite_bloq().to_cirq_circuit()
    should_be = cirq.Circuit(
        [cirq.Moment(cirq.H(cirq.NamedQubit('target')).controlled_by(cirq.NamedQubit('ctrl')))]
    )
    assert circuit == should_be

    (op,) = list(should_be.all_operations())
    assert op.gate is not None
    assert cirq_gate_to_bloq(op.gate) == CHadamard()


def test_pl_interop():
    import pennylane as qml

    bloq = Hadamard()
    pl_op_from_bloq = bloq.as_pl_op(wires=[0])
    pl_op = qml.Hadamard(wires=[0])
    assert pl_op_from_bloq == pl_op

    matrix = pl_op.matrix()
    should_be = bloq.tensor_contract()
    np.testing.assert_allclose(should_be, matrix)


def test_active_chadamard_is_hadamard():
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    ctrl_on = bb.add(OneState())
    ctrl_on, q = bb.add(CHadamard(), ctrl=ctrl_on, target=q)
    bb.add(OneEffect(), q=ctrl_on)
    cbloq = bb.finalize(q=q)

    np.testing.assert_allclose(Hadamard().tensor_contract(), cbloq.tensor_contract())


def test_chadamard_adjoint():
    bb = BloqBuilder()
    ctrl = bb.add_register('ctrl', 1)
    q = bb.add_register('q', 1)
    ctrl, q = bb.add(CHadamard(), ctrl=ctrl, target=q)
    ctrl, q = bb.add(CHadamard().adjoint(), ctrl=ctrl, target=q)
    cbloq = bb.finalize(ctrl=ctrl, q=q)

    np.testing.assert_allclose(np.eye(4), cbloq.tensor_contract(), atol=1e-12)


@frozen
class HadamardTransform(Bloq):
    n: int

    @property
    def signature(self) -> 'Signature':
        return Signature([Register('qubits', QBit(), shape=(self.n,))])

    def build_composite_bloq(self, bb: 'BloqBuilder', qubits: 'SoquetT') -> Dict[str, 'SoquetT']:
        for i in range(self.n):
            qubits[i] = bb.add(Hadamard(), q=qubits[i])

        return {'qubits': qubits}


def test_hadamard_transform():
    # H^\otimes n = 2^-n/2 \sum_{x, y} (-1)^{x dot y} |y><x|
    n = 100
    x_fix = [0] * n
    x_fix[1] = None
    x_fix[2] = None
    x_fix[9] = None
    n_x_free = sum(1 for x in x_fix if x is None)
    y_fix = [0] * n
    y_fix[9] = None
    y_fix[10] = None
    n_y_free = sum(1 for y in y_fix if y is None)
    bloq = HadamardTransform(n=n)
    bb = BloqBuilder()
    free_qubits = bb.add_register(
        Register('free_qubits', QBit(), shape=(n_x_free,), side=Side.LEFT)
    )
    bb.add_register(Register('free_qubits', QBit(), shape=(n_y_free,), side=Side.RIGHT))
    free_i = 0
    soqs = np.empty(n, dtype=object)
    for j, val in enumerate(x_fix):
        if val is None:
            soqs[j] = free_qubits[free_i]
            free_i += 1
        else:
            soqs[j] = bb.add(IntState(val, 1))

    qubits = bb.add(bloq, qubits=soqs)

    soqs = np.empty(n_y_free, dtype=object)
    free_i = 0
    for j, val in enumerate(y_fix):
        if val is None:
            soqs[free_i] = qubits[j]
            free_i += 1
        else:
            bb.add(IntEffect(val, 1), val=qubits[j])

    cbloq = bb.finalize(free_qubits=soqs)
    v = cbloq.tensor_contract()
    assert v.shape == (2**n_y_free, 2**n_x_free)

    u = np.zeros((2,) * (n_y_free) + (2,) * (n_x_free))
    for partial_x in itertools.product([0, 1], repeat=n_x_free):
        for partial_y in itertools.product([0, 1], repeat=n_y_free):
            px = iter(partial_x)
            x = [xf if xf is not None else next(px) for xf in x_fix]
            py = iter(partial_y)
            y = [yf if yf is not None else next(py) for yf in y_fix]
            u[*partial_y, *partial_x] = (-1) ** (np.asarray(x) @ np.asarray(y))
    u *= 2 ** (-n / 2)
    u = u.reshape((2**n_y_free, 2**n_x_free))
    np.testing.assert_allclose(u, v)
