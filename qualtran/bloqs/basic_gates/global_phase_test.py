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

import cirq
import numpy as np

from qualtran.bloqs.basic_gates.global_phase import _global_phase, GlobalPhase
from qualtran.cirq_interop import cirq_gate_to_bloq
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


def test_unitary():
    random_state = np.random.RandomState(2)

    for alpha in random_state.random(size=10):
        coefficient = np.exp(2j * np.pi * alpha)
        bloq = GlobalPhase(coefficient)
        np.testing.assert_allclose(cirq.unitary(bloq), coefficient)


def test_controlled():
    random_state = np.random.RandomState(2)
    for alpha in random_state.random(size=10):
        coefficient = np.exp(2j * np.pi * alpha)
        bloq = GlobalPhase(coefficient).controlled()
        np.testing.assert_allclose(
            cirq.unitary(cirq.GlobalPhaseGate(coefficient).controlled()), bloq.tensor_contract()
        )


def test_cirq_interop():
    cbloq = GlobalPhase(1.0j).as_composite_bloq()

    circuit = cbloq.to_cirq_circuit()
    assert circuit == cirq.Circuit(cirq.GlobalPhaseGate(1.0j).on())

    assert cirq_gate_to_bloq(cirq.GlobalPhaseGate(1.0j)) == GlobalPhase(1.0j)


def test_t_complexity():
    assert GlobalPhase(1j).t_complexity() == TComplexity()


def test_global_phase(bloq_autotester):
    bloq_autotester(_global_phase)
