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

import numpy as np
import pytest

from qualtran.surface_code import LogicalErrorModel, QECScheme


@pytest.mark.parametrize('qec,want', [(QECScheme.make_beverland_et_al(), 3e-4)])
def test_logical_error_rate(qec: QECScheme, want: float):
    assert qec.logical_error_rate(3, 1e-3) == pytest.approx(want)


@pytest.mark.parametrize('qec,want', [[QECScheme.make_beverland_et_al(), 242]])
def test_physical_qubits(qec: QECScheme, want: int):
    assert qec.physical_qubits(11) == want


def test_invert_error_at():
    phys_err = 1e-3
    budgets = np.logspace(-1, -18)
    for budget in budgets:
        d = QECScheme.make_gidney_fowler().code_distance_from_budget(
            physical_error=phys_err, budget=budget
        )
        assert d % 2 == 1
        assert (
            QECScheme.make_gidney_fowler().logical_error_rate(
                physical_error=phys_err, code_distance=d
            )
            <= budget
        )
        if d > 3:
            assert (
                QECScheme.make_gidney_fowler().logical_error_rate(
                    physical_error=phys_err, code_distance=d - 2
                )
                > budget
            )


def test_logical_error_model():
    ler = LogicalErrorModel(qec_scheme=QECScheme.make_beverland_et_al(), physical_error=1e-3)
    assert ler(code_distance=3) == pytest.approx(3e-4)
