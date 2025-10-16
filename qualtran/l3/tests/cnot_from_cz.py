#  Copyright 2025 Google LLC
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
from typing import TYPE_CHECKING

import qualtran as qlt
import qualtran.dtype as qdt

from qualtran.l3 import bloqify

import numpy as np


if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar


@bloqify
def cnot_from_cz(bb: 'BloqBuilder', ctrl, target):
    target = bb.H(target)
    ctrl, target = bb.CZ(ctrl, target)
    target = bb.H(target)
    return {'ctrl': ctrl, 'target': target}


def test_cnot_from_cz():
    bloq = cnot_from_cz.make(qlt.Signature.build(ctrl=1, target=1))
    should_be = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    unitary = bloq.tensor_contract()
    np.testing.assert_allclose(unitary, should_be, atol=1e-10)
