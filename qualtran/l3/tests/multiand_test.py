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

from typing import List, TYPE_CHECKING

import numpy as np

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran.l3 import bloqify

if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar


@bloqify
def multiand(bb, ctrls: List['QVar']):
    n = len(ctrls)
    ancs = [ctrls[0]]

    # Do a ladder of AND storing intermediate bits in `ancs`
    for i in range(1, n):
        (ancs[i - 1], ctrls[i]), anc = bb.And([ancs[i - 1], ctrls[i]])
        ancs.append(anc)

    # Copy the output to a new wire.
    out = bb.alloc_qbit(0)
    ancs[-1], out = bb.CNOT(ancs[-1], out)

    # Do a ladder of Uncompute-AND, cleaning up `ancs` array.
    for i in range(n - 1, 1 - 1, -1):
        (ancs[i - 1], ctrls[i]) = bb.UnAnd([ancs[i - 1], ctrls[i]], ancs[i])

    ctrls[0] = ancs[0]
    return {'ctrls': ctrls, 'out': out}


def test_multiand():
    bloq = multiand.make(qlt.Signature.build(ctrls=qdt.QBit()[4]))
    ctrl, out = bloq.call_classically(ctrls=[1, 1, 1, 0])
    assert ctrl.tolist() == [1, 1, 1, 0]
    assert out == 0

    ctrl, out = bloq.call_classically(ctrls=[1, 1, 1, 1])
    assert ctrl.tolist() == [1, 1, 1, 1]
    assert out == 1


def test_multiand_tensor():
    n = 4
    bloq = multiand.make(qlt.Signature.build(ctrls=qdt.QBit()[n]))
    active = np.array(np.where(bloq.tensor_contract().reshape((2**n, 2, 2**n)))).T
    for out_number, out_bit, in_number in active:
        assert in_number == out_number
        assert out_bit == (in_number == int('1' * n, base=2))
