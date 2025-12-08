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

import numpy as np

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran.bloqs.arithmetic import XorK
from qualtran.l3 import bloqify

if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar, QVarT


def cxor_k_bloq(k, bitsize=8):
    return XorK(qdt.QUInt(bitsize), k=k).controlled()


OPS = np.array([cxor_k_bloq(k=100 + i) for i in range(2**3)]).reshape((2,) * 3)


@bloqify
def cselect(bb, ctrl: 'QVar', selects: 'QVarT', system: 'QVarT', address=()):
    select = selects[0]
    subselects = selects[1:]
    assert ctrl.dtype == qdt.QBit(), ctrl
    assert select.dtype == qdt.QBit(), select

    [ctrl, select], active = bb.And([ctrl, select], cv1=1, cv2=0)

    if len(subselects) > 0:
        active, subselects, system = cselect(
            bb, ctrl=active, selects=subselects, system=system, address=address + (0,)
        )
    else:
        # base case address + (0,)
        op = OPS[address + (0,)]
        active, system = bb.add(op, ctrl=active, x=system)

    # Flip `active`
    ctrl, active = bb.CNOT(ctrl, active)

    if len(subselects) > 0:
        active, subselects, system = cselect(
            bb, ctrl=active, selects=subselects, system=system, address=address + (1,)
        )
    else:
        # base case address + (1,)
        op = OPS[address + (1,)]
        active, system = bb.add(op, ctrl=active, x=system)

    [ctrl, select] = bb.UnAnd([ctrl, select], active, cv1=1, cv2=1)

    selects = [select] + subselects.tolist()
    return {'ctrl': ctrl, 'selects': selects, 'system': system}


def test_cselect():
    cs = cselect.make(qlt.Signature.build(ctrl=1, selects=qdt.QBit()[3], system=8))
