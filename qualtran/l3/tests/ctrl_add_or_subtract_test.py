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
from qualtran import BloqBuilder
from qualtran import dtype as qdt
from qualtran import Register
from qualtran.l3.tracing import bloq_compile
from qualtran.l3.tracing_bloqs import bitwise_not, c_bitwise_not


@bloq_compile
def ctrl_add_or_subtract(bb: 'BloqBuilder', ctrl, a, b, add_when_ctrl_is_on: bool = True):
    if add_when_ctrl_is_on:
        # flip the control bit
        ctrl = ~ctrl

    # subcircuit to add when ctrl=0 and subtract when ctrl=1.
    # Start: (0, a, b) or (1, a, b)
    ctrl, b = c_bitwise_not(ctrl, b)
    # -> (0, a, b) or (1, a, -1 - b)
    a, b = a + b
    # -> (0, a, b + a) or (1, a, -1 - b + a)
    ctrl, b = c_bitwise_not(ctrl, b)
    # -> (0, a, b + a) or (1, a, b - a)

    if add_when_ctrl_is_on:
        ctrl = ~ctrl

    return {'ctrl': ctrl, 'a': a, 'b': b}


@bloq_compile
def preconfigure_add_or_subtract(bb: 'BloqBuilder', ctrl_val: int, a, b):
    ctrl = bb.alloc_qbit(ctrl_val)
    ctrl, a, b = ctrl_add_or_subtract(bb, ctrl=ctrl, a=a, b=b)
    bb.free_qubit(ctrl, ctrl_val)
    return {'a': a, 'b': b}


def test_ctrl_add_or_subtract():
    bloq = preconfigure_add_or_subtract.make(
        ctrl_val=1, a=Register('a', qdt.QInt(8)), b=Register('b', qdt.QInt(8))
    )
    a_out, b_out = bloq.call_classically(a=2, b=3)
    assert a_out == 2
    assert b_out == 3 + 2

    bloq = preconfigure_add_or_subtract.make(
        ctrl_val=0, a=Register('a', qdt.QInt(8)), b=Register('b', qdt.QInt(8))
    )
    a_out, b_out = bloq.call_classically(a=2, b=3)
    assert a_out == 2
    assert b_out == 3 - 2
