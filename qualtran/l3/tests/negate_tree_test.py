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
from typing import Dict, TYPE_CHECKING

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran import BloqBuilder, QVar
from qualtran.l3.tracing import bloq_compile


@bloq_compile
def bitwise_not(bb: 'BloqBuilder', x: 'QVar'):
    from qualtran.l3.tracing_bloqs import xgate

    outs = []
    for i in range(len(x)):
        out = xgate(x[i])
        outs.append(out)

    return {'x': bb.join(outs)}


@bloq_compile
def xor_k(bb: 'BloqBuilder', x: 'QVar', k: int):
    xs = x[:]
    for i, bit in enumerate(x.dtype.to_bits(k)):
        if bit == 1:
            xs[i] = ~xs[i]

    return {'x': bb.join(xs, dtype=x.dtype)}


@bloq_compile
def add_k(bb: 'BloqBuilder', x: 'QVar', k: int):
    from qualtran.bloqs.arithmetic import Add

    # load `k`
    qk = bb.allocate(dtype=x.dtype)
    qk = xor_k(bb, x=qk, k=k)

    # Add
    qk, x = Add.qcall(a=qk, b=x)

    # unload `k`
    qk = xor_k.adjoint(bb, x=qk, k=k)
    bb.free(qk)

    return {'x': x}


@bloq_compile
def negate(bb: 'BloqBuilder', x: 'QVar') -> Dict[str, 'QVar']:
    x = bitwise_not(bb, x=x)
    x = add_k(bb, x=x, k=1)
    return {'x': x}


def test_negate_tree():
    bloq = negate.make(qlt.Signature.build(x=qdt.QInt(4)))

    ret = bloq.call_classically(x=-5)
    print()
    print(bloq.debug_text())
    print(bloq)
    print(ret)
    assert ret == (5,)
