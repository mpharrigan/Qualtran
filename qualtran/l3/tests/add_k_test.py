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

import attrs

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran import BloqBuilder
from qualtran._infra.composite_bloq import QVar, QVarT
from qualtran.bloqs.arithmetic import Add
from qualtran.l3.tracing import bloq_compile
from qualtran.l3.tracing_bloqs import xgate
from qualtran.symbolics import is_symbolic


@bloq_compile
def add_k_complex(bb: 'BloqBuilder', x: 'QVar', k: int):
    dtype = x.dtype
    if not is_symbolic(k) and k < 0 and isinstance(dtype, (qdt.QUInt, qdt.QMontgomeryUInt)):
        # Simplification for negative k for unsigned integers:
        # Since this is unsigned addition, adding `-v` is equivalent to adding `2**bitsize - v`
        k %= 2**dtype.bitsize

    # load `k`
    qk = bb.allocate(dtype=dtype)
    qk = xor_k(bb, x=qk, k=k)

    # perform the quantum-quantum addition
    # we always perform this addition (even when controlled), so we wrap in `Always`
    # controlling the data loading is sufficient to control this bloq.
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Always

    qk, x = bb.add(Always(Add(dtype, dtype)), a=qk, b=x)

    # unload `k`
    qk = xor_k.adjoint(bb, x=qk, k=k)
    bb.free(qk)

    return {'x': x}
