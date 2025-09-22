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

from functools import cached_property

import attrs

import qualtran.l3 as qlt
from qualtran import Bloq, BloqBuilder, CompositeBloq, Signature


def _negate(bb: 'BloqBuilder', n: int):
    In = bb.add_register_from_dtype

    x = qlt.In('x', qlt.QInt(n))
    x = ~x
    x += 1
    return {'x': x}


@attrs.frozen
class negate(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return self.decompose_bloq().signature

    def decompose_bloq(self) -> 'CompositeBloq':
        bb = BloqBuilder()
        soqs = _negate(bb, n=self.n)
        return bb.finalize(**soqs)
