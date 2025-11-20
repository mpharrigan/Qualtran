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
from qualtran.l3 import bloqify
from qualtran.l3.tracing_bloqs import add_k, bitwise_not

if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar


@bloqify
def negate(bb: 'BloqBuilder', x: 'QVar') -> Dict[str, 'QVar']:
    x = ~x
    x += 1
    return {'x': x}


def test_negate_func():
    assert negate.name == 'negate'
    assert negate.pkg == 'negate_test'  # TODO: why not the full name?


@attrs.frozen
class Negate(qlt.Bloq):
    n: int

    @property
    def signature(self) -> 'qlt.Signature':
        return qlt.Signature.build(x=qdt.QInt(self.n))

    def decompose_bloq(self) -> 'qlt.CompositeBloq':
        return negate.make(self.signature)


def test_negate_bloq():
    bloq = Negate(32)
    assert bloq.signature.n_bits() == 32
    (x,) = bloq.call_classically(x=-6)
    assert x == 6


@bloqify
def negate_program(bb: 'BloqBuilder', n: int) -> Dict[str, 'QVar']:
    x = bb.alloc_qint(k=5, bitsize=n)
    x = negate(bb, x=x)
    return {'x': x}


def test_negate_program():
    bloq = negate_program.make(qlt.Signature.build(x=(None, qdt.QInt(8))), n=8)
    assert bloq.signature.n_bits() == 8
    (x,) = bloq.call_classically()
    assert x == qdt.QUInt(8).from_bits(qdt.QInt(8).to_bits(-5))  # TODO IntState


@attrs.frozen
class NegateProgram(qlt.Bloq):
    n: int

    @property
    def signature(self) -> 'qlt.Signature':
        # It is required that the bloq author give input registers somewhere, so why not
        # where we expect them
        return qlt.Signature([qlt.Register('x', qdt.QInt(self.n), side=qlt.Side.RIGHT)])

    def decompose_bloq(self) -> 'qlt.CompositeBloq':
        # This can be automatically deduced from attrs.fields() and self.signature()
        return negate_program.make(n=self.n, **{reg.name: reg for reg in self.signature.lefts()})
