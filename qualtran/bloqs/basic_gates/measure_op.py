#  Copyright 2024 Google LLC
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
from typing import Dict

from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    CBit,
    CtrlSpec,
    QBit,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)


@frozen
class MeasureSingleQOp(Bloq):
    """Measure a Hermitian single qubit operator.

    This bloq's decomposition shows how "measurement" can be constructed from basic gates
    and the ability to discard qubits. This bloq's decomposition will measure the
    system (the "q" register) into a new register (the "c" register) which is marked as a
    classical bit on output.

    To recover the "traditional" measurement signature that takes one quantum bit in and returns
    one classical bit, you can discard the output "q" register from this bloq.

    Args:
        op: The bloq encoding the operator to measure. This must take a single qubit input
            named `q`. We use a controlled versinoo of it.

    References:
        Quantum Computing and Quantum Information.
        Nielsen and Chuang. Exercise 4.34
    """

    op: Bloq

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit()), Register('c', CBit(), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: 'BloqBuilder', q: Soquet) -> Dict[str, 'SoquetT']:
        from qualtran.bloqs.basic_gates import Hadamard, ZeroState
        from qualtran.bloqs.bookkeeping import Cast

        meas_space = bb.add(ZeroState())
        meas_space = bb.add(Hadamard(), q=meas_space)

        _, add_ctrled = self.op.get_ctrl_system(CtrlSpec())
        (meas_space,), (q,) = add_ctrled(bb, ctrl_soqs=[meas_space], in_soqs={'q': q})

        meas_space = bb.add(Hadamard(), q=meas_space)
        meas_result = bb.add(Cast(QBit(), CBit()), reg=meas_space)

        return {'c': meas_result, 'q': q}
