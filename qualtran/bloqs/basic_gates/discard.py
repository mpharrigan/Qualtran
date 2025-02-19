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
from typing import Dict, List, TYPE_CHECKING, Union

from attrs import frozen

from qualtran import Bloq, BloqBuilder, CBit, ConnectionT, QBit, Register, Side, Signature
from qualtran.simulation.classical_sim import ClassicalValT

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.simulation.tensor import DiscardInd


@frozen
class Discard(Bloq):

    allow_qubits: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        if self.allow_qubits:
            return Signature([Register('x', QCBit(), side=Side.LEFT)])
        else:
            return Signature([Register('x', CBit(), side=Side.LEFT)])

    def on_classical_vals(self, x: int) -> Dict[str, 'ClassicalValT']:
        return {}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['DiscardInd']:
        import quimb.tensor as qtn

        from qualtran.simulation.tensor import DiscardInd

        return [DiscardInd((incoming['x'], 0))]


@frozen
class DiscardQ(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QBit(), side=Side.LEFT)])

    def on_classical_vals(self, x: int) -> Dict[str, 'ClassicalValT']:
        return {}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        from qualtran.simulation.tensor._quimb import DiscardInd  # TODO

        return [DiscardInd((incoming['x'], 0))]
