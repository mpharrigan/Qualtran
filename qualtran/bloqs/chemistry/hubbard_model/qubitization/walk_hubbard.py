#  Copyright 2023 Google LLC
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
from typing import Dict, Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.block_encoding import SelectBlockEncoding
from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import PrepareHubbard
from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import SelectHubbard
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
    QubitizationWalkOperatorBase,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolic


def get_walk_operator_for_hubbard_model(
    x_dim: int, y_dim: int, t: float, u: float
) -> 'QubitizationWalkOperator':
    select = SelectHubbard(x_dim, y_dim)
    prepare = PrepareHubbard(x_dim, y_dim, t, u)

    return QubitizationWalkOperator(SelectBlockEncoding(select=select, prepare=prepare))


@frozen
class WalkHubbard(QubitizationWalkOperatorBase):
    """Qubitization walk operator for the Hubbard model.

    Args:
        x_dim: The number of sites in the x-dimension.
        y_dim: The number of sites in the y-dimension.
        t: The hopping parameter.
        u: The on-site potential.

    Registers:
        system: The system register.
    """

    x_dim: int
    y_dim: int
    t: float
    u: float

    @cached_property
    def signature(self) -> 'Signature':
        return self.walk_operator.signature

    @cached_property
    def select(self) -> SelectHubbard:
        return SelectHubbard(self.x_dim, self.y_dim)

    @cached_property
    def prepare(self) -> PrepareHubbard:
        return PrepareHubbard(self.x_dim, self.y_dim, self.t, self.u)

    @cached_property
    def block_encoding(self):
        return SelectBlockEncoding(select=self.select, prepare=self.prepare)

    @cached_property
    def walk_operator(self) -> QubitizationWalkOperator:
        return QubitizationWalkOperator(
            SelectBlockEncoding(select=self.select, prepare=self.prepare)
        )

    # def build_composite_bloq(
    #     self, bb: 'BloqBuilder', **soqs: 'SoquetT'
    # ) -> Dict[str, 'SoquetT']:
    #     soqs =  bb.add_from(self.walk_operator, **soqs)
    #     return {reg.name: soq for reg, soq in zip(self.signature.rights(), soqs)}
    #
    # def build_call_graph(self, ssa: 'SympySymbolic') -> Set['BloqCountT']:
    #     return {
    #         (self.select, 1),
    #         (self.prepare, 1),
    #     }
