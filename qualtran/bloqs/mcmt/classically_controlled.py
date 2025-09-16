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
from typing import Set, Tuple, TYPE_CHECKING, Union

import attrs

from qualtran import AddControlledT, Bloq, CDType, CtrlSpec, QCDType
from qualtran._infra.classical_branching import HasClassicalBranches
from qualtran._infra.controlled import _ControlledBase
from qualtran.bloqs.basic_gates import Identity

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator


@attrs.frozen
class ClassicallyControlled(HasClassicalBranches, _ControlledBase):  # type: ignore[misc]
    """Bloq that represents a gate controlled by a classical value.

    Args:
        subbloq: Bloq representing the gate or operations to be controlled
        ctrl_spec: Control value specification.
    """

    subbloq: 'Bloq'
    ctrl_spec: 'CtrlSpec'
    active_probability: float = 0.5

    def __attrs_post_init__(self):
        for qcdtype in self.ctrl_spec.qdtypes:
            if not isinstance(qcdtype, QCDType):
                raise ValueError(f"Invalid type found in `ctrl_spec`: {qcdtype}")
            if not isinstance(qcdtype, CDType):
                raise ValueError(f"Invalid type found in `ctrl_spec`: {qcdtype}")

    @classmethod
    def make_ctrl_system(
        cls, bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
    ) -> Tuple['_ControlledBase', 'AddControlledT']:
        """A factory method for creating both the Controlled and the adder function.

        See `Bloq.get_ctrl_system`.
        """
        cb = cls(subbloq=bloq, ctrl_spec=ctrl_spec)
        return cls._make_ctrl_system(cb)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        # so here's the thing: we're making a static compute graph. Subbloq may or may
        # not be executed; but Wikipedia says that call graphs consider all possible branches.
        # TODO: make sure this works as expected with resource counting.
        return {self.subbloq: 1}

    def classical_branching_probabilities(self):
        return {
            self.subbloq: self.active_probability,
            Identity(self.subbloq.signature.n_qubits()): 1.0 - self.active_probability,
        }
