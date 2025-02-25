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
from typing import Set, Tuple, Union

import attrs

from qualtran import AddControlledT, Bloq, CDType, CtrlSpec, QCDType
from qualtran._infra.classical_branching import HasClassicalBranches
from qualtran._infra.controlled import _ControlledBase
from qualtran.bloqs.basic_gates import Identity


@attrs.frozen
class ClassicallyControlled(HasClassicalBranches, _ControlledBase, Bloq):

    subbloq: 'Bloq'
    ctrl_spec: 'CtrlSpec'
    active_probability: float = 0.5

    def __attrs_post_init__(self):
        for qdtype in self.ctrl_spec.qdtypes:
            if not isinstance(qdtype, QCDType):
                raise ValueError(f"Invalid type found in `ctrl_spec`: {qdtype}")
            if not isinstance(qdtype, CDType):
                raise ValueError("TODO")

    @classmethod
    def make_ctrl_system(
        cls, bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
    ) -> Tuple['Bloq', 'AddControlledT']:
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
