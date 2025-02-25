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
import abc

import attrs

from qualtran import Bloq, CDType, CtrlSpec, QCDType
from qualtran._infra.controlled import _ControlledBase  # TODO


class HasClassicalBranches(metaclass=abc.ABCMeta):
    """This mixin lets you annotate classical branches.

    Classical branches are fundamentally different than quantum branching in resource
    estimation. Quantum "branching" (via controlled or select operations) costs at least
    as much as executing each possible branch.
    """

    @abc.abstractmethod
    def classical_branching_probabilities(self): ...


class ClassicalApplyLthBloq(HasClassicalBranches): ...
