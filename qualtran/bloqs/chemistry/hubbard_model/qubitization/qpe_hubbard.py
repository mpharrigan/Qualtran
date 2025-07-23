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
from typing import Dict

import attrs
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT


@attrs.frozen
class PhaseEstimateHubbard(Bloq):
    x_dim: int
    y_dim: int
    t: float
    u: float
    pe_bits: int

    @cached_property
    def signature(self) -> 'Signature':
        return self.qpe.signature

    @cached_property
    def qpe(self):
        from qualtran.bloqs.chemistry.hubbard_model.qubitization import WalkHubbard
        from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

        walk = WalkHubbard(x_dim=self.x_dim, y_dim=self.y_dim, t=self.t, u=self.u)

        # algo_eps = self.t / 100
        # N = self.x_dim * self.y_dim * 2
        # qlambda = 2 * N * self.t + (N * self.u) // 2
        # qpe_eps = algo_eps / (qlambda * np.sqrt(2))

        return QubitizationQPE(walk, LPResourceState(self.pe_bits))

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs = bb.add_from(self.qpe, **soqs)
        return {reg.name: soq for reg, soq in zip(self.signature.rights(), soqs)}
