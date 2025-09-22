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
from typing import Dict, Mapping, Union

import attrs

import qualtran as qlt
from qualtran import (
    Bloq,
    BloqBuilder,
    CompositeBloq,
    QBit,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class EqK(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QUInt(self.n)),
                Register('y', QUInt(self.n)),
                Register('out', QBit(), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        from qualtran.bloqs.arithmetic import Xor
        from qualtran.bloqs.basic_gates import ZeroState
        from qualtran.bloqs.mcmt import MultiControlX

        x, y = bb.add(Xor(QUInt(self.n)), x=x, y=y)
        ys = bb.split(y)
        out = bb.add(ZeroState())
        ys, out = bb.add(MultiControlX(cvs=[0] * self.n), controls=ys, target=out)
        y = bb.join(ys, dtype=QUInt(self.n))
        x, y = bb.add(Xor(QUInt(self.n)), x=x, y=y)
        return {'x': x, 'y': y, 'out': out}

    def on_classical_vals(self, x: int, y: int) -> Mapping[str, 'ClassicalValT']:
        return {'x': x, 'y': y, 'out': int(x == y)}


def main():
    bloq = EqK(8)
    print(bloq.signature)
    print(bloq.decompose_bloq().debug_text())


if __name__ == '__main__':
    main()
