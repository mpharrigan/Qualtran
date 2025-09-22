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

import attrs

import qualtran as qlt
from qualtran import Bloq, BloqBuilder, CompositeBloq, Signature


def _EqK(bb: 'BloqBuilder', n: int):
    import qualtran.bloqs.arithmetic
    import qualtran.bloqs.basic_gates
    import qualtran.bloqs.bookkeeping
    import qualtran.bloqs.mcmt

    Join = bb.cfg(qualtran.bloqs.bookkeeping.Join)
    MultiControlX = bb.cfg(qualtran.bloqs.mcmt.MultiControlX)
    Split = bb.cfg(qualtran.bloqs.bookkeeping.Split)
    Xor = bb.cfg(qualtran.bloqs.arithmetic.Xor)
    ZeroState = bb.cfg(qualtran.bloqs.basic_gates.ZeroState)

    In = bb.add_register_from_dtype

    # --------------------------------------------------------------

    x = In('x', qlt.QUInt(n))
    y = In('y', qlt.QUInt(n))

    x, y = Xor(x=x, y=y)
    ys = Split(reg=y)
    out = ZeroState()
    ys, out = MultiControlX(controls=ys, target=out)
    y = Join(qlt.QUInt(n), reg=ys)
    x, y = Xor(x=x, y=y)

    return bb.finalize(**{'x': x, 'y': y, 'out': out})


@attrs.frozen
class EqK(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return self.decompose_bloq().signature

    def decompose_bloq(self) -> 'CompositeBloq':
        bb = qlt.BloqBuilder()
        soqs = _EqK(bb, n=self.n)
        return bb.finalize(**soqs)


def main():
    bloq = EqK(8)
    print(bloq.signature)
    print(bloq.decompose_bloq().debug_text())


if __name__ == '__main__':
    main()
