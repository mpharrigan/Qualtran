from typing import Dict, List, TYPE_CHECKING

import attrs
import numpy as np

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran.bloqs.bookkeeping import Join2, Split2
from qualtran.l3 import bloqify
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar


@bloqify
def AndR(bb: 'BloqBuilder', ctrl: 'QVar', x: 'QVar'):
    if x.dtype.num_bits == 0:
        # Base case (can make this a templated parameter)
        out = bb.alloc_qbit()
        ctrl, out = bb.CNOT(ctrl, out)
        return {'ctrl': ctrl, 'x': x, 'out': out}

    # Set-up
    take1 = Split2(1, x.dtype.num_bits - 1)
    put1 = take1.adjoint()

    # Compute
    x0, xrest = bb.add(take1, x=x)
    (ctrl, x0), c1 = bb.And([ctrl, x0])

    # Recurse
    c1, xrest, out = AndR(bb, c1, xrest)

    # Uncompute
    [ctrl, x0] = bb.UnAnd([ctrl, x0], c1)
    x = bb.add(put1, y1=x0, y2=xrest)

    return {'ctrl': ctrl, 'x': x, 'out': out}


@attrs.frozen
class SymbolicMultiAnd(qlt.Bloq):
    n: SymbolicInt

    @property
    def signature(self) -> 'qlt.Signature':
        return qlt.Signature.build(ctrl=1, x=qdt.QAny(self.n), out=(None, qdt.QBit()))

    def build_composite_bloq(self, bb: 'BloqBuilder', ctrl, x):
        if self.n == 0:
            out = bb.alloc_qbit()
            ctrl, out = bb.CNOT(ctrl, out)
            return {'ctrl': ctrl, 'x': x, 'out': out}

        # Set-up
        take1 = Split2(1, self.n - 1)
        put1 = take1.adjoint()

        # Compute
        x0, xrest = bb.add(take1, x=x)
        (ctrl, x0), c1 = bb.And([ctrl, x0])

        # Recurse
        c1, xrest, out = bb.add(SymbolicMultiAnd(self.n - 1), ctrl=c1, x=xrest)

        # Uncompute
        [ctrl, x0] = bb.UnAnd([ctrl, x0], c1)
        x = bb.add(put1, y1=x0, y2=xrest)

        return {'ctrl': ctrl, 'x': x, 'out': out}
