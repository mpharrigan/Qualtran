from typing import List, TYPE_CHECKING

import numpy as np

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran.l3 import bloqify

from qualtran.bloqs.bookkeeping import Split2, Join2

if TYPE_CHECKING:
    from qualtran import BloqBuilder, QVar


@bloqify
def AndR(bb: 'BloqBuilder', ctrl: 'QVar', x: 'QVar', *, rfunc, bfunc):
    take1 = Split2(1, x.dtype.num_bits - 1)
    put1 = take1.adjoint()

    x0, xrest = bb.add(take1, x=x)
    (ctrl, x0), c1 = bb.And([ctrl, x0])

    if xrest.dtype.num_bits > 0:
        c1, xrest, out = rfunc(bb, c1, xrest)
    else:
        c1, out = bfunc(bb, c1)

    [ctrl, x0] = bb.UnAnd([ctrl, x0], c1)
    x = bb.add(put1, y1=x0, y2=xrest)
    return {
        'ctrl': ctrl,
        'x': x,
        'out': out
    }
