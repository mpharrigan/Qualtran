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
from typing import Dict, List, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    QCDType,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Cast(_BookkeepingBloq):
    """Cast a register from one n-bit QCDType to another QCDType.

    This simply re-interprets the register's data, and is a bookkeeping operation.

    Args:
        inp_dtype: Input QCDType to cast from.
        out_dtype: Output QCDType to cast to.
        shape: Optional multidimensional shape of the register to cast.

    Registers:
        reg: The quantum variable to cast
    """

    inp_dtype: QCDType
    out_dtype: QCDType
    shape: Tuple[int, ...] = attrs.field(
        default=tuple(), converter=lambda v: (v,) if isinstance(v, int) else tuple(v)
    )

    def __attrs_post_init__(self):
        if is_symbolic(self.inp_dtype.num_bits):
            return

        if self.inp_dtype.num_bits != self.out_dtype.num_bits:
            raise ValueError(
                f"Casting must preserve the number of bits in the data. Cannot cast {self.inp_dtype} to {self.out_dtype}: {self.inp_dtype.num_bits} != {self.out_dtype.num_bits}."
            )

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f'{self} is atomic')

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('reg', dtype=self.inp_dtype, shape=self.shape, side=Side.LEFT),
                Register('reg', dtype=self.out_dtype, shape=self.shape, side=Side.RIGHT),
            ]
        )

    def adjoint(self) -> 'Bloq':
        return Cast(inp_dtype=self.out_dtype, out_dtype=self.inp_dtype)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=np.eye(2), inds=[(outgoing['reg'], j), (incoming['reg'], j)], tags=[str(self)]
            )
            for j in range(self.out_dtype.num_bits)
        ]

    def on_classical_vals(self, reg: int) -> Dict[str, 'ClassicalValT']:
        res = self.out_dtype.from_bits(self.inp_dtype.to_bits(reg))
        return {'reg': res}

    def as_cirq_op(self, qubit_manager, reg: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'reg': reg}


@bloq_example
def _cast() -> Cast:
    from qualtran import QFxp, QInt

    cast = Cast(QInt(32), QFxp(32, 32))
    return cast


_CAST_DOC = BloqDocSpec(bloq_cls=Cast, examples=[_cast], call_graph_example=None)
