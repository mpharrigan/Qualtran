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
from typing import Dict, Optional, Tuple

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    CBit,
    CUInt,
    DecomposeTypeError,
    QAny,
    QBit,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import MeasZ
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.qft import QFTTextBook
from qualtran.drawing import directional_text_box, RarrowTextBox, Text, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics.types import SymbolicInt


@frozen
class MeasQFT(Bloq):
    n: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QUInt(self.n), side=Side.LEFT),
                Register('x', CUInt(self.n), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> Dict[str, 'SoquetT']:
        x = bb.add(QFTTextBook(self.n).adjoint(), q=x)
        xs = bb.split(x)
        for i in range(self.n):
            xs[i] = bb.add(MeasZ(), q=xs[i])
            xs[i] = bb.add(Cast(CBit(), QBit(), allow_quantum_to_classical=True), reg=xs[i])

        # TODO: need classical join
        x = bb.join(xs, QUInt(self.n))
        x = bb.add(Cast(QUInt(self.n), CUInt(self.n), allow_quantum_to_classical=True), reg=x)
        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {QFTTextBook(self.n): 1, MeasZ(): self.n}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'x':
            return directional_text_box('MeasQFT', reg.side)
        raise ValueError(f'Unrecognized register name {reg.name}')
