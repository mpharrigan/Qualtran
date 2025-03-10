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
from typing import Dict, Iterator, Optional, Tuple, TYPE_CHECKING

import attrs
import cirq
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CUInt,
    GateWithRegisters,
    QAny,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.phase_estimation.qpe_window_state import QPEWindowStateBase
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class ApplyLTimes(Bloq):
    subbloq: 'Bloq'
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('l', QUInt(self.m)), Register('system', QAny(self.system_bitsize))]
        )

    @property
    def system_bitsize(self) -> int:
        # TODO
        return self.subbloq.signature.n_qubits()

    def build_composite_bloq(
        self, bb: 'BloqBuilder', l: 'Soquet', system: 'SoquetT'
    ) -> Dict[str, 'SoquetT']: ...

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'l':
            return TextBox("$l$ times")
        if reg.name == 'system':
            # TODO: specific signature
            return TextBox(f"{self.subbloq}$^l$")

        return self.subbloq.wire_symbol(reg=reg, idx=idx)
