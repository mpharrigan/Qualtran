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
from typing import Dict, Iterator, Optional, Set, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    CUInt,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    GateWithRegisters,
    QAny,
    QDType,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Power
from qualtran.bloqs.bookkeeping import AutoPartition
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

    def __attrs_post_init__(self):
        if not self.subbloq.signature.thru_registers_only:
            raise ValueError(f"ApplyLTimes can only be applied to thru bloqs, not {self.subbloq}")

    @property
    def system_bitsize(self) -> int:
        # TODO
        return self.subbloq.signature.n_qubits()

    def build_composite_bloq(
        self, bb: 'BloqBuilder', l: 'Soquet', system: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.m):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")

        if self.subbloq.signature == Signature([Register('system', QAny(self.system_bitsize))]):
            wrapped_subbloq = self.subbloq
        else:
            wrapped_subbloq = AutoPartition(
                self.subbloq,
                [
                    (
                        Register('system', QAny(self.subbloq.signature.n_qubits())),
                        [reg.name for reg in self.subbloq.signature],
                    )
                ],
            )

        lbits = bb.split(l)
        for i in range(self.m - 1, 0 - 1, -1):
            # pow_bloq = Power(wrapped_subbloq, power=2**(self.m - i - 1))
            pow_bloq = wrapped_subbloq ** (2 ** (self.m - i - 1))
            _, add_ctrled = pow_bloq.get_ctrl_system(CtrlSpec())
            (lbits[i],), (system,) = add_ctrled(
                bb=bb, ctrl_soqs=(lbits[i],), in_soqs={'system': system}
            )

        return {'l': bb.join(lbits), 'system': system}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        # Approximation: assume fast-forwardability
        return {self.subbloq: self.m}

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

    def __str__(self):
        return f'ApplyLTimes({self.subbloq})'
