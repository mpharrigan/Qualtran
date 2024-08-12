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
"""This module has a selection of minimally-implemented modular arithmetic primitives.

These bloqs serve as the callees in the call graphs of the algorithms found
in `qualtran.bloqs.factoring` and `qualtran.bloqs.mod_arithmetic`. They are place-holders,
so we don't have undefined symbols and can still merge the high-level algorithms. These shims
will be fleshed out and moved to their final organizational location soon (written: 2024-05-06).
"""
import math
from functools import cached_property
from typing import Dict, Set

from attrs import frozen

from qualtran import Bloq, QBit, QInt, QUInt, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli, XGate
from qualtran.resource_counting import BloqCountT, CostKey, QubitCount, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class MultiCToffoli(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit(), shape=(self.n,)), Register('target', QBit())])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Toffoli(), self.n - 2)}

    def my_static_costs(self, cost_key: 'CostKey'):
        # TODO https://github.com/quantumlib/Qualtran/issues/1261
        if cost_key == QubitCount():
            return self.n + 1
        return NotImplemented


@frozen
class Xor(Bloq):
    n: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit(), shape=(self.n,)), Register('target', QBit())])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Upper bound -- depends on actual data
        return {(XGate(), self.n)}


@frozen
class CAdd(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('ctrl', QBit()), Register('x', QUInt(self.n)), Register('y', QUInt(self.n))]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # litinski 2023
        return {(Toffoli(), 2 * self.n)}


@frozen
class CAddK(Bloq):
    n: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', QUInt(self.n))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # litinski 2023
        from qualtran.bloqs.arithmetic import AddK

        return AddK(self.n, k=self.k).build_call_graph(ssa=ssa)


@frozen
class Sub(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('y', QUInt(self.n))])


@frozen
class SubK(Bloq):
    n: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QInt(self.n))])

    def on_classical_vals(self, x: int) -> Dict[str, 'ClassicalValT']:
        # N = 2**self.dtype.bitsize if unsigned else 2 ** (self.dtype.bitsize - 1)
        N = 2 ** (self.n - 1)
        # Use `fmod` to get the correct wrap-around behavior
        result = int(math.fmod(x - self.k, N))
        return {'x': result}


@frozen
class LtK(Bloq):
    n: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('out', QBit())])


@frozen
class Lt(Bloq):
    n: int
    signed: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('x', QUInt(self.n)), Register('y', QUInt(self.n)), Register('out', QBit())]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # litinski
        return {(Toffoli(), self.n)}


@frozen
class CHalf(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', QUInt(self.n))])
