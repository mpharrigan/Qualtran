#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List, Dict, Set
import attrs
import sympy

import qualtran.testing as qlt_testing
from qualtran import (
    Bloq, BloqBuilder,
    QBit, QAny, QInt, QUInt,
    Register, Side, Signature,
)

from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt
from qualtran.drawing import show_bloq, show_call_graph
from qualtran.bloqs.factoring import CtrlModMul


# In[2]:


# new imports
from qualtran.simulation.classical_sim import ClassicalValT


# In[3]:


@attrs.frozen
class ModExp(Bloq):
    """Perform modular exponentiation `x = g^exponent mod p`.

    x and exponent are quantum variables; g and the modulus are classical constants.

    The exponent register has `n_exponent` bits.

    For simplicity, let's assume
        - a 32-bit "x" register.
    """

    g: int
    mod: int
    n_exponent: SymbolicInt = 4

    @property
    def signature(self) -> 'Signature':
        return Signature([
            Register('exponent', QUInt(bitsize=self.n_exponent)),
            # Change alert! "x" is now a RIGHT register.
            Register('x', QUInt(bitsize=32), side=Side.RIGHT)
        ])

    def on_classical_vals(
            self, exponent: 'ClassicalValT',
    ) -> Dict[str, 'ClassicalValT']:
        # We can override this method to show how this bloq acts on classical values,
        # (computational basis states). The method takes in arguments named according
        # to this bloq's registers and must return a dictionary of output values (keyed
        # by the bloq's register names)

        # ! ---------------------
        return {
            'exponent': exponent,
            'x': (self.g ** exponent) % self.mod
        }
        # ! ---------------------

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        k = ssa.new_symbol('k')
        return {
            (CtrlModMul(k=k, mod=self.mod, bitsize=32), self.n_exponent)
        }


# In[4]:


m = ModExp(g=7, mod=15)
for exponent in range(10):
    exp_out, x_out = m.call_classically(exponent=exponent)
    assert exp_out == exponent
    print(exponent, x_out)

