#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard imports
from typing import List, Dict, Set
import attrs
import sympy

import qualtran.testing as qlt_testing
from qualtran import (
    Bloq, BloqBuilder,
    QBit, QAny, QInt, QUInt,
    Register, Side, Signature,
)

from qualtran.drawing import show_bloq


# In[2]:


@attrs.frozen  # Makes an immutable dataclass where class attributes are listed per below.
class ModExp(Bloq):
    """Perform modular exponentiation `x = g^exponent mod p`.

    x and exponent are quantum variables; g and the modulus are classical constants.

    For simplicity, let's assume
        - a 4-bit exponent register.
        - a 32-bit "x" register.
    """

    g: int
    mod: int

    @property
    def signature(self) -> 'Signature':
        return Signature([
            # A bloq's signature is a list of its input and output registers.
            # See the API reference docs for Signature and Register:
            # https://qualtran.readthedocs.io/en/latest/reference/qualtran/Signature.html
            # https://qualtran.readthedocs.io/en/latest/reference/qualtran/Register.html
            #
            # and fill in this method with the list of registers according to the bloq's docstring.
            # ! -----------------------------
            Register('exponent', QUInt(bitsize=4)),
            Register('x', QUInt(bitsize=32))
            # ! -----------------------------
        ])


# In[3]:


mod_exp = ModExp(g=7, mod=15)
show_bloq(mod_exp)

