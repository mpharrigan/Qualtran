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
import inspect
import sys
from typing import Type

import qualtran as qlt
import qualtran.bloqs.arithmetic
import qualtran.bloqs.basic_gates
import qualtran.bloqs.bookkeeping
import qualtran.bloqs.mcmt

_CALLED = set()
_DTYPED = set()


def with_bb(bb, BT):
    def call(*args, **kwargs):
        b = BT.from_soqs(*args, **kwargs)
        return bb.add(b, **kwargs)

    return call


def with_bb_type(bb: qlt.BloqBuilder):
    def call(reg_name: str, dtype):
        return bb.add_register_from_dtype(reg_name, dtype)

    return call


def mock_bloq(bloq_cls: Type[qlt.Bloq], n_out_soqs):
    def _inner(*args, **kwargs):
        global _CALLED
        _CALLED.add(bloq_cls)
        if n_out_soqs > 1:
            return [None] * n_out_soqs
        elif n_out_soqs == 1:
            return None
        else:
            return

    _inner.is_mock_bloq = True
    _inner.bloq_cls = bloq_cls

    return _inner


Xor = mock_bloq(qualtran.bloqs.arithmetic.Xor, 2)
Split = mock_bloq(qualtran.bloqs.bookkeeping.Split, 1)
ZeroState = mock_bloq(qualtran.bloqs.basic_gates.ZeroState, 0)
MultiControlX = mock_bloq(qualtran.bloqs.mcmt.MultiControlX, 2)
Join = mock_bloq(qualtran.bloqs.bookkeeping.Join, 1)


def mock_dtype(dtype_cls: Type[qlt.QCDType]):
    def _inner(reg_name, dtype):
        return reg_name

    return _inner


In = mock_dtype(qlt.QUInt)


if __name__ == '__main__':
    main()
