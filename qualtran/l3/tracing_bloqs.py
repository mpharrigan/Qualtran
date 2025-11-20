#  Copyright 2025 Google LLC
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

# TODO: These should go via Bloq.qcall and any convenience functions should live in the bloqs
#       library


def c_bitwise_not(ctrl, x):
    from qualtran.bloqs.arithmetic import BitwiseNot as _BitwiseNot

    bb = x.bb
    return bb.add(_BitwiseNot(dtype=x.dtype).controlled(), ctrl=ctrl, x=x)


def bitwise_not(x):
    from qualtran.bloqs.arithmetic import BitwiseNot as _BitwiseNot

    bb = x.bb
    return bb.add(_BitwiseNot(dtype=x.dtype), x=x)


def add_k(x, *, k: int):
    from qualtran.bloqs.arithmetic import AddK as _AddK

    bb = x.bb
    return bb.add(_AddK(dtype=x.dtype, k=k), x=x)
