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
from typing import Protocol, Sequence, Union, Any, Dict

from qualtran import Bloq, BloqBuilder, Register, Signature
from qualtran._infra.composite_bloq import QVarT
from qualtran._infra.quantum_graph import _QVar


class _TracingBloqFuncT(Protocol):
    def __call__(self, bb: 'BloqBuilder', *args: Any, **kwargs: Any) -> Dict[str, Any]: ...


class _TracingBloqIntermediate:
    def __init__(self, func):
        self.func = func

    def _prep(self, **kwargs):
        bb = BloqBuilder()
        soqs = {}
        classical_kwargs = {}

        # TODO: inspect.signature.bind
        for k, v in kwargs.items():
            # v is either a qvar or a register ... they both have dtype
            if isinstance(v, Register):
                soqs[k] = bb.in_register(name=k, dtype=v.dtype)
            elif isinstance(v, _QVar):
                soqs[k] = bb.in_register(name=k, dtype=v.dtype)
            else:
                classical_kwargs[k] = v

        soqs = self.func(bb, **classical_kwargs, **soqs)
        return bb.finalize(**soqs), set(soqs.keys())

    def make_from_signature(self, signature: 'Signature', **classical_kwargs):
        bb, soqs = BloqBuilder.from_signature(signature)
        soqs = self.func(bb, **classical_kwargs, **soqs)
        return bb.finalize(**soqs)

    def make(self, **kwargs):
        bloq, _ = self._prep(**kwargs)
        return bloq

    def __call__(self, bb: 'BloqBuilder', /, **kwargs: Any):
        # Note: no return type annotation (same as bb.add())
        bloq, soqnames = self._prep(**kwargs)
        return bb.add(bloq, **{k: v for k, v in kwargs.items() if k in soqnames})


def bloq_compile(func: _TracingBloqFuncT) -> _TracingBloqIntermediate:
    return _TracingBloqIntermediate(func)
