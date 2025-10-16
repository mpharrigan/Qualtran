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
import inspect
from typing import Any, Dict, Protocol, Sequence, Union

from qualtran import Bloq, BloqBuilder, Register, Signature
from qualtran._infra.composite_bloq import QVarT
from qualtran._infra.quantum_graph import _QVar


class _TracingBloqFuncT(Protocol):
    def __call__(self, bb: 'BloqBuilder', *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """the structure of the function.

        During normal operation
         - a `bb: BloqBuilder` will be passed positionally as the first argument.
         - all other quantum and classical args and kwargs will be bound according to the Python
           rules and passed to the function by keyword. There are no additional positional-only
           arguments allowed (other than optionally `bb`).

        During `make`
         - a `bb: BloqBuilder` will be passed positionally as the first argument.
         - positional arguments will the be passed
         - any provided keyword arguments will be merged with a dictionary of initial quantum
           variables constructed by this method, and passed by keyword. An error is raised
           if a keyworkd argument is provided that interferes with this.

         - During normal operation, all quantum and classical arguments will be
         - Quantum arguments will always be passed by keyword, so you should probably put them last.
        """


class _TracingBloqIntermediate:
    def __init__(self, func: _TracingBloqFuncT):
        self.func: _TracingBloqFuncT = func

    @property
    def name(self) -> str:
        return self.func.__name__

    @property
    def pkg(self) -> str:
        return self.func.__module__

    def _prep_qstackframe(self, *args, **kwargs):
        bb = BloqBuilder(bloq_name=self.name, bloq_pkg_name=self.pkg)
        qkwargs = {}
        ckwargs = {}

        for k, v in inspect.signature(self.func).bind_partial(*args, **kwargs).arguments.items():
            if isinstance(v, _QVar):
                qkwargs[k] = bb.in_register(name=k, dtype=v.dtype)
            else:
                ckwargs[k] = v

        out_qvars = self.func(bb, **ckwargs, **qkwargs)
        if not isinstance(out_qvars, dict):
            raise ValueError(
                f"{self.name} is expected to return a dictionary mapping "
                f"output register name to output quantum variable."
            )
        cbloq = bb.finalize(**out_qvars)
        return cbloq, set(qkwargs.keys()), set(out_qvars.keys())

    def make(self, signature: 'Signature', *classical_args, **classical_kwargs):
        bb, soqs = BloqBuilder.from_signature(
            signature, bloq_name=self.name, bloq_pkg_name=self.pkg
        )

        dupes = set(classical_kwargs.keys()) & set(soqs.keys())
        if dupes:
            raise ValueError(
                f"`make` called with keyword arguments that shadow quantum "
                "register names: {dupes}. Please do not provide quantum variables "
                "when calling `make`."
            )

        kwargs = classical_kwargs | soqs
        soqs = self.func(bb, *classical_args, **kwargs)
        return bb.finalize(**soqs)

    def __call__(self, bb: 'BloqBuilder', /, *args, **kwargs):
        bloq, in_soqnames, _ = self._prep_qstackframe(*args, **kwargs)
        return bb.add(bloq, **{k: v for k, v in kwargs.items() if k in in_soqnames})

    def adjoint(self, bb: 'BloqBuilder', /, *args, **kwargs):
        bloq, in_soqnames, out_soqnames = self._prep_qstackframe(*args, **kwargs)
        return bb.add(bloq.adjoint(), **{k: v for k, v in kwargs.items() if k in out_soqnames})


def bloq_compile(func: _TracingBloqFuncT) -> _TracingBloqIntermediate:
    return _TracingBloqIntermediate(func)
